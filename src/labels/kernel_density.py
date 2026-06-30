"""
Density-based fire labelling: bin fire points into the grid, smooth
with a Gaussian filter, and classify into 4 susceptibility classes.

Uses a binning + convolution approach. 
"""

import numpy as np
import matplotlib.pyplot as plt

import rasterio
from rasterio.features import rasterize
from rasterio.enums import MergeAlg

from scipy.ndimage import gaussian_filter
from sklearn.neighbors import KernelDensity

from pathlib import Path
from typing import Union, Optional, Tuple
import logging

from ..core.base import VarBuilder

logger = logging.getLogger(__name__)

class KernelDensityClassifier(VarBuilder):

    def process(self):
        return

    def compute_density(
            self, 
            fire_gdf, 
            season=None, 
            method: Optional[str] = None
        ) -> np.ndarray:

        method = method or self.config["labels"].get("density_method", "convolution")
        if method == "convolution":
            return self._density_convolution(fire_gdf, season=season)
        elif method == "kde":
            return self._density_kde(fire_gdf, season=season)
        else:
            raise ValueError(f"Unknown density_method '{method}' (use 'convolution' or 'kde')")

    # --- existing approach, renamed -----------------------------------
    def _density_convolution(self, fire_gdf, season=None, sigma_cell: float = 3.0) -> np.ndarray:
        out_paths = {"density": self.output_dir / f"{self._seasonal_name('fire_density_conv', season)}.tif"}
        if self._check_cache(f"KDClassifier[conv][{season}]", out_paths):
            with rasterio.open(out_paths["density"]) as src:
                return src.read(1)

        with rasterio.open(self.ref_path) as ref:
            height, width, transform = ref.height, ref.width, ref.transform
            meta = ref.meta.copy()
            land_mask = ref.read(1)

        fire_x, fire_y = fire_gdf.geometry.x.values, fire_gdf.geometry.y.values
        if len(fire_x) < 5:
            raise ValueError(f"[{season}] Too few fire points: {len(fire_x)}")

        shapes = [({"type": "Point", "coordinates": (x, y)}, 1) for x, y in zip(fire_x, fire_y)]
        fire_counts = rasterize(shapes, out_shape=(height, width), transform=transform,
                                 fill=0, dtype="float32", merge_alg=MergeAlg.add)

        density = gaussian_filter(fire_counts, sigma=sigma_cell)
        density[np.isnan(land_mask)] = np.nan

        meta.update({"dtype": "float32", "nodata": np.nan})
        with rasterio.open(out_paths["density"], "w", **meta) as dst:
            dst.write(density[np.newaxis, :, :])
        return density

    # --- new: proper Gaussian KDE over fire point coordinates ---------
    def _density_kde(self, fire_gdf, season=None) -> np.ndarray:
        out_paths = {"density": self.output_dir / f"{self._seasonal_name('fire_density_kde', season)}.tif"}
        if self._check_cache(f"KDClassifier[kde][{season}]", out_paths):
            with rasterio.open(out_paths["density"]) as src:
                return src.read(1)

        with rasterio.open(self.ref_path) as ref:
            height, width, transform = ref.height, ref.width, ref.transform
            meta = ref.meta.copy()
            land_mask = ref.read(1)

        coords = np.column_stack([fire_gdf.geometry.x.values, fire_gdf.geometry.y.values])
        if len(coords) < 5:
            raise ValueError(f"[{season}] Too few fire points: {len(coords)}")

        bandwidth = self.config["labels"]["kde_bandwidth_m"]
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
        kde.fit(coords)

        # Evaluate only on valid (non-NaN) land-mask cells, chunked to bound memory.
        valid_rows, valid_cols = np.where(~np.isnan(land_mask))
        xs, ys = rasterio.transform.xy(transform, valid_rows, valid_cols)
        grid_coords = np.column_stack([xs, ys])

        density_flat = np.full(grid_coords.shape[0], np.nan, dtype="float32")
        chunk = 200_000
        for i in range(0, grid_coords.shape[0], chunk):
            density_flat[i:i+chunk] = np.exp(kde.score_samples(grid_coords[i:i+chunk]))

        density = np.full((height, width), np.nan, dtype="float32")
        density[valid_rows, valid_cols] = density_flat

        meta.update({"dtype": "float32", "nodata": np.nan})
        with rasterio.open(out_paths["density"], "w", **meta) as dst:
            dst.write(density[np.newaxis, :, :])
        return density

    def classify(self, density: np.ndarray, season=None, method: Optional[str] = None) -> np.ndarray:
        method = method or self.config["labels"].get("density_method", "convolution")
        out_paths = {"risk_labels": self.output_dir / f"{self._seasonal_name(f'risk_labels_{method}', season)}.tif"}

        if self._check_cache(f"KDClassifier[{method}][{season}]", out_paths):
            with rasterio.open(out_paths["risk_labels"]) as src:
                return src.read(1)

        zero_threshold = self.config["labels"].get("kde_zero_threshold", 0.0) if method == "kde" else 0.0
        valid = density[density > zero_threshold].copy()
        valid = valid[~np.isnan(valid)]

        if len(valid) == 0:
            raise ValueError(f"[{season}] No positive density values to classify")

        p_low, p_mid, p_high = np.percentile(valid, self.config["labels"]["percentiles"])

        labels = np.full(density.shape, np.nan, dtype="float32")
        mask = (density > zero_threshold) & (~np.isnan(density))

        labels[mask & (density < p_low)] = 0
        labels[mask & (density >= p_low) & (density < p_mid)] = 1
        labels[mask & (density >= p_mid) & (density < p_high)] = 2
        labels[mask & (density >= p_high)] = 3

        n_total = np.isfinite(density).sum()
        counts = {int(c): int(np.sum(labels == c)) for c in [0, 1, 2, 3]}
        n_nan_domain = int(np.isnan(density).sum())

        logger.info(
            f"[{season}][{method}] class counts: {counts} | "
            f"zero-density (unlabelled) cells: {n_total - sum(counts.values())} | "
            f"out-of-domain NaN cells: {n_nan_domain}"
        )
        for c, n in counts.items():
            if n == 0:
                logger.warning(f"[{season}][{method}] class {c} has 0 samples!")

        self._plot_diagnostics(valid, p_low, p_mid, p_high, season, method)

        with rasterio.open(self.ref_path) as ref:
            meta = ref.meta.copy()
        meta.update({"dtype": "float32", "nodata": np.nan})
        with rasterio.open(out_paths["risk_labels"], "w", **meta) as dst:
            dst.write(labels[np.newaxis, :, :])

        return labels

    def _plot_diagnostics(self, valid_density, p_low, p_mid, p_high, season, method):
        figures_dir = Path(self.config["base"]["figures_dir"])
        figures_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(valid_density, bins=80, color="steelblue", alpha=0.8)
        for val, label, color in [(p_low, "60th", "orange"), (p_mid, "75th", "red"), (p_high, "90th", "darkred")]:
            ax.axvline(val, color=color, linestyle="--", label=f"{label} pct = {val:.4g}")
        ax.set_title(f"Fire density distribution — {season} ({method})")
        ax.set_xlabel("Density")
        ax.set_ylabel("Pixel count")
        ax.legend()
        fig.tight_layout()
        out_path = figures_dir / f"density_dist_{season}_{method}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info(f"[{season}][{method}] diagnostic plot -> {out_path}")