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

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

import jenkspy

from ..core.base import VarBuilder

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

    def classify(
        self,
        density: np.ndarray,
        season=None,
        method: Optional[str] = None,
        classify_method: Optional[str] = None,
    ) -> np.ndarray:
        """
        Bin continuous fire density into 4 ordinal susceptibility classes.

        Zero-density pixels are always left unlabelled (NaN) regardless of
        classify_method — they represent "no fire signal", not "class 0 / low
        risk", and mixing the two would bias every downstream model toward
        treating absence-of-evidence as evidence-of-low-risk.

        classify_method controls how the 3 breakpoints between the 4 classes
        are chosen, and is deliberately decoupled from density_method (the
        density surface can be built by convolution or KDE; either can then
        be classified by any of the three methods below):

            "percentile" — fixed percentiles of the paper's percentiles config. 
                            Simple, but the exact cut points are not specified 
                            in the source paper; treat this as a baseline, 
                            not a replication.
            "jenks"      — Fisher-Jenks natural breaks, minimizing within-class
                            variance. Matches ArcGIS's default classification
                            method, which the source paper used for its GIS
                            workflow (tool named, method not specified).
            "gmm"        — 4-component Gaussian Mixture Model fit on the
                            non-zero density values; components ordered by
                            mean and mapped to ordinal classes. Fully
                            data-driven, no fixed percentiles at all.
        """
        method = method or self.config["labels"].get("density_method", "convolution")
        classify_method = classify_method or self.config["labels"].get("classify_method", "percentile")

        out_paths = {
            "risk_labels": self.output_dir
            / f"{self._seasonal_name(f'risk_labels_{method}_{classify_method}', season)}.tif"
        }

        if self._check_cache(f"KDClassifier[{method}][{classify_method}][{season}]", out_paths):
            with rasterio.open(out_paths["risk_labels"]) as src:
                return src.read(1)

        zero_threshold = self.config["labels"].get("kde_zero_threshold", 0.0) if method == "kde" else 0.0
        valid = density[density > zero_threshold].copy()
        valid = valid[~np.isnan(valid)]

        if len(valid) == 0:
            raise ValueError(f"[{season}] No positive density values to classify")

        mask = (density > zero_threshold) & (~np.isnan(density))

        dispatch = {
            "percentile": self._classify_percentile,
            "jenks": self._classify_jenks,
            "gmm": self._classify_gmm,
        }
        if classify_method not in dispatch:
            raise ValueError(
                f"Unknown classify_method '{classify_method}' (use one of {list(dispatch)})"
            )

        labels, breakpoints = dispatch[classify_method](density, valid, mask)

        n_total = int(np.isfinite(density).sum())
        counts = {int(c): int(np.sum(labels == c)) for c in [0, 1, 2, 3]}
        n_nan_domain = int(np.isnan(density).sum())

        self.logger.info(
            f"[{season}][{method}][{classify_method}] class counts: {counts} | "
            f"zero-density (unlabelled) cells: {n_total - sum(counts.values())} | "
            f"out-of-domain NaN cells: {n_nan_domain} | "
            f"breakpoints: {np.round(breakpoints, 4).tolist()}"
        )
        for c, n in counts.items():
            if n == 0:
                self.logger.warning(f"[{season}][{method}][{classify_method}] class {c} has 0 samples!")

        self._plot_diagnostics(valid, breakpoints, season, method, classify_method)

        with rasterio.open(self.ref_path) as ref:
            meta = ref.meta.copy()
        meta.update({"dtype": "float32", "nodata": np.nan})
        with rasterio.open(out_paths["risk_labels"], "w", **meta) as dst:
            dst.write(labels[np.newaxis, :, :])

        return labels

    # --- classify_method implementations -------------------------------

    def _classify_percentile(self, density, valid, mask):
        """Fixed percentiles of non-zero density (paper does not specify exact cut points)."""
        p_low, p_mid, p_high = np.percentile(valid, self.config["labels"]["percentiles"])
        labels = self._apply_thresholds(density, mask, (p_low, p_mid, p_high))
        return labels, np.array([p_low, p_mid, p_high])

    def _classify_jenks(self, density, valid, mask):
        """Fisher-Jenks natural breaks — ArcGIS's default classification method."""

        n_classes = self.config["labels"].get("jenks_n_classes", 4)
        max_sample = self.config["labels"].get("jenks_max_sample", 200_000)
        sample = self._subsample(valid, max_sample)

        breaks = jenkspy.jenks_breaks(sample.tolist(), n_classes=n_classes)
        thresholds = np.array(breaks[1:-1])  # inner edges only, drop global min/max

        labels = self._apply_thresholds(density, mask, thresholds)
        return labels, thresholds

    def _classify_gmm(self, density, valid, mask):
        n_components = self.config["labels"].get("gmm_n_components", 4)
        max_sample = self.config["labels"].get("gmm_max_sample", 200_000)
        random_state = self.config["labels"].get("gmm_random_state", 42)
        sample = self._subsample(valid, max_sample)

        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(sample.reshape(-1, 1))

        means = gmm.means_.flatten()
        order = np.argsort(means)              
        remap = {old: new for new, old in enumerate(order)}

        # Predict the ordinal classes for the valid map pixels
        valid_vals = density[mask]
        raw = gmm.predict(valid_vals.reshape(-1, 1))
        ordinal = np.vectorize(remap.get)(raw).astype("float32")

        labels = np.full(density.shape, np.nan, dtype="float32")
        labels[mask] = ordinal

        # Create a fine linspace spanning the data range to find where predictions shift
        test_grid = np.linspace(valid.min(), valid.max(), 10000).reshape(-1, 1)
        test_preds = np.vectorize(remap.get)(gmm.predict(test_grid))
        
        thresholds = []
        for c in range(n_components - 1):
            # Find the last value belonging to class c before it transitions to c+1
            transition_idx = np.where(test_preds == c)[0][-1]
            thresholds.append(test_grid[transition_idx][0])

        return labels, np.array(thresholds)

    # --- shared helpers --------------------------------------------------

    @staticmethod
    def _subsample(values: np.ndarray, max_n: int) -> np.ndarray:
        """Randomly subsample large arrays for O(n log n)-or-worse fitting steps."""
        if len(values) <= max_n:
            return values
        return np.random.choice(values, max_n, replace=False)

    @staticmethod
    def _apply_thresholds(density: np.ndarray, mask: np.ndarray, thresholds) -> np.ndarray:
        """Bin masked density values into ordinal classes using ascending thresholds."""
        labels = np.full(density.shape, np.nan, dtype="float32")
        labels[mask] = np.digitize(density[mask], thresholds).astype("float32")
        return labels

    def _plot_diagnostics(self, valid_density, breakpoints, season, method, classify_method):
        figures_dir = Path(self.config["base"]["figures_dir"])
        figures_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(valid_density, bins=80, color="steelblue", alpha=0.8)

        line_label = "break" if classify_method != "gmm" else "component mean"
        colors = ["orange", "red", "darkred", "purple"]
        for i, val in enumerate(breakpoints):
            ax.axvline(val, color=colors[i % len(colors)], linestyle="--",
                       label=f"{line_label} {i + 1} = {val:.4g}")

        ax.set_title(f"Fire density distribution — {season} ({method} / {classify_method})")
        ax.set_xlabel("Density")
        ax.set_ylabel("Pixel count")
        ax.legend()
        fig.tight_layout()
        out_path = figures_dir / f"density_dist_{season}_{method}_{classify_method}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        self.logger.info(f"[{season}][{method}][{classify_method}] diagnostic plot -> {out_path}")