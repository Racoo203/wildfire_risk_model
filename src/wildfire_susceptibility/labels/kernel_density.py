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
import pickle

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
        fitted: Optional[dict] = None,
    ):
        """
        Bin continuous fire density into 4 ordinal susceptibility classes.

        CRITICAL: `fitted` lets a caller reuse a classifier already fit on train
        density, so test labels are drawn from the SAME class boundaries as
        train rather than a fresh boundary refit on test's own (different)
        density distribution. Passing `fitted=None` fits a new classifier and
        returns the fit artifact so the caller can pass it back in for the
        paired split.

        Returns:
            (labels, fit_artifact) — fit_artifact is a dict with at least a
            "thresholds" key (used only for logging); for classify_method="gmm"
            it also holds the fitted GaussianMixture + class-index remap needed
            to reproduce identical class assignments without refitting.
        """
        method = method or self.config["labels"].get("density_method", "convolution")
        classify_method = classify_method or self.config["labels"].get("classify_method", "percentile")

        out_paths = {
            "risk_labels": self.output_dir
            / f"{self._seasonal_name(f'risk_labels_{method}_{classify_method}', season)}.tif"
        }
        fit_path = self.output_dir / f"{self._seasonal_name(f'risk_labels_{method}_{classify_method}_fit', season)}.pkl"

        # Cache is only valid if BOTH the raster and its fit artifact exist —
        # this also auto-invalidates any pre-existing cache from before this
        # fix, so you don't need to manually delete stale label rasters.
        if self._check_cache(f"KDClassifier[{method}][{classify_method}][{season}]", {**out_paths, "_fit": fit_path}):
            with rasterio.open(out_paths["risk_labels"]) as src:
                labels = src.read(1)
            with open(fit_path, "rb") as f:
                fit_artifact = pickle.load(f)
            return labels, fit_artifact

        zero_threshold = self.config["labels"].get("kde_zero_threshold", 0.0) if method == "kde" else 0.0
        valid = density[density > zero_threshold].copy()
        valid = valid[~np.isnan(valid)]

        if len(valid) == 0:
            raise ValueError(f"[{season}] No positive density values to classify")

        mask = (density > zero_threshold) & (~np.isnan(density))

        if fitted is not None:
            labels = self._apply_fitted(density, mask, fitted, classify_method)
            fit_artifact = fitted
            self.logger.info(f"[{season}][{method}][{classify_method}] applied FROZEN fit from paired split (no refit).")
        else:
            dispatch = {
                "percentile": self._classify_percentile,
                "jenks": self._classify_jenks,
                "gmm": self._classify_gmm,
            }
            if classify_method not in dispatch:
                raise ValueError(f"Unknown classify_method '{classify_method}' (use one of {list(dispatch)})")

            labels, fit_artifact = dispatch[classify_method](density, valid, mask)

        n_total = int(np.isfinite(density).sum())
        counts = {int(c): int(np.sum(labels == c)) for c in [0, 1, 2, 3]}
        n_nan_domain = int(np.isnan(density).sum())

        self.logger.info(
            f"[{season}][{method}][{classify_method}] class counts: {counts} | "
            f"zero-density (unlabelled) cells: {n_total - sum(counts.values())} | "
            f"out-of-domain NaN cells: {n_nan_domain} | "
            f"thresholds: {np.round(fit_artifact['thresholds'], 4).tolist()}"
        )
        for c, n in counts.items():
            if n == 0:
                self.logger.warning(f"[{season}][{method}][{classify_method}] class {c} has 0 samples!")

        with rasterio.open(self.ref_path) as ref:
            meta = ref.meta.copy()
        meta.update({"dtype": "float32", "nodata": np.nan})
        with rasterio.open(out_paths["risk_labels"], "w", **meta) as dst:
            dst.write(labels[np.newaxis, :, :])

        with open(fit_path, "wb") as f:
            pickle.dump(fit_artifact, f)

        return labels, fit_artifact
    
    def _apply_fitted(self, density, mask, fitted, classify_method):
        """Apply an already-fit classifier (from the paired train split) to a
        new density array — never refits, so class boundaries stay identical
        between train and test."""
        if classify_method in ("percentile", "jenks"):
            return self._apply_thresholds(density, mask, fitted["thresholds"])
        elif classify_method == "gmm":
            gmm, remap = fitted["gmm"], fitted["remap"]
            valid_vals = density[mask]
            raw = gmm.predict(valid_vals.reshape(-1, 1))
            ordinal = np.vectorize(remap.get)(raw).astype("float32")
            labels = np.full(density.shape, np.nan, dtype="float32")
            labels[mask] = ordinal
            return labels
        else:
            raise ValueError(f"Unknown classify_method '{classify_method}' for frozen-fit application")

    # --- classify_method implementations -------------------------------

    def _classify_percentile(self, density, valid, mask):
        p_low, p_mid, p_high = np.percentile(valid, self.config["labels"]["percentiles"])
        thresholds = np.array([p_low, p_mid, p_high])
        labels = self._apply_thresholds(density, mask, thresholds)
        return labels, {"thresholds": thresholds}

    def _classify_jenks(self, density, valid, mask):
        n_classes = self.config["labels"].get("jenks_n_classes", 4)
        max_sample = self.config["labels"].get("jenks_max_sample", 200_000)
        sample = self._subsample(valid, max_sample)

        breaks = jenkspy.jenks_breaks(sample.tolist(), n_classes=n_classes)
        thresholds = np.array(breaks[1:-1])

        labels = self._apply_thresholds(density, mask, thresholds)
        return labels, {"thresholds": thresholds}

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

        valid_vals = density[mask]
        raw = gmm.predict(valid_vals.reshape(-1, 1))
        ordinal = np.vectorize(remap.get)(raw).astype("float32")

        labels = np.full(density.shape, np.nan, dtype="float32")
        labels[mask] = ordinal

        test_grid = np.linspace(valid.min(), valid.max(), 10000).reshape(-1, 1)
        test_preds = np.vectorize(remap.get)(gmm.predict(test_grid))

        thresholds = []
        for c in range(n_components - 1):
            transition_idx = np.where(test_preds == c)[0][-1]
            thresholds.append(test_grid[transition_idx][0])

        fit_artifact = {"thresholds": np.array(thresholds), "gmm": gmm, "remap": remap}
        return labels, fit_artifact

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