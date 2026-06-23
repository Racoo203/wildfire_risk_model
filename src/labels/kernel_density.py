"""
Density-based fire labelling: bin fire points into the grid, smooth
with a Gaussian filter, and classify into 4 susceptibility classes.

Uses a binning + convolution approach. 
"""

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.enums import MergeAlg
from scipy.ndimage import gaussian_filter
from pathlib import Path
from typing import Union, Optional, Tuple
import logging

from ..core.base import VarBuilder

logger = logging.getLogger(__name__)

class KernelDensityClassifier(VarBuilder):
    """
    Compute a smoothed fire-density surface and classify into 4 suceptibility
    classes, per season.
    """

    def process(self):
        return

    def compute_kde(
        self,
        fire_gdf,
        season: Optional[str] = None,
        sigma_cell: float = 3.0,
    ) -> np.ndarray:
        """
        Bin fires into grid, then convolve with Gaussian Kernel.

        sigma_cells: std dev of Gaussian in pixels (3 pixels = 90 m at 30m resolution)
        """
        out_paths = {
            "density": self.output_dir / f"{self._seasonal_name("fire_density", season)}.tif"
        }

        if self._check_cache(f"KDClassifier[{season}]", out_paths):
            logger.info(f"[CACHED] Fire density ({season}) already exists")
            with rasterio.open(out_paths["density"]) as src:
                return src.read(1)

        with rasterio.open(self.ref_path) as ref:
            height = ref.height
            width = ref.width
            transform = ref.transform
            meta = ref.meta.copy()
            land_mask = ref.read(1)
        
        # print(fire_gdf.geometry.type)
        fire_x = fire_gdf.geometry.x.values
        fire_y = fire_gdf.geometry.y.values

        if len(fire_x) < 5:
            raise ValueError(f"[{season}] Too few fire points: {len(fire_x)}")

        shapes = [
            ({"type": "Point", "coordinates": (x,y)}, 1) 
            for x, y in zip(fire_x, fire_y)
        ]

        # shapes = [(geom, 1) for geom in fire_gdf.geometry]

        fire_counts = rasterize(
            shapes,
            out_shape = (height, width),
            transform = transform,
            fill = 0,
            dtype = "float32",
            merge_alg = MergeAlg.add,
        )

        logger.info(f"[{season}] Binned {fire_counts.sum():.0f} fires into {int((fire_counts > 0).sum()):,} cells")
        
        density = gaussian_filter(fire_counts, sigma = sigma_cell)
        density[np.isnan(land_mask)] = np.nan

        meta.update({"dtype": "float32", "nodata": np.nan})

        with rasterio.open(out_paths["density"], "w", **meta) as dst:
            dst.write(density[np.newaxis, :, :])

        return density

    def classify(
        self,
        density: np.ndarray,
        season: Optional[str] = None,
    ) -> np.ndarray:
        out_paths = {
            "risk_labels": self.output_dir / f"{self._seasonal_name('risk_labels', season)}.tif"
        }

        if self._check_cache(f"KDClassifier[{season}]", out_paths):
            logger.info(f"[CACHED] Classification ({season}) already exists")
            with rasterio.open(out_paths["risk_labels"]) as src:
                return src.read(1)

        # valid = density[(density > 0) & (~np.isnan(density))].copy()
        valid = density[(density > 0)].copy()

        if len(valid) == 0:
            raise ValueError(f"[{season}] No positive density values to classify")

        p_low, p_mid, p_high = np.percentile(valid, self.config["labels"]["percentiles"])

        labels = np.full(density.shape, np.nan, dtype = "float32")
        # mask = ~np.isnan(density)
        mask = density > 0


        labels[mask & (density < p_low)] = 0
        labels[mask & (density >= p_low) & (density < p_mid)] = 1
        labels[mask & (density >= p_mid) & (density < p_high)] = 2
        labels[mask & (density >= p_high)] = 3

        counts = {
            int(c): int(np.sum(labels == c)) for c in [0, 1, 2, 3]
        }

        # print(counts)
        
        with rasterio.open(self.ref_path) as ref:
            meta = ref.meta.copy()
        meta.update({"dtype": "float32", "nodata": np.nan})
        with rasterio.open(out_paths["risk_labels"], "w", **meta) as dst:
            dst.write(labels[np.newaxis, :, :])

        return labels
    