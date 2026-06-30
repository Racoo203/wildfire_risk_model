"""Topography features: elevation, slope, aspect."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.merge import merge

from ..core.base import VarBuilder
from ..core.raster import RasterManager
import logging

logger = logging.getLogger(__name__)


class TopographyBuilder(VarBuilder):
    """
    Build elevation, slope, and aspect from SRTM DEM tiles.

    TopographyBuilder is the only builder that *produces* the reference
    raster rather than consuming one; ref_path is therefore not required
    in its constructor. The clipped elevation raster becomes the reference
    grid for every other builder in the pipeline.

    Pipeline:
        merge SRTM tiles -> reproject to 30 m BNG -> clip to Essex -> derive slope & aspect from clipped DEM
    """

    def process(self) -> Dict[str, Path]:
        data_config = self.config["data_sources"]["srtm"]

        output_paths = {
            "elevation": self.output_dir / "topo_elevation.tif",
            "slope":     self.output_dir / "topo_slope.tif",
            "aspect":    self.output_dir / "topo_aspect.tif",
        }

        if self._check_cache("TopographyBuilder", output_paths):
            return output_paths

        data_dir = Path(data_config["data_dir"])
        tile_paths = [data_dir / tile for tile in data_config["tiles"]]

        merged_path = self.output_dir / "_tmp_srtm_merged.tif"
        elev_tmp_path = self.output_dir / "_tmp_elev.tif"
        aspect_tmp_path = self.output_dir / "_tmp_aspect.tif"
        slope_tmp_path = self.output_dir / "_tmp_slope.tif"
        
        self._merge_tiles(tile_paths, merged_path)
        RasterManager.reproject_and_resample(
            merged_path,
            elev_tmp_path,
            target_crs="EPSG:27700",
            target_res=30.0,
        )

        self._derive_slope_and_aspect(
            dem_path = elev_tmp_path,
            slope_path = slope_tmp_path,
            aspect_path = aspect_tmp_path,
        )

        self._clip_to_boundary(elev_tmp_path, output_paths["elevation"])
        self._clip_to_boundary(aspect_tmp_path, output_paths["aspect"])
        self._clip_to_boundary(slope_tmp_path, output_paths["slope"])

        merged_path.unlink()
        elev_tmp_path.unlink()
        aspect_tmp_path.unlink()
        slope_tmp_path.unlink()

        logger.info("Topography features complete.")
        return output_paths

    def _merge_tiles(self, tile_paths: List[Path], out_path: Path) -> None:
        """Mosaic multiple SRTM tiles into a single raster."""
        datasets = [rasterio.open(p) for p in tile_paths]
        try:
            merged, merged_transform = merge(datasets)
            meta = datasets[0].meta.copy()
            meta.update({
                "height": merged.shape[1],
                "width":  merged.shape[2],
                "transform": merged_transform,
            })
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(merged)
        finally:
            for ds in datasets:
                ds.close()

        logger.info(f"Merged {len(tile_paths)} SRTM tiles → {out_path.name}")

    def _derive_slope_and_aspect(
        self,
        dem_path: Path,
        slope_path: Path,
        aspect_path: Path,
    ) -> None:
        """
        Derive slope (degrees from horizontal) and aspect (degrees clockwise
        from north) from the clipped DEM using numpy gradient.

        NoData cells in the DEM propagate to both output rasters.
        """
        with rasterio.open(dem_path) as src:
            elevation = src.read(1).astype("float32")
            meta = src.meta.copy()
            meta.update({"dtype": "float32", "nodata": np.nan})
            cell_size = src.res[0]   # metres (square pixels assumed)

        dy, dx = np.gradient(elevation, cell_size, cell_size)

        slope  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        aspect = (np.degrees(np.arctan2(-dy, dx)) + 360) % 360

        nodata_mask = np.isnan(elevation)
        slope[nodata_mask]  = np.nan
        aspect[nodata_mask] = np.nan

        RasterManager.write(slope,  slope_path,  meta)
        RasterManager.write(aspect, aspect_path, meta)

        logger.info("Slope and aspect derived from clipped DEM.")