"""Topography features: elevation, slope, aspect."""

from pathlib import Path
from typing import Union, List, Tuple
import numpy as np
import rasterio
from rasterio.merge import merge

from ..core.base import VarBuilder
from ..core.raster import RasterManager
import logging

logger = logging.getLogger(__name__)

class TopographyBuilder(VarBuilder):
    """Build elevation, slope and aspect from SRTM DEM."""

    def process(self) -> dict:
        """
        Process SRTM tiles into elevation, slope, aspect.

        Args:
            data_config: Configuration parameters for data source

        Returns: 
            dict with keys 'elevation', 'slope', 'aspect' saved to output paths 
        """
        data_config = self.config["data_sources"]["srtm"]

        output_paths = {
            "elevation": self.output_dir / "topo_elevation.tif",
            "slope": self.output_dir / "topo_slope.tif",
            "aspect": self.output_dir / "topo_aspect.tif",
        }

        if self._check_cache("TopographyBuilder", output_paths):
            return output_paths

        #############################################
        
        data_dir = Path(data_config["data_dir"])
        tiles = data_config["tiles"]
        srtm_tile_paths = [data_dir / tile for tile in tiles]

        merged_path = self.output_dir / "_merged_srtm.tif"
        self._merge_tiles(srtm_tile_paths, merged_path)

        bng_path = self.output_dir / "_srtm_bng.tif"
        RasterManager.reproject_and_resample(
            merged_path,
            bng_path,
            target_crs = "EPSG:27700",
            target_res = 30.0
        )

        clipped_path = self.output_dir / "topo_elevation.tif"
        RasterManager.clip_to_boundary(
            bng_path,
            self._load_boundary(),
            clipped_path
        )

        slope_path, aspect_path = self._derive_slope_and_aspect(clipped_path)

        merged_path.unlink()
        bng_path.unlink()

        logger.info("Topography features complete")
        return output_paths

    def _merge_tiles(self, tile_paths: List[Path], out_path: Path) -> None:
        """Merge multiple SRTM tiles into one."""
        datasets = [rasterio.open(path) for path in tile_paths]
        merged, merged_transform = merge(datasets)
        meta = datasets[0].meta.copy()
        meta.update({
            "height": merged.shape[1],
            "width": merged.shape[2],
            "transform": merged_transform
        })

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(merged)
        for ds in datasets:
            ds.close()

        logger.info(f"Merged {len(tile_paths)} SRTM tiles to {out_path}")

    def _derive_slope_and_aspect(self, dem_path: Path) -> Tuple[Path, Path]:
        """Compute slope from DEM using richdem."""
        with rasterio.open(dem_path) as src:
            elevation = src.read(1).astype("float32")
            meta = src.meta.copy()
            meta.update({"dtype": "float32", "nodata": np.nan})
            cellsize = src.res[0]

        # Gradients
        dy, dx = np.gradient(elevation, cellsize, cellsize)

        # Slope
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

        # Aspect
        aspect = np.degrees(np.arctan2(-dy, dx))
        aspect = (aspect + 360) % 360

        nodata_mask = np.isnan(elevation)
        slope[nodata_mask] = np.nan
        aspect[nodata_mask] = np.nan

        slope_path = self.output_dir / "topo_slope.tif"
        aspect_path = self.output_dir / "topo_aspect.tif"

        RasterManager.write(
            np.array(slope).astype("float32"),
            slope_path,
            meta
        )

        RasterManager.write(
            np.array(aspect).astype("float32"),
            aspect_path,
            meta
        )

        return slope_path, aspect_path

    def _load_boundary(self):
        import geopandas as gpd
        return gpd.read_file(self.boundary_path).to_crs("EPSG:27700")

        

