"""Abstract base class for all variable builders."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional, Dict
import logging

import rasterio
import geopandas as gpd

from ..utils.checkpoint import output_exists, log_cached
from ..utils.logger import setup_logger
from ..core.raster import RasterManager

class VarBuilder(ABC):
    """
    Abstract base for all variable builders.

    Lifecycle for every builder:
        raw source  →  reproject/resample  →  clip to Essex  →  align to reference grid

    Shared helpers (defined here, used by all subclasses):
        _boundary()          - load Essex boundary as a GeoDataFrame (cached after first call)
        _clip_to_boundary()  - clip any raster path to Essex and write to an output path
        _to_reference()      - run the full reproject → clip → align pipeline in one call

    Checkpointing:
        If force_recompute is False (default) and all expected outputs already exist,
        process() returns the cached paths immediately.

    ref_path is optional because TopographyBuilder produces the reference raster itself
    rather than consuming one.
    """

    def __init__(
        self,
        config: dict,
        ref_path: Optional[Union[str, Path]] = None,
    ):
        self.config = config
        self.output_dir = Path(config["base"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ref_path = Path(ref_path) if ref_path else None
        self.boundary_path = Path(config["base"]["boundary_shapefile"])
        self.force_recompute = config["processing"]["force_recompute"]

        self._boundary_cache: Optional[gpd.GeoDataFrame] = None

        self.logger = setup_logger(
            log_file = self.config["logging"]["log_path"],
            level = self.config["logging"]["level"]
        )

    @abstractmethod
    def process(self, **kwargs) -> Dict[str, Path]:
        """Transform raw source data into an aligned 30 m GeoTIFF."""
        pass

    def _boundary(self) -> gpd.GeoDataFrame:
        """Return the Essex boundary in BNG (EPSG:27700), loading once and caching."""
        if self._boundary_cache is None:
            self._boundary_cache = (
                gpd.read_file(self.boundary_path).to_crs("EPSG:27700")
            )
        return self._boundary_cache

    def _clip_to_boundary(
        self,
        src_path: Union[str, Path],
        dst_path: Union[str, Path],
    ) -> Path:
        """
        Clip *src_path* to the Essex boundary and write the result to *dst_path*.

        The boundary is reprojected to match the source raster's CRS before
        clipping, so this works correctly regardless of whether the source is
        in BNG, WGS84, UTM, or any other CRS. The output inherits the source
        raster's CRS — reprojection to BNG happens in the subsequent
        align_to_reference call.
        """
        dst_path = Path(dst_path)

        with rasterio.open(src_path) as src:
            src_crs = src.crs

        boundary_in_src_crs = self._boundary().to_crs(src_crs)
        RasterManager.clip_to_boundary(src_path, boundary_in_src_crs, dst_path)
        return dst_path

    def _to_reference(
        self,
        src_path: Union[str, Path],
        dst_path: Union[str, Path],
        *,
        tmp_stem: Optional[str] = None,
    ) -> Path:
        """
        Bring *src_path* fully into the reference grid in three steps:

            1. align to reference grid (reproject + resample to 30 m BNG)
            2. clip to Essex boundary  (removes data outside study area)

        A temporary intermediate file is written to output_dir and deleted
        after the final alignment succeeds.

        Args:
            src_path:  Input raster (any CRS / resolution).
            dst_path:  Final output path, aligned to the reference grid.
            tmp_stem:  Optional name stem for the intermediate clip file.
                       Defaults to `_tmp_clip_<dst_path.stem>`.

        Returns:
            dst_path, as a resolved Path.
        """
        self._validate_reference()

        src_path = Path(src_path)
        dst_path = Path(dst_path)
        stem = tmp_stem or f"_tmp_clip_{dst_path.stem}"
        tmp_clip = self.output_dir / f"{stem}.tif"

        try:
            RasterManager.align_to_reference(src_path, self.ref_path, tmp_clip)
            self._clip_to_boundary(tmp_clip, dst_path)
        finally:
            if tmp_clip.exists():
                tmp_clip.unlink()

        return dst_path

    def _check_cache(
        self,
        step_name: str,
        output_paths: Dict[str, Path],
    ) -> bool:
        if self.force_recompute:
            return False
        if output_exists(output_paths):
            log_cached(step_name, output_paths)
            return True
        return False

    def _validate_reference(self) -> None:
        """Raise if the reference raster has not been set or does not exist."""
        if self.ref_path is None:
            raise ValueError("ref_path has not been set on this builder.")
        if not self.ref_path.exists():
            raise FileNotFoundError(f"Reference raster not found: {self.ref_path}")
        
    @staticmethod
    def _seasonal_name(base_name: str, season: Optional[str]) -> str:
        """Return '<base_name>_<season>' when season is set, else '<base_name>'."""
        return f"{base_name}_{season}" if season else base_name