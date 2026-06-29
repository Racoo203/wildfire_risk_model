"""Vegetation features: NDVI from MODIS MOD13Q1 composites."""

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import numpy as np
import rasterio

from ..core.base import VarBuilder
from ..core.raster import RasterManager
import logging

logger = logging.getLogger(__name__)


class VegetationBuilder(VarBuilder):
    """
    Build a mean NDVI surface from MODIS MOD13Q1 250 m composites.

    When *months* and *season* are supplied (seasonal_ndvi mode) only files
    whose acquisition date falls in the requested months and training years
    are included. Otherwise all available files are averaged.

    Pipeline:
        load + scale MODIS files  →  average  →  write native GeoTIFF
        →  clip to Essex  →  align to reference grid
    """

    # MODIS scale factor: stored DN → reflectance
    _NDVI_SCALE = 0.0001
    _NDVI_VALID = (-0.2, 1.0)

    def process(
        self,
        months: Optional[Tuple[int, ...]] = None,
        season: Optional[str] = None,
    ) -> Dict[str, Path]:
        data_config = self.config["data_sources"]["modis_nvdi"]
        training_years: Tuple[int, int] = self.config["processing"]["training_years"]

        out_name = self._seasonal_name("ndvi", season)
        output_paths = {"ndvi": self.output_dir / f"{out_name}.tif"}

        if self._check_cache(f"VegetationBuilder[{season}]", output_paths):
            return output_paths

        self._validate_reference()
        modis_dir = Path(data_config["data_dir"])

        tif_files = self._find_files(modis_dir, training_years, months)
        mean_ndvi, source_meta = self._average_ndvi(tif_files)

        native_path = self.output_dir / f"_tmp_ndvi_native_{season or 'static'}.tif"
        self._write_native(mean_ndvi, source_meta, native_path)
        self._to_reference(native_path, output_paths["ndvi"], tmp_stem=f"_tmp_clip_ndvi_{season}")
        native_path.unlink()

        logger.info(f"[{season}] NDVI ready → {output_paths['ndvi'].name}")
        return output_paths

    def _find_files(
        self,
        modis_dir: Path,
        training_years: Tuple[int, int],
        months: Optional[Tuple[int, ...]],
    ) -> List[Path]:
        """Return MODIS .tif files filtered by training window and season."""
        all_files = sorted(modis_dir.glob("**/*.tif"))
        if not all_files:
            raise FileNotFoundError(f"No MODIS .tif files found under {modis_dir}")

        if months is None:
            return all_files

        year_start, year_end = training_years
        filtered = [
            f for f in all_files
            if self._file_in_season(f, range(year_start, year_end), months)
        ]
        if not filtered:
            raise FileNotFoundError(
                f"No MODIS files matched training years {training_years} "
                f"and months {months}."
            )
        return filtered

    @staticmethod
    def _file_in_season(
        fpath: Path,
        valid_years: range,
        months: Tuple[int, ...],
    ) -> bool:
        """Return True if the DOY-encoded filename falls in the requested season."""
        match = re.search(r"doy(\d{4})(\d{3})", fpath.name)
        if not match:
            return True   # no date encoding → include by default
        year, doy = int(match.group(1)), int(match.group(2))
        acq_date = date(year, 1, 1) + timedelta(days=doy - 1)
        return year in valid_years and acq_date.month in months

    def _average_ndvi(
        self, tif_files: List[Path]
    ) -> Tuple[np.ndarray, dict]:
        """
        Load each MODIS file, apply the scale factor, mask invalid values,
        and return the pixel-wise temporal mean and the metadata of the
        first valid file (used to georeference the output).
        """
        arrays: List[np.ndarray] = []
        first_meta: Optional[dict] = None

        for fpath in tif_files:
            ndvi, meta = self._load_and_scale(fpath)
            if ndvi is not None:
                arrays.append(ndvi)
                if first_meta is None:
                    first_meta = meta

        if not arrays:
            raise ValueError("No valid NDVI arrays could be extracted from MODIS files.")

        mean_ndvi = np.nanmean(np.stack(arrays, axis=0), axis=0).astype("float32")
        return mean_ndvi, first_meta

    def _load_and_scale(self, fpath: Path) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Read one MODIS tile, apply scale factor, and mask out-of-range pixels.

        Returns (None, None) on read failure so the caller can skip silently.
        """
        try:
            with rasterio.open(fpath) as src:
                raw  = src.read(1).astype("float32")
                meta = src.meta.copy()

            ndvi = raw * self._NDVI_SCALE
            lo, hi = self._NDVI_VALID
            ndvi[(ndvi < lo) | (ndvi > hi)] = np.nan
            return ndvi, meta

        except Exception as exc:
            logger.warning(f"Skipping {fpath.name}: {exc}")
            return None, None
        
    @staticmethod
    def _write_native(
        ndvi_data: np.ndarray,
        meta: dict,
        out_path: Path,
    ) -> None:
        """Write the averaged NDVI array in its native CRS / resolution."""
        meta = meta.copy()
        meta.update({
            "count":   1,
            "dtype":   "float32",
            "nodata":  np.nan,
            "compress": "deflate",
            "height":  ndvi_data.shape[0],
            "width":   ndvi_data.shape[1],
        })
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(ndvi_data[np.newaxis, :, :])