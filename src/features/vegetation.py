"""Vegetation features: NDVI (Normalied Difference Vegetation Index)"""

from pathlib import Path
from typing import Union, List, Optional, Tuple
import numpy as np
import rasterio
from rasterio.transform import from_bounds

from ..core.base import VarBuilder
from ..core.raster import RasterManager
import logging

class VegetationBuilder(VarBuilder):
    """Build NDVI from MODIS MOD13Q1 composites."""

    def process(
        self, 
        months: Optional[Tuple[int, ...]] = None, 
        season: Optional[str] = None,
    ):
        data_config = self.config["data_sources"]["modis_nvdi"]
        years = self.config["processing"]["training_years"]

        out_name = self._seasonal_name('ndvi', season)
        output_paths = {
            "ndvi": self.output_dir / f"{out_name}.tif"
        }

        if self._check_cache(f"VegetationBuilder[{season}]", output_paths):
            return output_paths

        modis_dir = Path(data_config["data_dir"])
        self._validate_reference()

        tif_files = sorted(modis_dir.glob("**/*.tif"))

        if not tif_files:
            raise FileNotFoundError(f"No MODIS files found on {modis_dir}")

        if months is not None:
            tif_files = [f for f in tif_files if self._file_filter_matches(f, years, months)]
            if not tif_files:
                raise FileNotFoundError(f"No MODIS files matched for the season.")
        
        months_arrays, first_meta = [], None
        for fpath in tif_files:
            ndvi, file_meta = self._load_modis_ndvi(fpath)
            if ndvi is not None:
                months_arrays.append(ndvi)
                first_meta = first_meta or file_meta

        if not months_arrays:
            raise ValueError("No valid NDVI arrays extracted from MODIS files.")
        
        ndvi_mean = np.nanmean(np.stack(months_arrays, axis = 0), axis = 0).astype("float32")

        tmp_path = self.output_dir / f"_tmp_ndvi_native_{season or 'static'}.tif"
        self._save_ndvi_temp(ndvi_mean, first_meta, tmp_path)

        RasterManager.align_to_reference(tmp_path, self.ref_path, output_paths["ndvi"])
        tmp_path.unlink()

        return output_paths
    
    def _load_modis_ndvi(self, fpath):
        """
        """
        try:
            with rasterio.open(fpath) as src:
                data = src.read(1).astype("float32")
                nodata = src.nodata if src.nodata else -28672
                meta = src.meta.copy()

            data = data * 0.0001
            data[(data < -0.2) | (data > 1.0)] = np.nan

            return data, meta

        except Exception as e:
            # logger.warning()
            return None, None

    def _save_ndvi_temp(
        self,
        ndvi_data,
        meta_data,
        out_path
    ):
        meta_data.update({
            "dtype": "float32",
            "nodata": np.nan,
            "compress": "deflate",
            "height": ndvi_data.shape[0],
            "width": ndvi_data.shape[1],
            "count": 1,
        })
        
        with rasterio.open(out_path, "w", **meta_data) as dst:
            dst.write(ndvi_data[np.newaxis, :, :])

    def _file_month_matches(
        fpath: Path,
        years: Tuple[int, ...],
        months: Tuple[int, ...],
    ) -> bool:
        match = re.search(r"doy(\d{4})(\d{3})", fpath.name)
        if not match:
            return True
        else:
            year, doy = int(match.group(1)), int(match.group(2))
            d = date(year, 1, 1) + timedelta(days = doy - 1)
            return (d.month in months) and (year in years)
