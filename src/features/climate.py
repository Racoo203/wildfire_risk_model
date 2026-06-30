"""Climate features from HadUK-Grid files."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import Affine

from ..core.base import VarBuilder
import logging

class ClimateBuilder(VarBuilder):
    """
    Build seasonal climate averages from HadUK-Grid monthly NetCDF files.

    Pipeline per variable:
        load + average NetCDF ->  write native-BNG GeoTIFF -> clip -> align
    """

    def process(
        self,
        months: Tuple[int, ...],
        season: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Compute long-term seasonal means over the training period only.

        Using training-period data exclusively is essential: including
        validation- or test-period climate would constitute data leakage.

        Args:
            months: Calendar months that define this season (e.g. (6, 7, 8)).
            season: Label used in output filenames (e.g. 'summer').

        Returns:
            Mapping from HadUK variable name to aligned GeoTIFF path.
        """
        data_config = self.config["data_sources"]["haduk"]
        training_years: Tuple[int, int] = self.config["processing"]["training_years"]

        output_paths = {
            var: self.output_dir / f"meteo_{self._seasonal_name(var, season)}.tif"
            for var in data_config["sources"]
        }

        if self._check_cache(f"ClimateBuilder[{season}]", output_paths):
            return output_paths

        haduk_dir = Path(data_config["data_dir"])
        self._validate_reference()

        for var_name, out_path in output_paths.items():
            mean_grid, lons, lats = self._load_seasonal_mean(
                haduk_dir, var_name, training_years, months
            )
            native_bng = self._write_native_bng(mean_grid, lons, lats, var_name, season)
            self._to_reference(native_bng, out_path, tmp_stem=f"_tmp_clip_{var_name}_{season}")
            native_bng.unlink()
            self.logger.info(f"[{season}] Climate feature ready: {var_name} → {out_path.name}")

        return output_paths
    
    def _load_seasonal_mean(
        self,
        haduk_dir: Path,
        var_name: str,
        training_years: Tuple[int, int],
        months: Tuple[int, ...],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Open all HadUK files for *var_name* that fall within the training window,
        filter to the requested calendar months, and return the temporal mean.

        Returns:
            (mean_2d, eastings_1d, northings_1d)
        """
        year_start, year_end = training_years
        files = sorted((haduk_dir / var_name).glob("*.nc"))
        files = [
            f for f in files
            if any(str(y) in f.name for y in range(year_start, year_end))
        ]
        if not files:
            raise FileNotFoundError(
                f"No HadUK files for '{var_name}' in {haduk_dir / var_name} "
                f"covering {year_start} - {year_end}."
            )

        ds = xr.open_mfdataset(files, combine="by_coords", data_vars="all")
        data = ds[var_name]

        time_month = data.time.dt.month.values
        time_year = data.time.dt.year.values
        time_mask = (
            (time_year >= year_start) & (time_year < year_end)
            & np.isin(time_month, months)
        )
        data_filtered = data.isel(time=time_mask)

        if len(data_filtered.time) == 0:
            raise ValueError(
                f"No timesteps found for '{var_name}' in years "
                f"{year_start}–{year_end} and months {months}."
            )

        mean_grid = data_filtered.mean(dim="time").values.astype("float32")

        # HadUK projection coordinates are 1-D BNG eastings / northings
        lons = data.coords["projection_x_coordinate"].values  # eastings (m)
        lats = data.coords["projection_y_coordinate"].values  # northings (m)

        return mean_grid, lons, lats

    def _write_native_bng(
        self,
        data: np.ndarray,
        eastings: np.ndarray,
        northings: np.ndarray,
        var_name: str,
        season: Optional[str],
    ) -> Path:
        """
        Write the 2-D mean grid as a GeoTIFF declared natively in BNG
        (EPSG:27700). The HadUK grid is already in BNG; declaring it
        correctly avoids a needless CRS reprojection later.

        Ensures north-up orientation: if northings run south-to-north
        (ascending), the array and coordinate array are both flipped so
        the raster origin sits at the north-west corner, as rasterio expects.
        """
        if northings[0] < northings[-1]:          # south-to-north → flip
            northings = northings[::-1]
            data = data[::-1, :]

        res_x = abs(eastings[1] - eastings[0])
        res_y = abs(northings[0] - northings[1])   # after flip this is always positive

        transform = (
            Affine.translation(eastings.min(), northings.max())
            * Affine.scale(res_x, -res_y)
        )

        tmp_path = self.output_dir / f"_tmp_native_{var_name}_{season or 'static'}.tif"
        meta = {
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:27700",
            "transform": transform,
            "nodata": np.nan,
            "compress": "deflate",
        }
        with rasterio.open(tmp_path, "w", **meta) as dst:
            dst.write(data[np.newaxis, :, :])

        return tmp_path