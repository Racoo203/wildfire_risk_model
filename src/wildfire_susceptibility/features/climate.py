"""Climate features from HadUK-Grid files."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.transform import Affine

from ..core.base import VarBuilder
from ..core.registry import FEATURE_BUILDERS

@FEATURE_BUILDERS.register("climate")
class ClimateBuilder(VarBuilder):
    """
    Build seasonal climate averages from HadUK-Grid monthly NetCDF files.

    Pipeline per variable:
        load + average NetCDF ->  write native-BNG GeoTIFF -> clip -> align

    In addition to the raw variables listed in data_sources.haduk.sources,
    this also derives `diurnal_range` (tasmax - tasmin) whenever both are
    present, so it's always available as a feature. Which temperature
    columns actually get *used* for modeling/label-cleaning (e.g. dropping
    the redundant `tas` in favour of `tasmax` + `diurnal_range`) is a
    separate, config-driven decision made later — see
    labels.clustering_exclude_features and modeling.excluded_features.
    """

    def process(
        self,
        months: Tuple[int, ...],
        year_range: Tuple[int, int],
        split: str,
        season: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Compute seasonal climate means for an explicit (year_range, split).

        `split` ("train" | "test") only affects output filenames/caching —
        it does not change the computation itself. Called once per split so
        the test-set feature stack reflects the test period's own climate
        rather than reusing training-period means, which would otherwise
        create a train/test feature mismatch for the years that matter most.
        """
        data_config = self.config["data_sources"]["haduk"]
        sources = list(data_config["sources"])
        derive_diurnal_range = "tasmax" in sources and "tasmin" in sources

        output_paths = {
            var: self.output_dir / f"meteo_{self._seasonal_name(var, season)}_{split}.tif"
            for var in sources
        }
        if derive_diurnal_range:
            output_paths["diurnal_range"] = (
                self.output_dir / f"meteo_{self._seasonal_name('diurnal_range', season)}_{split}.tif"
            )

        if self._check_cache(f"ClimateBuilder[{season}][{split}]", output_paths):
            return output_paths

        haduk_dir = Path(data_config["data_dir"])
        self._validate_reference()

        diurnal_inputs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for var_name in sources:
            out_path = output_paths[var_name]
            mean_grid, lons, lats = self._load_seasonal_mean(haduk_dir, var_name, year_range, months)
            if derive_diurnal_range and var_name in ("tasmax", "tasmin"):
                diurnal_inputs[var_name] = (mean_grid, lons, lats)

            native_bng = self._write_native_bng(mean_grid, lons, lats, var_name, f"{season}_{split}")
            self._to_reference(native_bng, out_path, tmp_stem=f"_tmp_clip_{var_name}_{season}_{split}")
            native_bng.unlink()
            self.logger.info(f"[{season}][{split}] Climate feature ready: {var_name}: {out_path.name}")

        if derive_diurnal_range:
            self._write_diurnal_range(diurnal_inputs, output_paths["diurnal_range"], season, split)

        return output_paths
    
    def eda_debug(self, var_name):
        data_config = self.config["data_sources"]["haduk"]
        haduk_dir = Path(data_config["data_dir"])
        
        files = sorted((haduk_dir / var_name).glob("*.nc"))
        ds = xr.open_mfdataset(files, combine="by_coords", data_vars="all")
        data = ds[var_name].values

        var_flat_df = pd.DataFrame(data.reshape((data.shape[0], -1))).dropna(how="any", axis=1)
        var_long = (
            var_flat_df.reset_index(drop=False, names="time")
            .melt(id_vars="time", var_name="pixel", value_name=var_name)
        )
        return var_long

    def _write_diurnal_range(
        self,
        diurnal_inputs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
        out_path: Path,
        season: Optional[str],
        split: str,
    ) -> None:
        """
        Derive mean diurnal temperature range (tasmax - tasmin) from the two
        seasonal-mean grids already computed in process(). This is
        mean-of-maxima minus mean-of-minima rather than a true per-day
        range averaged over the season — HadUK-Grid's monthly files don't
        expose daily max/min separately, so this is the closest faithful
        derivation available at this temporal resolution.
        """
        tasmax_grid, lons, lats = diurnal_inputs["tasmax"]
        tasmin_grid, _, _ = diurnal_inputs["tasmin"]
        diurnal_grid = (tasmax_grid - tasmin_grid).astype("float32")

        native_bng = self._write_native_bng(diurnal_grid, lons, lats, "diurnal_range", f"{season}_{split}")
        self._to_reference(native_bng, out_path, tmp_stem=f"_tmp_clip_diurnal_range_{season}_{split}")
        native_bng.unlink()
        self.logger.info(f"[{season}][{split}] Climate feature ready: diurnal_range → {out_path.name}")

    def _load_seasonal_mean(
        self,
        haduk_dir: Path,
        var_name: str,
        year_range: Tuple[int, int],
        months: Tuple[int, ...],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Open all HadUK files for *var_name* that fall within the training window,
        filter to the requested calendar months, and return the temporal mean.

        Returns:
            (mean_2d, eastings_1d, northings_1d)
        """
        year_start, year_end = year_range
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