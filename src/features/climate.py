from pathlib import Path
import geopandas as gpd
from typing import Union, Tuple, Dict, Optional
import numpy as np
import xarray as xr

import rasterio
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.mask import mask

from ..core.base import VarBuilder
from ..core.raster import RasterManager
import logging

logger = logging.getLogger(__name__)

class ClimateBuilder(VarBuilder):
    """Build climate features from CHESS-met NetCDF files."""

    def process(
        self, 
        months: Tuple[int, ...], 
        season: Optional[str] = None,
    ) -> Dict[str, Path]:
        """
        Build climate features averaged over the specified years and months.
        
        Args:
            years:  (start, end) training years only. Passing the 
                    full date range would leak validation-period climate
                    into the training features.
            months: season months to average over
            season: season name for output file naming.
        
        Returns:
            dict mapping feature names to output GeoTIFF paths.
        """
        data_config = self.config["data_sources"]["haduk"]
        years = self.config["processing"]["training_years"]

        output_paths = {
            src: self.output_dir / f"meteo_{self._seasonal_name(src, season)}.tif"
            for src in data_config["sources"]
        }

        if self._check_cache(f"ClimateBuilder[{season}]", output_paths):
            return output_paths

        #############################################
        
        haduk_path = Path(data_config["data_dir"])
        self._validate_reference()

        tas, lons, lats = self._load_haduk_var(haduk_path, "tas", years, months)
        tasmax, _, _ = self._load_haduk_var(haduk_path, "tasmax", years, months)
        tasmin, _, _ = self._load_haduk_var(haduk_path, "tasmin", years, months)
        rainfall, _, _ = self._load_haduk_var(haduk_path, "rainfall", years, months)
        sfcWind, _, _ = self._load_haduk_var(haduk_path, "sfcWind", years, months)
        hurs, _, _ = self._load_haduk_var(haduk_path, "hurs", years, months)

        data_map = {
            "tas": tas,
            "tasmax": tasmax,
            "tasmin": tasmin,
            "rainfall": rainfall,
            "sfcWind": sfcWind,
            "hurs": hurs,
        }

        boundary = self._load_boundary()

        for name, data in data_map.items():
            # print(data.coords)

            print(lats.shape, lons.shape)

            tmp_wgs84 = self._save_haduk_to_geotiff(data, lons, lats, f"meteo_{name}_{season or 'static'}_wgs84")
            tmp_bng = self.output_dir / f"meteo_{name}_{season or 'static'}.tif"
            self._reproject_clip_resample(tmp_wgs84, boundary, tmp_bng)

            RasterManager.align_to_reference(tmp_bng, self.ref_path, output_paths[name])
            tmp_wgs84.unlink()
            # tmp_bng.unlink()

            logger.info(f"[{season}] Climate feature complete: {name}")

        return output_paths

    def _load_haduk_var(
        self,
        haduk_dir: Path, 
        var_name: str, 
        years: Tuple[int, int],
        months: Tuple[int, ...],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load HadUK files for one variable across multiple years, 
        filter by months, compute long-term mean.

        Returns:
            mean_2d_array
        """
        files = sorted((haduk_dir / var_name).glob("*.nc"))
        files = [
            f for f in files 
            if any(str(y) in f.name for y in range(years[0], years[1] + 1))
        ]

        if not files: 
            raise FileNotFoundError(
                f"No HadUK files found in {haduk_dir / var_name} "
                f"for years {years[0]} - {years[1]}"
            )

        ds = xr.open_mfdataset(files, combine = "by_coords")
        data = ds[var_name]

        lons_1d = data.coords["projection_x_coordinate"].values
        lats_1d = data.coords["projection_y_coordinate"].values

        time_month = data.time.dt.month.values
        time_year = data.time.dt.year.values

        mask = (
            (time_year >= years[0]) & (time_year < years[1]) &
            np.isin(time_month, months)
        )

        data_filtered = data.isel(time = mask)

        if len(data_filtered.time) == 0:
            raise ValueError(
                f"No data found for {var_name} in range of years and months."
            )

        seasonal_mean = data_filtered.mean(dim = "time").values.astype("float32")

        return seasonal_mean, lons_1d, lats_1d

    def _save_haduk_to_geotiff(
        self,
        data: np.ndarray, 
        lons: np.ndarray, # These are actually BNG Eastings in meters
        lats: np.ndarray, # These are actually BNG Northings in meters
        name: str
    ):
        # HadUK grids are often stored south-to-north; invert if needed for raster standard
        if lats[0] < lats[-1]:
            lats = lats[::-1]
            data = data[::-1, :]
        
        res_lat = abs(lats[1] - lats[0])
        res_lon = abs(lons[1] - lons[0])

        transform = Affine.translation(
            lons.min(), lats.max()
        ) * Affine.scale(res_lon, -res_lat)

        tmp_path = self.output_dir / f"_tmp_{name}.tif"

        meta = {
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:27700",  # FIX: Declare BNG natively instead of EPSG:4326
            "transform": transform,
            "nodata": np.nan,
            "compress": "deflate"
        }

        with rasterio.open(tmp_path, "w", **meta) as dst:
            dst.write(data[np.newaxis, :, :])

        return tmp_path

    def _reproject_clip_resample(
        self,
        src_path,
        boundary_gdf,
        out_path
    ):
        # Ensure boundary features match target projection (EPSG:27700)
        if boundary_gdf.crs != "EPSG:27700":
            boundary_gdf = boundary_gdf.to_crs("EPSG:27700")
            
        geoms = [geom for geom in boundary_gdf.geometry]
        
        # The source raster is now natively BNG, clip it directly
        with rasterio.open(src_path) as src:
            clipped, clipped_transform = mask(src, geoms, crop = True, nodata = np.nan)
            meta_clip = src.meta.copy()
            meta_clip.update({
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": clipped_transform,
            })

            tmp_clip = self.output_dir / f"_tmp_clip_{out_path.stem}.tif"
            with rasterio.open(tmp_clip, "w", **meta_clip) as dst:
                dst.write(clipped.astype("float32"))

        # Hand off to your standard RasterManager alignment sequence
        RasterManager.align_to_reference(tmp_clip, self.ref_path, out_path)
        
        # Clean up the intermediate clipping asset
        if tmp_clip.exists():
            tmp_clip.unlink()


    def _load_boundary(self):
        return gpd.read_file(self.boundary_path).to_crs("EPSG:27700")
        