"""Load and filter fire occurence data."""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Union, Tuple, Optional
import logging

from scipy.ndimage import gaussian_filter

import rasterio
from rasterio.features import rasterize
from rasterio.enums import MergeAlg

from ..core.base import VarBuilder
from ..core.raster import RasterManager

logger = logging.getLogger(__name__)

class FireBuilder(VarBuilder):

    def process(
        self,
        months: Tuple[int, ...], 
        season: Optional[str] = None,
    ):
        data_config = self.config["data_sources"]["incidents"]

        output_paths = {
            "train": self.output_dir / f"{self._seasonal_name('fire_points_train', season)}.gpkg",
            "val": self.output_dir / f"{self._seasonal_name('fire_points_val', season)}.gpkg",
            "test": self.output_dir / f"{self._seasonal_name('fire_points_test', season)}.gpkg",
        }

        if self._check_cache(f"FireBuilder[{season}]", output_paths):
            train = gpd.read_file(output_paths["train"])
            val = gpd.read_file(output_paths["val"])
            test = gpd.read_file(output_paths["test"])
            return train, val, test

        force_recompute = self.config["processing"]["force_recompute"]

        data_dir = Path(data_config["data_dir"]) / "OutdoorFIres_2009_2025.csv"
        gdf_all = self._load()
        
        clean_months = tuple(int(m) for m in months)

        gdf_season = gdf_all[gdf_all["month"].isin(clean_months)].copy()

        print(f"Rows remaining after month filter: {len(gdf_season)}")
        train, val, test = self._split(gdf_season, output_paths)

        return train, val, test

    def _load(self):
        data_config = self.config["data_sources"]["incidents"]
        data_path = Path(data_config["data_dir"]) / "OutdoorFIres_2009_2025.csv"

        date_cols = data_config["date_col"]
        geo_cols = data_config["geo_cols"]
        severity_cols = data_config["severity_cols"]
        cols = [date_cols] + geo_cols + severity_cols

        incidents_df = pd.read_csv(data_path)
        incidents_df = incidents_df[cols].copy()
        incidents_df[date_cols] = pd.to_datetime(incidents_df[date_cols], format = "%d/%m/%Y")
        incidents_df["year"] = incidents_df[date_cols].dt.year
        incidents_df["month"] = incidents_df[date_cols].dt.month
        incidents_df["day"] = incidents_df[date_cols].dt.day

        easting, northing = geo_cols
        gdf = gpd.GeoDataFrame(
            incidents_df,
            geometry = gpd.points_from_xy(incidents_df[easting], incidents_df[northing]),
            crs = self.config["processing"]["crs"]
        )
        
        return gdf
    
    def _split(
        self, 
        gdf: gpd.GeoDataFrame, 
        output_paths,
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:

        boundary = gpd.read_file(self.boundary_path)

        gdf_clipped = gpd.sjoin(gdf, boundary, how = "inner", predicate = "within")
        gdf_clipped = gdf_clipped.drop(columns = "index_right")

        train = self._within(gdf_clipped, self.config["processing"]["training_years"])
        val = self._within(gdf_clipped, self.config["processing"]["validation_years"])
        test = self._within(gdf_clipped, self.config["processing"]["test_years"])

        train.to_file(output_paths["train"], driver = "GPKG")
        val.to_file(output_paths["val"], driver = "GPKG")
        test.to_file(output_paths["test"], driver = "GPKG")

        return train, val, test
    
    def _within(self, df, set_years):
        start, end = set_years
        date_col = "year"
        
        new_df = df[
            (df[date_col] >= start) &
            (df[date_col] < end)
        ]

        return new_df