"""
Load and filter fire occurence data.
IMPORTANT: this is built ONCE from fire_train and reused, unmodified, in both
the train and test feature stacks. It is not rebuilt per split — rebuilding
it from test-period fires would leak future fire locations directly into the
model's input features, which is a strictly worse leak than a climate mean
mismatch and is not an experiment axis the way `include_d_fires_as_feature`
already is.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Tuple, Optional

from scipy.ndimage import gaussian_filter

from rasterio.features import rasterize

from ..core.base import VarBuilder

class FireBuilder(VarBuilder):

    def process(
        self,
        months: Tuple[int, ...],
        season: Optional[str] = None,
    ):
        output_paths = {
            "train": self.output_dir / f"{self._seasonal_name('fire_points_train', season)}.gpkg",
            "test": self.output_dir / f"{self._seasonal_name('fire_points_test', season)}.gpkg",
        }

        if self._check_cache(f"FireBuilder[{season}]", output_paths):
            train = gpd.read_file(output_paths["train"])
            test = gpd.read_file(output_paths["test"])
            return train, test

        gdf_all = self._load()
        clean_months = tuple(int(m) for m in months)
        gdf_season = gdf_all[gdf_all["month"].isin(clean_months)].copy()

        self.logger.info(f"Rows remaining after month filter: {len(gdf_season)}")
        train, test = self._split(gdf_season, output_paths)

        return train, test

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

        incidents_df.drop_duplicates(
            subset = ["CallDateID", "IncidentEasting", "IncidentNorthing"],
            inplace = True
        )

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
    ) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:

        boundary = gpd.read_file(self.boundary_path)
        gdf_clipped = gpd.sjoin(gdf, boundary, how="inner", predicate="within")
        gdf_clipped = gdf_clipped.drop(columns="index_right")

        train = self._within(gdf_clipped, self.config["processing"]["training_years"])
        test = self._within(gdf_clipped, self.config["processing"]["test_years"])

        train.to_file(output_paths["train"], driver="GPKG")
        test.to_file(output_paths["test"], driver="GPKG")

        return train, test
    
    def _within(self, df, set_years):
        start, end = set_years
        date_col = "year"
        
        new_df = df[
            (df[date_col] >= start) & (df[date_col] < end)
        ]

        return new_df