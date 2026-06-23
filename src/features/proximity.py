from pathlib import Path
from typing import Union, List
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

from ..core.base import VarBuilder
from ..core.raster import RasterManager
import logging

logger = logging.getLogger(__name__)

class ProximityBuilder(VarBuilder):
    def process(self):
        data_config = self.config["data_sources"]["proximity"]
        prox_path = Path(data_config["data_dir"])
        os_roads_path = prox_path / "oproad_gpkg_gb/Data/oproad_gb.gpkg"
        os_rvrs_path = prox_path / "oprvrs_gpkg_gb/Data/oprvrs_gb.gpkg"
        osm_path = prox_path / "essex-260607-free.gpkg/essex.gpkg"

        self._validate_reference()

        output_paths = {
            "d_roads": self.output_dir / "dist_roads.tif",
            "d_rivers": self.output_dir / "dist_rivers.tif",
            "d_activity": self.output_dir / "dist_activity.tif",
        }

        if self._check_cache("ProximityBuilder", output_paths):
            return output_paths

        roads = self._load(os_roads_path, layer = "road_link")
        self._vector_to_distance_raster(roads, "D_Roads", output_paths["d_roads"])

        rivers = self._load(os_rvrs_path, layer = "watercourse_link")
        self._vector_to_distance_raster(rivers, "D_Rivers", output_paths["d_rivers"])
        
        landuse = self._load(
            osm_path, 
            layer = "gis_osm_landuse_a_free", 
            filter_col = "fclass", 
            filter_values = data_config["human_fclasses"]
        )

        buildings = self._load(osm_path, layer = "gis_osm_buildings_a_free")

        human_activity = pd.concat([landuse, buildings], ignore_index = True)
        human_activity = gpd.GeoDataFrame(human_activity, crs = "EPSG:27700")
        human_activity = human_activity.dissolve().explode(index_parts = False)

        self._vector_to_distance_raster(human_activity, "D_Human_Activity", output_paths["d_activity"])

        return output_paths

    def _load(
        self,
        vector_path: Union[str, Path],
        layer: str = None,
        filter_col: str = None,
        filter_values: List[str] = None,
    ) -> gpd.GeoDataFrame:
        load_kwargs = {"filename": vector_path, "layer": layer}
        gdf = gpd.read_file(**load_kwargs)

        if filter_col and filter_values:
            gdf = gdf[gdf[filter_col].isin(filter_values)].copy()
        
        gdf = gdf.to_crs("EPSG:27700")

        return gdf
    
    def _vector_to_distance_raster(
        self,
        gdf: gpd.GeoDataFrame,
        label: str,
        out_path: Path
    ) -> None:
        with rasterio.open(self.ref_path) as ref:
            ref_transform = ref.transform
            width = ref.width
            height = ref.height
            ref_meta = ref.meta.copy()
            land_mask = ref.read(1)

        shapes_list = []
        for geom in gdf.geometry:
            if geom is None or geom.is_empty:
                continue
            try:
                shapes_list.append((geom, 1))
            except Exception as e:
                # logger.warning()
                continue
        
        kwargs = {
            "out_shape": [height, width],
            "transform": ref_transform,
            "fill": 0,
            "dtype": "uint8",
            "all_touched": True
        }

        presence = rasterize(
            shapes_list,
            **kwargs
        )

        absence = (presence == 0).astype("uint8")
        dist_pixels = np.zeros((height, width), dtype = "float32")
        dist_pixels = distance_transform_edt(absence).astype("float32")
        
        cell_size_m = abs(ref_transform.a)
        dist_km = (dist_pixels * cell_size_m / 1000.0).astype("float32")
        dist_km[np.isnan(land_mask)] = np.nan

        ref_meta.update({"dtype": "float32", "nodata": np.nan, "count": 1})

        with rasterio.open(out_path, "w", **ref_meta) as dst:
            dst.write(dist_km[np.newaxis, :, :])
        
        # logger.info()

        return out_path
        

