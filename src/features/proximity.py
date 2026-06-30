"""Proximity features: distance to roads, rivers, and human activity."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import geopandas as gpd
import pandas as pd

from ..core.base import VarBuilder
from ..core.raster import RasterManager

class ProximityBuilder(VarBuilder):
    """
    Compute Euclidean distance (km) from every 30 m grid cell to the
    nearest road, river, and area of human activity.

    Why distance is computed before clipping (not after):
        distance_transform_edt treats NoData / absent cells identically to
        'no feature here'. Clipping the vector data first would cause cells
        near the Essex border to measure distance to the (truncated) edge of
        the vector layer rather than to the nearest real feature, inflating
        distances near the boundary. We therefore:
            1. Rasterise each layer over the *full reference grid extent*.
            2. Run distance_transform_edt on the full grid.
            3. Mask the result to Essex using the land mask embedded in the
               reference raster (NaN outside Essex).

    Pipeline per layer:
        load vector -> rasterise to reference grid -> EDT (full extent) ->
        convert pixels -> km -> apply land mask -> write

    The actual rasterise/EDT/mask/write logic lives in
    RasterManager.distance_to_features, shared with FireProximityBuilder.
    """

    def process(self) -> Dict[str, Path]:
        data_config = self.config["data_sources"]["proximity"]
        prox_path = Path(data_config["data_dir"])

        output_paths = {
            "d_roads":    self.output_dir / "dist_roads.tif",
            "d_rivers":   self.output_dir / "dist_rivers.tif",
            "d_activity": self.output_dir / "dist_activity.tif",
        }

        if self._check_cache("ProximityBuilder", output_paths):
            return output_paths

        self._validate_reference()

        roads = self._load_vector(
            prox_path / "oproad_gpkg_gb/Data/oproad_gb.gpkg",
            layer="road_link",
        )
        self._write_distance(roads, output_paths["d_roads"], label="Roads")

        rivers = self._load_vector(
            prox_path / "oprvrs_gpkg_gb/Data/oprvrs_gb.gpkg",
            layer="watercourse_link",
        )
        self._write_distance(rivers, output_paths["d_rivers"], label="Rivers")

        human_activity = self._build_human_activity_layer(prox_path, data_config)
        self._write_distance(human_activity, output_paths["d_activity"], label="Human activity")

        return output_paths

    def _load_vector(
        self,
        path: Union[str, Path],
        layer: Optional[str] = None,
        filter_col: Optional[str] = None,
        filter_values: Optional[List[str]] = None,
    ) -> gpd.GeoDataFrame:
        """Read a vector layer, optionally filter rows, and reproject to BNG."""
        gdf = gpd.read_file(path, layer=layer)
        if filter_col and filter_values:
            gdf = gdf[gdf[filter_col].isin(filter_values)].copy()
        return gdf.to_crs("EPSG:27700")

    def _build_human_activity_layer(
        self,
        prox_path: Path,
        data_config: dict,
    ) -> gpd.GeoDataFrame:
        """
        Combine OSM land-use polygons and building footprints into a single
        dissolved geometry representing areas of human activity.
        """
        osm_path = prox_path / "essex-260607-free.gpkg/essex.gpkg"

        landuse = self._load_vector(
            osm_path,
            layer="gis_osm_landuse_a_free",
            filter_col="fclass",
            filter_values=data_config["human_fclasses"],
        )
        buildings = self._load_vector(osm_path, layer="gis_osm_buildings_a_free")

        combined = gpd.GeoDataFrame(
            pd.concat([landuse, buildings], ignore_index=True),
            crs="EPSG:27700",
        )
        return combined.dissolve().explode(index_parts=False)

    def _write_distance(
        self,
        gdf: gpd.GeoDataFrame,
        out_path: Path,
        label: str = "",
    ) -> None:
        """Build (geometry, value) shapes from a GeoDataFrame and delegate to RasterManager."""
        shapes = [
            (geom, 1) for geom in gdf.geometry
            if geom is not None and not geom.is_empty
        ]
        RasterManager.distance_to_features(shapes, self.ref_path, out_path)
        self.logger.info(f"{label}: distance raster written -> {out_path.name}")
class FireProximityBuilder(VarBuilder):
    """
    Distance (km) from every grid cell to the nearest *training-set* fire
    point, per season. Must be built strictly from fire_train — using val/test
    fire locations here would leak future fire locations into the features.
    """
    def process(self, fire_train_gdf, season: Optional[str] = None) -> Dict[str, Path]:
        out_name = self._seasonal_name("dist_fires", season)
        output_paths = {"d_fires": self.output_dir / f"{out_name}.tif"}

        if self._check_cache(f"FireProximityBuilder[{season}]", output_paths):
            return output_paths

        self._validate_reference()

        if len(fire_train_gdf) == 0:
            raise ValueError(f"[{season}] No training fire points to build d_fires from.")

        shapes = [
            ({"type": "Point", "coordinates": (geom.x, geom.y)}, 1)
            for geom in fire_train_gdf.geometry
        ]
        RasterManager.distance_to_features(shapes, self.ref_path, output_paths["d_fires"])

        self.logger.info(f"[{season}] d_fires built from {len(fire_train_gdf)} training fires")
        return output_paths