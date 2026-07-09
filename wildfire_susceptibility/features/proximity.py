"""Proximity features: distance to roads, rivers, buildings, and a
dominant land-use class per pixel."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize

from ..core.base import VarBuilder
from ..core.raster import RasterManager

from ..core.registry import FEATURE_BUILDERS

@FEATURE_BUILDERS.register("proximity")
class ProximityBuilder(VarBuilder):
    """
    Compute Euclidean distance (km) from every 30 m grid cell to the
    nearest road, river, and building, plus a dominant land-use class
    per pixel.

    d_buildings and landuse_class replace the previous merged
    `d_activity` layer (buildings + landuse dissolved into one distance
    surface), which had near-zero variance: buildings are sparse and
    geometrically discriminating, while the landuse polygons cover most
    of the non-urban raster and compressed the merged distance range
    toward near-zero almost everywhere. Splitting them gives (a) a
    distance layer with real variance, and (b) a genuinely new signal
    (what kind of activity is nearby, not just how far).

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

    The actual rasterise/EDT/mask/write logic for distance layers lives in
    RasterManager.distance_to_features, shared with FireProximityBuilder.
    """

    def process(self) -> Dict[str, Path]:
        data_config = self.config["data_sources"]["proximity"]
        prox_path = Path(data_config["data_dir"])

        output_paths = {
            "d_roads":      self.output_dir / "dist_roads.tif",
            "d_rivers":     self.output_dir / "dist_rivers.tif",
            "d_buildings":  self.output_dir / "dist_buildings.tif",
            "landuse_class": self.output_dir / "landuse_class.tif",
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

        osm_path = prox_path / "essex-260607-free.gpkg/essex.gpkg"

        buildings = self._load_vector(osm_path, layer="gis_osm_buildings_a_free")
        self._write_distance(buildings, output_paths["d_buildings"], label="Buildings")

        landuse = self._load_vector(
            osm_path,
            layer="gis_osm_landuse_a_free",
            filter_col="fclass",
            filter_values=data_config["human_fclasses"],
        )
        self._write_landuse_class(landuse, output_paths["landuse_class"], data_config["human_fclasses"])

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

    def _write_landuse_class(
        self,
        gdf: gpd.GeoDataFrame,
        out_path: Path,
        fclasses: List[str],
    ) -> None:
        """
        Rasterise the dominant human-activity land-use class per pixel.

        Pixel value is an ordinal code from 1..len(fclasses) — the 1-indexed
        position of the fclass in the configured `human_fclasses` list; 0
        means no human-activity land use present in that pixel. Where
        multiple polygons overlap a single 30 m cell, the last-drawn
        geometry wins (rasterize does not do area-weighted resolution) —
        acceptable at this resolution since human_fclasses polygons rarely
        overlap each other in practice.
        """
        class_map = {name: i + 1 for i, name in enumerate(fclasses)}
        shapes = [
            (geom, class_map[fclass])
            for geom, fclass in zip(gdf.geometry, gdf["fclass"])
            if geom is not None and not geom.is_empty and fclass in class_map
        ]

        with rasterio.open(self.ref_path) as ref:
            height, width = ref.height, ref.width
            transform = ref.transform
            meta = ref.meta.copy()
            land_mask = ref.read(1)

        landuse_arr = rasterize(
            shapes, out_shape=(height, width), transform=transform,
            fill=0, dtype="float32", all_touched=True,
        )
        landuse_arr[np.isnan(land_mask)] = np.nan

        meta.update({"dtype": "float32", "nodata": np.nan, "count": 1})
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(landuse_arr[np.newaxis, :, :])

        self.logger.info(
            f"Land-use class raster written -> {out_path.name} "
            f"({len(shapes):,} polygons, {len(class_map)} classes)"
        )


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