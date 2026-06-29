"""Proximity features: distance to roads, rivers, and human activity."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import distance_transform_edt

from ..core.base import VarBuilder
import logging

logger = logging.getLogger(__name__)


class ProximityBuilder(VarBuilder):
    """
    Compute Euclidean distance (km) from every 30 m grid cell to the
    nearest road, river, and area of human activity.

    Why distance is computed before clipping (not after):
        distance_transform_edt treats NoData / absent cells identically to
        'no feature here'. Clipping the vector data first would cause cells
        near the Essex border to measure distance to the (truncated) edge of
        the vector layer rather than to the nearest real feature, inflating
        distances near the boundary.  We therefore:
            1. Rasterise each layer over the *full reference grid extent*.
            2. Run distance_transform_edt on the full grid.
            3. Mask the result to Essex using the land mask embedded in the
               reference raster (NaN outside Essex).

    Pipeline per layer:
        load vector -> rasterise to reference grid -> EDT (full extent) ->
        convert pixels -> km -> apply land mask -> write
    """

    def process(self) -> Dict[str, Path]:
        data_config = self.config["data_sources"]["proximity"]
        prox_path   = Path(data_config["data_dir"])

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
        self._distance_raster(roads, output_paths["d_roads"], label="Roads")

        rivers = self._load_vector(
            prox_path / "oprvrs_gpkg_gb/Data/oprvrs_gb.gpkg",
            layer="watercourse_link",
        )
        self._distance_raster(rivers, output_paths["d_rivers"], label="Rivers")

        human_activity = self._build_human_activity_layer(prox_path, data_config)
        self._distance_raster(human_activity, output_paths["d_activity"], label="Human activity")

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

    def _distance_raster(
        self,
        gdf: gpd.GeoDataFrame,
        out_path: Path,
        label: str = "",
    ) -> None:
        """
        Rasterise gdf onto the reference grid, run Euclidean distance
        transform, convert to km, apply the land mask, and write to *out_path*.
        """
        with rasterio.open(self.ref_path) as ref:
            height     = ref.height
            width      = ref.width
            transform  = ref.transform
            meta       = ref.meta.copy()
            land_mask  = ref.read(1)   # NaN outside Essex

        cell_size_m = abs(transform.a)   # metres per pixel (square grid assumed)

        # --- 1. Rasterise: 1 where a feature exists, 0 elsewhere ----------
        shapes = [
            (geom, 1) for geom in gdf.geometry
            if geom is not None and not geom.is_empty
        ]
        presence = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        )

        # --- 2. EDT on the full grid (absence = 1 where no feature) -------
        absence      = (presence == 0).astype("uint8")
        dist_pixels  = distance_transform_edt(absence).astype("float32")
        dist_km      = dist_pixels * cell_size_m / 1000.0

        # --- 3. Mask to Essex study area -----------------------------------
        dist_km[np.isnan(land_mask)] = np.nan

        # --- 4. Write -------------------------------------------------------
        meta.update({"dtype": "float32", "nodata": np.nan, "count": 1})
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(dist_km[np.newaxis, :, :])

        logger.info(
            f"{label}: distance raster written → {out_path.name} "
            f"(cell size {cell_size_m:.0f} m, {int((presence > 0).sum()):,} feature pixels)"
        )