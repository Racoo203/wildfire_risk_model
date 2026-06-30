from pathlib import Path
from typing import Union, List, Tuple
import numpy as np
import geopandas as gpd

from ..core.base import VarBuilder
from ..core.raster import RasterManager

class BoundaryBuilder(VarBuilder):
    """From the boundaries of all historic counties of the UK, save only the boundary of Essex."""

    def process(self) -> None:
        """
        ...
        """
        data_config = self.config["data_sources"]["cua"]

        output_paths = {
            "boundary": self.output_dir / "boundary.shp"
        }

        if self._check_cache("BoundaryBuilder", output_paths):
            return output_paths

        #############################################

        boundary_path = Path(data_config["data_dir"])
        gdf = gpd.read_file(boundary_path / "CTYUA_DEC_2024_UK_BFC.shp")

        essex = gdf[gdf["CTYUA24NM"] == "Essex"].copy()

        essex_bng = essex.to_crs(self.config["processing"]["crs"])
        essex_bng.to_file(output_paths["boundary"])

        return None
