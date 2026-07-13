"""Stage: static feature construction.

Builds the reference raster and every static (non-seasonal) feature
layer: topography, boundary, proximity.

    stage_static(config, input_paths={}) -> {
        "ref_path": Path,
        "elevation": Path, "slope": Path, "aspect": Path,
        "d_roads": Path, "d_rivers": Path, "d_buildings": Path,
        "landuse_class": Path,
        "boundary": Path,
    }
"""
from pathlib import Path
from typing import Dict

from ..features.boundary import BoundaryBuilder
from ..features.topography import TopographyBuilder
from ..features.proximity import ProximityBuilder
from ..utils.logger import setup_logger


def stage_static(config: dict, input_paths: Dict[str, Path]) -> Dict[str, Path]:
    # logger = setup_logger(log_file=config["logging"]["log_path"], level=config["logging"]["level"])
    logger = setup_logger()
    logger.info("[stage_static] Building static features...")

    boundary_builder = BoundaryBuilder(config)
    boundary_paths = boundary_builder.process()

    topo_builder = TopographyBuilder(config)
    topo_paths = topo_builder.process()
    ref_path = topo_paths["elevation"]

    prox_builder = ProximityBuilder(config, ref_path)
    prox_paths = prox_builder.process()

    output_paths: Dict[str, Path] = {
        "ref_path": ref_path,
        **boundary_paths,
        **topo_paths,
        **prox_paths,
    }
    logger.info(f"[stage_static] Complete — {len(output_paths)} output path(s).")
    return output_paths