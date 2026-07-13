"""Stage: seasonal feature construction (climate, NDVI).

No label logic here — that's stage_labels.py. Depends on ref_path only.

    stage_seasonal(config, input_paths) -> {
        "<season>": {
            "train": {<feature_name>: Path, ...},
            "test":  {<feature_name>: Path, ...},
        },
        ...
    }
"""
from pathlib import Path
from typing import Dict, Tuple

from ..features.climate import ClimateBuilder
from ..features.vegetation import VegetationBuilder
from ..utils.logger import setup_logger


def stage_seasonal(config: dict, input_paths: Dict[str, Path]) -> Dict[str, dict]:
    logger = setup_logger(log_file=config["logging"]["log_path"], level=config["logging"]["level"])
    ref_path = input_paths["ref_path"]
    season_defs = config["seasons"]["definitions"]

    results: Dict[str, dict] = {}
    for season in config["seasons"]["active"]:
        months = tuple(season_defs[season])
        logger.info(f"[stage_seasonal] [{season}] Building seasonal features...")
        results[season] = {
            "train": _build(config, season, months, config["processing"]["training_years"], "train", ref_path),
            "test": _build(config, season, months, config["processing"]["test_years"], "test", ref_path),
        }
    return results


def _build(config, season, months, year_range, split, ref_path) -> Dict[str, Path]:
    climate_builder = ClimateBuilder(config, ref_path)
    paths = dict(climate_builder.process(months=months, year_range=year_range, split=split, season=season))

    if config["seasons"].get("seasonal_ndvi", False):
        veg_builder = VegetationBuilder(config, ref_path)
        paths.update(veg_builder.process(months=months, year_range=year_range, split=split, season=season))

    return paths