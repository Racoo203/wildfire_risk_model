"""Stage: fire incident + label raster construction.

Depends on ref_path only (not stage_seasonal's output) — labels are
built from fire point data, independent of climate/NDVI features.

    stage_labels(config, input_paths) -> {
        "<season>": {
            "train": {"raw_labels": Path},
            "test":  {"raw_labels": Path},
            "fire_train": Path,   # gpkg
            "fire_test": Path,    # gpkg
        },
        ...
    }
"""
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..features.proximity import FireProximityBuilder
from ..labels.fire_incidents import FireBuilder
from ..labels.kernel_density import KernelDensityClassifier
from ..utils.logger import setup_logger


def stage_labels(config: dict, input_paths: Dict[str, Path]) -> Dict[str, dict]:
    logger = setup_logger(log_file=config["logging"]["log_path"], level=config["logging"]["level"])
    ref_path = input_paths["ref_path"]
    season_defs = config["seasons"]["definitions"]

    results: Dict[str, dict] = {}
    for season in config["seasons"]["active"]:
        months = tuple(season_defs[season])
        logger.info(f"[stage_labels] [{season}] Building fire labels...")
        results[season] = _build_season(config, season, months, ref_path)
    return results


def _build_season(config: dict, season: str, months: Tuple[int, ...], ref_path: Path) -> dict:
    fire_builder = FireBuilder(config)
    fire_train_gdf, fire_test_gdf = fire_builder.process(months=months, season=season)

    d_fires_paths = {}
    if config["labels"].get("include_d_fires_as_feature", True):
        fire_prox_builder = FireProximityBuilder(config, ref_path)
        d_fires_paths = fire_prox_builder.process(fire_train_gdf, season=season)

    splits = {
        "train": (config["processing"]["training_years"], fire_train_gdf),
        "test": (config["processing"]["test_years"], fire_test_gdf),
    }

    out: dict = {}
    label_fit = None  # frozen after train classifies; reused, never refit, on test
    for split, (year_range, fire_gdf) in splits.items():
        raw_labels_path, label_fit = _build_raw_labels(config, fire_gdf, season, split, ref_path, fitted=label_fit)
        out[split] = {"raw_labels": raw_labels_path, **d_fires_paths}

    out["fire_train"] = fire_builder.output_dir / f"{fire_builder._seasonal_name('fire_points_train', season)}.gpkg"
    out["fire_test"] = fire_builder.output_dir / f"{fire_builder._seasonal_name('fire_points_test', season)}.gpkg"
    return out


def _build_raw_labels(
    config: dict, fire_gdf, season: str, split: str, ref_path: Path, fitted: Optional[dict] = None
) -> Tuple[Path, dict]:
    density_method = config["labels"].get("density_method", "convolution")
    classify_method = config["labels"].get("classify_method", "percentile")

    kde = KernelDensityClassifier(config, ref_path)
    density = kde.compute_density(fire_gdf, season=f"{season}_{split}", method=density_method)
    labels, fit_artifact = kde.classify(
        density, season=f"{season}_{split}",
        method=density_method, classify_method=classify_method, fitted=fitted,
    )
    return fit_artifact["label_path"], fit_artifact