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
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ..features.proximity import FireProximityBuilder
from ..labels.fire_incidents import FireBuilder
from ..labels.kernel_density import KernelDensityClassifier
from ..utils.logger import setup_logger

logger = logging.getLogger(__name__)


def stage_labels(config: dict, input_paths: Dict[str, Path]) -> Dict[str, dict]:
    # logger = setup_logger(log_file=config["logging"]["log_path"], level=config["logging"]["level"])
    logger = setup_logger()

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

    # Methodology-comparison side-run: does not change `out` (downstream
    # stages still consume the single classify_method build above) -- this
    # only fits + freeze-applies every method in compare_classify_methods
    # and logs each one distinctly to MLflow for the dissertation's
    # method-comparison section.
    compare_methods = config["labels"].get("compare_classify_methods", [])
    if compare_methods:
        _run_classify_comparison(config, season, fire_train_gdf, fire_test_gdf, ref_path, compare_methods)

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


def _run_classify_comparison(
    config: dict,
    season: str,
    fire_train_gdf,
    fire_test_gdf,
    ref_path: Path,
    methods: list,
) -> None:
    """Fit + freeze-apply every classify_method in compare_classify_methods
    for this season (same train-fit/test-freeze-apply contract as the
    primary build in _build_raw_labels), logging each one to its own nested
    MLflow run so per-method params/metrics/artifacts don't collide and can
    be pulled into a comparison table later.

    density_method and n_classes/auto_find_k are held fixed at whatever
    the run is configured with -- only classify_method varies across the
    comparison, so the class *count* is comparable across methods.
    """
    import mlflow
    from ..modeling.training.trainer import ModelTrainer

    ModelTrainer._ensure_mlflow_backend()
    mlflow.set_experiment(config["modeling"]["mlflow_experiment"])

    density_method = config["labels"].get("density_method", "convolution")
    kde = KernelDensityClassifier(config, ref_path)

    with mlflow.start_run(run_name=f"{season}_classify_comparison"):
        mlflow.set_tags({
            "season": season,
            "density_method": density_method,
            "comparison": "classify_method",
            "compared_methods": ",".join(methods),
        })

        for method in methods:
            logger.info(f"[stage_labels] [{season}] comparison run: classify_method='{method}'")

            density_train = kde.compute_density(fire_train_gdf, season=f"{season}_train", method=density_method)
            labels_train, fit_train = kde.classify(
                density_train, season=f"{season}_train",
                method=density_method, classify_method=method, fitted=None,
            )

            density_test = kde.compute_density(fire_test_gdf, season=f"{season}_test", method=density_method)
            labels_test, fit_test = kde.classify(
                density_test, season=f"{season}_test",
                method=density_method, classify_method=method, fitted=fit_train,
            )

            with mlflow.start_run(run_name=f"{season}_{method}", nested=True):
                mlflow.set_tags({
                    "season": season,
                    "classify_method": method,
                    "density_method": density_method,
                })
                mlflow.log_params({
                    "n_classes": config["labels"].get("n_classes", 4),
                    "auto_find_k": config["labels"].get("auto_find_k", False),
                    "trim_bottom_pct": config["labels"].get("trim_bottom_pct", 0.1),
                })
                for i, threshold in enumerate(fit_train["thresholds"]):
                    mlflow.log_metric(f"threshold_{i}", float(threshold))

                _log_class_counts(labels_train, prefix="train")
                _log_class_counts(labels_test, prefix="test")

                mlflow.log_artifact(str(fit_train["label_path"]))
                mlflow.log_artifact(str(fit_test["label_path"]))


def _log_class_counts(labels: np.ndarray, prefix: str) -> None:
    """Log per-class cell counts/fractions for one split's label raster as
    mlflow metrics -- the basis for the dissertation's method-comparison
    table (class balance side-by-side across percentile/jenks/gmm)."""
    import mlflow

    valid = labels[~np.isnan(labels)]
    n_total = int(valid.size)
    mlflow.log_metric(f"{prefix}_n_total", n_total)
    if n_total == 0:
        return

    n_classes = int(valid.max()) + 1
    for c in range(n_classes):
        count = int(np.sum(valid == c))
        mlflow.log_metric(f"{prefix}_count_class_{c}", count)
        mlflow.log_metric(f"{prefix}_frac_class_{c}", count / n_total)