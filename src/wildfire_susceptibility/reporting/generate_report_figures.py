"""
scripts/generate_report_figures.py

Canonical figure-generation entry point. This is the ONLY place figure
generation logic should live — eda.ipynb, the dissertation build.sh, and
app/control_panel.py's "Figures" stage button should all call into this
script (or its `generate_all` function directly) rather than each
re-implementing calls to viz.maps / viz.charts. See blueprint Section 9
and the architecture-decision note in project memory.

Every figure written here goes through wildfire_susceptibility/viz/, so
every one is automatically indexed in figures/manifest.json — nothing in
this script writes a PNG directly.

Usage:
    python scripts/generate_report_figures.py
    python scripts/generate_report_figures.py --season summer
    python scripts/generate_report_figures.py --season summer --skip-models
    python scripts/generate_report_figures.py --only factor_maps,eda

Categories (--only accepts a comma-separated subset):
    factor_maps         — one terrain-backdrop map per feature raster, per season
    susceptibility_maps  — the 4-class output map, per season (+ per model if available)
    eda                  — VIF/correlation (Pearson+Spearman), class balance, NaN coverage
    models               — ROC curves, CV comparison, SHAP (requires mlflow + trained runs)
    optuna_plots         — optimization history + param importance per model (requires
                            a completed/in-progress Optuna search per (season, model))
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from wildfire_susceptibility.config.loader import ConfigLoader, DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_FILES
from wildfire_susceptibility.core.registry import MODELS
from wildfire_susceptibility.modeling import models as _models  # noqa: F401 — registers wrappers
from wildfire_susceptibility import viz

logger = logging.getLogger("wildfire_susceptibility.scripts.generate_report_figures")

ALL_CATEGORIES = ("factor_maps", "susceptibility_maps", "eda", "models", "optuna_plots")

# --------------------------------------------------------------------------
# Path discovery — mirrors the naming conventions used by the feature
# builders / preprocessor, so this script never has to be told paths
# explicitly; it derives them from config the same way the pipeline does.
# --------------------------------------------------------------------------

def _layers_dir(cfg: dict) -> Path:
    return Path(cfg["base"]["output_dir"])


def _model_data_dir(cfg: dict) -> Path:
    return Path(cfg["base"]["model_data_dir"])


def _figures_dir(cfg: dict) -> Path:
    return Path(cfg["base"]["figures_dir"])


def _dem_path(cfg: dict) -> Path:
    return _layers_dir(cfg) / "topo_elevation.tif"


def discover_feature_rasters(cfg: dict, season: str) -> Dict[str, Path]:
    """
    Find every feature raster relevant to `season` by scanning output_dir
    for the naming patterns the builders actually write. Static features
    (topography, proximity) have no season suffix; climate/NDVI/d_fires
    do. Missing files are skipped with a warning rather than raising —
    a partial figure set is more useful than a crashed run when a season
    hasn't been fully processed yet.
    """
    layers_dir = _layers_dir(cfg)
    haduk_vars = cfg["data_sources"]["haduk"]["sources"]

    candidates = {
        "elevation": layers_dir / "topo_elevation.tif",
        "slope": layers_dir / "topo_slope.tif",
        "aspect": layers_dir / "topo_aspect.tif",
        "d_roads": layers_dir / "dist_roads.tif",
        "d_rivers": layers_dir / "dist_rivers.tif",
        "d_activity": layers_dir / "dist_activity.tif",
        "d_fires": layers_dir / f"dist_fires_{season}.tif",
        "ndvi": layers_dir / f"ndvi_{season}.tif",
    }
    for var in haduk_vars:
        candidates[var] = layers_dir / f"meteo_{var}_{season}.tif"

    found = {name: path for name, path in candidates.items() if path.exists()}
    missing = set(candidates) - set(found)
    if missing:
        logger.warning(f"[{season}] Skipping {len(missing)} feature raster(s) not found: {sorted(missing)}")
    return found


def discover_susceptibility_raster(cfg: dict, season: str) -> Optional[Path]:
    """Ground-truth density/classification label raster (from
    stage_labels), NOT a model's predicted output — used only for the EDA
    class-balance comparison. For the model-predicted risk map, see
    discover_predicted_susceptibility_rasters below."""
    path = _layers_dir(cfg) / f"risk_labels_clean_{season}.tif"
    return path if path.exists() else None


def discover_predicted_susceptibility_rasters(cfg: dict, season: str) -> Dict[str, Path]:
    """Every model-predicted susceptibility raster for `season`, as written
    by modeling/evaluate.py::generate_susceptibility_raster during
    stage_evaluate (`susceptibility_<model_name>_<season>.tif`), keyed by
    model name."""
    layers_dir = _layers_dir(cfg)
    suffix = f"_{season}.tif"
    found = {}
    for path in layers_dir.glob(f"susceptibility_*{suffix}"):
        model_name = path.stem[len("susceptibility_"):-len(f"_{season}")]
        found[model_name] = path
    return found


def discover_selected_model(cfg: dict, season: str) -> Optional[str]:
    """The best model per season, as chosen by stage_selection.py's
    selection_rule. Returns None if stage_selection hasn't run yet."""
    summary_path = _figures_dir(cfg) / "selection_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    return summary.get(season, {}).get("selected_model")


def discover_fire_test_points(cfg: dict, season: str) -> Optional[Path]:
    path = _layers_dir(cfg) / f"fire_points_test_{season}.gpkg"
    return path if path.exists() else None


def discover_dataset_csv(cfg: dict, season: str) -> Optional[Path]:
    path = _model_data_dir(cfg) / f"dataset_clean_{season}.csv"
    return path if path.exists() else None


# --------------------------------------------------------------------------
# Category generators
# --------------------------------------------------------------------------

def generate_factor_maps(cfg: dict, season: str) -> None:
    dem_path = _dem_path(cfg)
    if not dem_path.exists():
        logger.warning(f"[{season}] No DEM found at {dem_path}; skipping factor maps.")
        return

    feature_paths = discover_feature_rasters(cfg, season)
    if not feature_paths:
        logger.warning(f"[{season}] No feature rasters found; skipping factor maps.")
        return

    viz.render_all_factor_maps(feature_paths, dem_path, _figures_dir(cfg), season=season)
    logger.info(f"[{season}] Rendered {len(feature_paths)} factor map(s).")


def generate_susceptibility_map(cfg: dict, season: str) -> None:
    """
    Renders the dissertation's headline figure: the model-predicted 0-3
    risk class across Essex, with test-period fire incident locations
    overlaid. Uses stage_selection.py's chosen best model per season when
    available; otherwise falls back to rendering every model that has a
    predicted raster on disk, so the script stays usable before
    stage_selection has run.
    """
    dem_path = _dem_path(cfg)
    if not dem_path.exists():
        logger.warning(f"[{season}] No DEM found at {dem_path}; skipping susceptibility map.")
        return

    predicted_rasters = discover_predicted_susceptibility_rasters(cfg, season)
    if not predicted_rasters:
        logger.warning(f"[{season}] No model-predicted susceptibility raster found; skipping.")
        return

    selected_model = discover_selected_model(cfg, season)
    if selected_model and selected_model in predicted_rasters:
        to_render = {selected_model: predicted_rasters[selected_model]}
    else:
        if selected_model:
            logger.warning(
                f"[{season}] selection_summary.json names '{selected_model}' but no matching "
                f"raster was found; rendering all {len(predicted_rasters)} available model(s) instead."
            )
        else:
            logger.info(
                f"[{season}] No selection_summary.json yet; rendering all "
                f"{len(predicted_rasters)} available model(s)."
            )
        to_render = predicted_rasters

    fire_points_path = discover_fire_test_points(cfg, season)
    fire_points_gdf = gpd.read_file(fire_points_path) if fire_points_path else None
    if fire_points_path is None:
        logger.info(f"[{season}] No fire-test points found; rendering susceptibility map without overlay.")

    n_classes = cfg["labels"].get("n_classes", 4)
    for model_name, labels_path in to_render.items():
        viz.render_susceptibility_map(
            labels_path, dem_path, _figures_dir(cfg),
            n_classes=n_classes,
            season=season, model_name=model_name,
            fire_points_gdf=fire_points_gdf,
        )
    logger.info(f"[{season}] Rendered susceptibility map(s) for: {sorted(to_render)}.")


def generate_eda_figures(cfg: dict, season: str) -> None:
    csv_path = discover_dataset_csv(cfg, season)
    if csv_path is None:
        logger.warning(f"[{season}] No dataset CSV found at expected path; skipping EDA figures.")
        return

    df = pd.read_csv(csv_path)
    figures_dir = _figures_dir(cfg)

    feature_cols = [c for c in df.columns if c not in ("label",) and not c.startswith("_")]

    # VIF + Pearson + Spearman correlation, using the paper's exact thresholds
    viz.plot_vif_correlation(df, feature_cols, figures_dir, season=season)

    # NaN coverage — read directly from the raster feature stack so pre-
    # imputation NaN rates are shown (the CSV may already be post-cleaning).
    feature_paths = discover_feature_rasters(cfg, season)
    if feature_paths:
        arrays = {}
        for name, path in feature_paths.items():
            with rasterio.open(path) as src:
                arrays[name] = src.read(1)
        viz.plot_nan_coverage(arrays, figures_dir, season=season)

    # Class balance before/after cleaning — needs both raw and cleaned label
    # rasters; raw classify output uses the configured classify_method name.
    layers_dir = _layers_dir(cfg)
    density_method = cfg["labels"].get("density_method", "convolution")
    classify_method = cfg["labels"].get("classify_method", "gmm")
    raw_labels_path = layers_dir / f"risk_labels_{density_method}_{classify_method}_{season}.tif"
    clean_labels_path = discover_susceptibility_raster(cfg, season)

    if raw_labels_path.exists() and clean_labels_path is not None:
        with rasterio.open(raw_labels_path) as src:
            labels_before = src.read(1)
        with rasterio.open(clean_labels_path) as src:
            labels_after = src.read(1)
        viz.plot_class_balance(labels_before, labels_after, figures_dir, season=season)
    else:
        logger.warning(f"[{season}] Raw and/or cleaned label raster missing; skipping class balance plot.")

    logger.info(f"[{season}] EDA figures complete.")


def generate_model_figures(cfg: dict, season: str) -> None:
    """
    ROC comparison + CV comparison (standard vs spatial) pulled from mlflow
    run history, and a SHAP summary for the selection_rule-chosen best
    model. Skips gracefully (with a warning, not an error) if mlflow has no
    matching runs yet — this category depends on modeling/train.py having
    actually been run for this season.
    """
    import mlflow

    figures_dir = _figures_dir(cfg)
    experiment_name = cfg["modeling"]["mlflow_experiment"]

    try:
        mlflow.set_experiment(experiment_name)
        runs = mlflow.search_runs(experiment_names=[experiment_name])
    except Exception as exc:
        logger.warning(f"[{season}] Could not query mlflow experiment '{experiment_name}': {exc}")
        return

    if runs.empty:
        logger.warning(f"[{season}] No mlflow runs found in '{experiment_name}'; skipping model figures.")
        return

    season_runs = runs[runs.get("tags.season", pd.Series(dtype=str)) == season]
    if season_runs.empty:
        logger.warning(f"[{season}] No mlflow runs tagged for this season; skipping model figures.")
        return

    # CV comparison (standard vs spatial) — only meaningful if both metrics exist.
    if {"metrics.cv_auc_standard", "metrics.cv_auc_spatial"}.issubset(season_runs.columns):
        try:
            viz.plot_cv_comparison(figures_dir, season=season, mlflow_experiment=experiment_name)
            logger.info(f"[{season}] Rendered CV comparison chart.")
        except Exception as exc:
            logger.warning(f"[{season}] plot_cv_comparison failed: {exc}")
    else:
        logger.info(f"[{season}] Spatial CV metrics not present in mlflow runs; skipping CV comparison.")

    # ROC curves and SHAP need actual fitted model objects / prediction
    # arrays, which live only in-process during modeling/train.py — mlflow
    # search_runs() gives metrics/params, not artifacts we can replot from
    # without re-running inference. This script therefore reports what it
    # can from logged metrics and flags the rest as needing a live run.
    logger.info(
        f"[{season}] ROC/SHAP regeneration requires re-running inference with fitted "
        f"models (not reconstructable from mlflow metrics alone) — call "
        f"viz.plot_roc_curves / viz.plot_shap_summary directly from the training "
        f"session (e.g. at the end of modeling/train.py's train_all) if you want "
        f"those figures refreshed."
    )


def generate_optuna_plots(cfg: dict, season: str) -> None:
    """
    Optimization-history + param-importance plots for every model's Optuna
    search this season. Study names are reconstructed with the exact same
    deterministic scheme HyperparamSearch.get_or_create_study() uses
    (season + model name + param-space signature + objective version), so
    no separate index of study names needs to be persisted anywhere —
    the SQLite storage is already fully queryable as-is.
    """
    import optuna
    from wildfire_susceptibility.modeling.training.search import (
        HyperparamSearch, OPTUNA_STORAGE, OBJECTIVE_VERSION,
    )

    figures_dir = _figures_dir(cfg)
    for model_name in cfg["modeling"]["models"]:
        model_cls = MODELS[model_name]
        sig = HyperparamSearch._param_space_signature(model_cls)
        study_name = f"{season}_{model_name}_{sig}_{OBJECTIVE_VERSION}"

        try:
            study = optuna.load_study(study_name=study_name, storage=OPTUNA_STORAGE)
        except KeyError:
            logger.warning(f"[{season}][{model_name}] No Optuna study '{study_name}' found yet; skipping.")
            continue
        except Exception as exc:
            logger.warning(f"[{season}][{model_name}] Could not load Optuna study '{study_name}': {exc}")
            continue

        try:
            viz.plot_optuna_optimization_history(study, figures_dir, season=season, model_name=model_name)
            viz.plot_optuna_param_importances(study, figures_dir, season=season, model_name=model_name)
            logger.info(f"[{season}][{model_name}] Rendered Optuna plots.")
        except Exception as exc:
            logger.warning(f"[{season}][{model_name}] Optuna plot rendering failed: {exc}")


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

CATEGORY_FUNCS = {
    "factor_maps": generate_factor_maps,
    "susceptibility_maps": generate_susceptibility_map,
    "eda": generate_eda_figures,
    "models": generate_model_figures,
    "optuna_plots": generate_optuna_plots,
}



...

def generate_all(
    config_dir: Union[str, Path] = DEFAULT_CONFIG_DIR,
    config_files: List[str] = DEFAULT_CONFIG_FILES,
    seasons: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    skip_models: bool = False,
) -> None:
    """
    Main entry point — importable directly (e.g. from a notebook or the
    control panel) as an alternative to shelling out to this script.
    """
    cfg_obj = ConfigLoader.load(config_dir=config_dir, files=config_files)
    cfg = cfg_obj.model_dump(mode="python")

    active_seasons = seasons or cfg["seasons"]["active"]
    active_categories = categories or list(ALL_CATEGORIES)
    if skip_models:
        active_categories = [c for c in active_categories if c != "models"]

    unknown = set(active_categories) - set(ALL_CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown figure categor{'y' if len(unknown) == 1 else 'ies'}: {sorted(unknown)}")

    logger.info(f"Generating figures for seasons={active_seasons}, categories={active_categories}")

    for season in active_seasons:
        for category in active_categories:
            try:
                CATEGORY_FUNCS[category](cfg, season)
            except Exception:
                logger.exception(f"[{season}] Figure category '{category}' failed — continuing with remaining categories.")

    logger.info("Figure generation complete. See figures/manifest.json for the full index.")


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config-dir", 
        type=Path, 
        default=DEFAULT_CONFIG_DIR,
        help="Directory containing config YAML files."
    
    )

    parser.add_argument(
        "--config-file", 
        action="append", 
        dest="config_files", 
        default=None,
        help=f"Config YAML filename within --config-dir (repeatable). Defaults to {DEFAULT_CONFIG_FILES}."
    )

    parser.add_argument(
        "--season", 
        action="append", 
        dest="seasons", 
        default=None,
        help="Season to process (repeatable). Defaults to config.seasons.active."
    )

    parser.add_argument(
        "--only", 
        type=str, 
        default=None,
        help=f"Comma-separated subset of {ALL_CATEGORIES}. Defaults to all."
    )

    parser.add_argument(
        "--skip-models", 
        action="store_true",
        help="Skip the 'models' category (useful when mlflow has no runs yet)."
    )
    
    parser.add_argument(
        "-v", 
        "--verbose", 
        action="store_true", 
        help="Debug-level logging."
    )

    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    categories = args.only.split(",") if args.only else None
    generate_all(
        config_dir=args.config_dir,
        config_files=args.config_files or DEFAULT_CONFIG_FILES,
        seasons=args.seasons,
        categories=categories,
        skip_models=args.skip_models,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())