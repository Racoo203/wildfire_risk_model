import json
from pathlib import Path
from typing import Dict

import geopandas as gpd
import joblib
import mlflow
import pandas as pd

from .. import viz
from ..modeling.categorical import cat_features_of, to_model_array
from ..modeling.dataset_prep import DatasetPrep
from ..modeling.training.evaluation import PostTrainingEvaluator
from ..modeling.training.trainer import ModelTrainer
from ..utils.logger import setup_logger


def stage_evaluate(config: dict, input_paths: dict) -> Dict[str, Dict[str, dict]]:
    # logger = setup_logger(log_file=config["logging"]["log_path"], level=config["logging"]["level"])
    logger = setup_logger()
    ref_path = input_paths["ref_path"]
    dem_path = Path(config["base"]["output_dir"]) / "topo_elevation.tif"

    ModelTrainer._ensure_mlflow_backend()
    mlflow.set_experiment(config["modeling"]["mlflow_experiment"])

    evaluator = PostTrainingEvaluator(config)
    prep = DatasetPrep(config)
    excluded = set(config["modeling"].get("excluded_features", []))
    figures_dir = Path(config["base"]["figures_dir"])

    out: Dict[str, Dict[str, dict]] = {}
    for season, season_paths in input_paths.items():
        if season == "ref_path":
            continue

        df_test = pd.read_csv(season_paths["test"])
        X_test, y_test = df_test.drop(columns=["label"]), df_test["label"]
        feature_cols = [c for c in X_test.columns if not c.startswith("_") and c not in excluded]

        x_coords = df_test["_x"].to_numpy() if "_x" in df_test.columns else None
        y_coords = df_test["_y"].to_numpy() if "_y" in df_test.columns else None
        fire_test_gdf = gpd.read_file(season_paths["fire_test"]) if "fire_test" in season_paths else None

        # "full" is the same feature domain as "test" but without the
        # NaN-label drop, so the susceptibility raster covers every
        # in-domain pixel — not just ones with a ground-truth density
        # label. Falls back to the label-filtered test set if a "full"
        # path wasn't supplied (e.g. older cached pipeline state).
        X_full = full_x_coords = full_y_coords = None
        if "full" in season_paths:
            df_full = pd.read_csv(season_paths["full"])
            X_full = df_full[feature_cols]
            full_x_coords = df_full["_x"].to_numpy() if "_x" in df_full.columns else None
            full_y_coords = df_full["_y"].to_numpy() if "_y" in df_full.columns else None

        out[season] = {}
        proba_by_model = {}
        for model_name, artifact_path in season_paths.get("artifacts", {}).items():
            logger.info(f"[stage_evaluate] [{season}][{model_name}] Reloading + evaluating...")
            artifact_path = Path(artifact_path)
            model = joblib.load(artifact_path / "model.joblib")

            # Same categorical encoding this model was trained under (fit on
            # train data only, persisted at training time) must be re-applied
            # here — X_test/X_full are freshly reloaded from CSV, so they are
            # still in raw/unencoded form regardless of what dataset_prep did
            # inside trainer.train_one() during the training stage.
            manifest = json.loads((artifact_path / "manifest.json").read_text())
            encoding_meta = manifest.get("categorical_encoding", {"kind": "none", "columns": [], "categories": {}})
            cat_positions = cat_features_of(model)

            X_test_encoded = prep.apply_categorical_encoding(X_test[feature_cols], encoding_meta)
            X_test_arr = to_model_array(X_test_encoded, cat_positions)

            proba_by_model[model_name] = model.predict_proba(X_test_arr)

            X_full_encoded = None
            if X_full is not None:
                X_full_encoded = prep.apply_categorical_encoding(X_full, encoding_meta)

            result = evaluator.evaluate(
                model, X_test_encoded, y_test, season, model_name,
                ref_path, fire_test_gdf, x_coords, y_coords,
                X_full=X_full_encoded, full_x_coords=full_x_coords, full_y_coords=full_y_coords,
            )
            out[season][model_name] = result

            raster_path = result.get("susceptibility_map_path")
            if raster_path is not None and dem_path.exists():
                try:
                    viz.render_susceptibility_map(
                        raster_path, dem_path, figures_dir,
                        season=season, model_name=model_name,
                        fire_points_gdf=fire_test_gdf,
                    )
                    logger.info(f"[stage_evaluate] [{season}][{model_name}] Susceptibility map figure written.")
                except Exception as exc:
                    logger.warning(f"[stage_evaluate] [{season}][{model_name}] Susceptibility map figure failed, skipping: {exc}")
            elif raster_path is not None:
                logger.warning(f"[stage_evaluate] [{season}][{model_name}] No DEM found at {dem_path}; skipping susceptibility map figure.")

        if proba_by_model:
            n_classes = next(iter(proba_by_model.values())).shape[1]
            try:
                viz.plot_roc_curves(y_test.to_numpy(), proba_by_model, n_classes, figures_dir, season=season)
                logger.info(
                    f"[stage_evaluate] [{season}] ROC comparison chart written "
                    f"({len(proba_by_model)} model(s))."
                )
            except Exception as exc:
                logger.warning(f"[stage_evaluate] [{season}] ROC comparison chart failed, skipping: {exc}")

    return out