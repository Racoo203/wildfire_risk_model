from pathlib import Path
from typing import Dict

import geopandas as gpd
import joblib
import mlflow
import pandas as pd

from .. import viz
from ..modeling.training.evaluation import PostTrainingEvaluator
from ..modeling.training.trainer import ModelTrainer
from ..utils.logger import setup_logger


def stage_evaluate(config: dict, input_paths: dict) -> Dict[str, Dict[str, dict]]:
    # logger = setup_logger(log_file=config["logging"]["log_path"], level=config["logging"]["level"])
    logger = setup_logger()
    ref_path = input_paths["ref_path"]

    ModelTrainer._ensure_mlflow_backend()
    mlflow.set_experiment(config["modeling"]["mlflow_experiment"])

    evaluator = PostTrainingEvaluator(config)
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

        out[season] = {}
        proba_by_model = {}
        for model_name, artifact_path in season_paths.get("artifacts", {}).items():
            logger.info(f"[stage_evaluate] [{season}][{model_name}] Reloading + evaluating...")
            model = joblib.load(Path(artifact_path) / "model.joblib")
            proba_by_model[model_name] = model.predict_proba(X_test[feature_cols].values)
            out[season][model_name] = evaluator.evaluate(
                model, X_test[feature_cols], y_test, season, model_name,
                ref_path, fire_test_gdf, x_coords, y_coords,
            )

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