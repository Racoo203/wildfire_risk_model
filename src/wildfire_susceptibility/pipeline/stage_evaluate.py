"""Stage: post-training evaluation.

Reloads model artifacts from stage_train and evaluates against the
CLEANED test set (stage_preprocessing's output) — never retrains.

`input_paths` must contain:
    "ref_path": Path
    "<season>": {"test": Path, "fire_test": Path, "artifacts": {"<model_name>": Path}}
"""
from pathlib import Path
from typing import Dict

import geopandas as gpd
import joblib
import pandas as pd

from ..modeling.training.evaluation import PostTrainingEvaluator
from ..utils.logger import setup_logger


def stage_evaluate(config: dict, input_paths: dict) -> Dict[str, Dict[str, dict]]:
    logger = setup_logger(log_file=config["logging"]["log_path"], level=config["logging"]["level"])
    ref_path = input_paths["ref_path"]
    evaluator = PostTrainingEvaluator(config)
    excluded = set(config["modeling"].get("excluded_features", []))

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
        for model_name, artifact_path in season_paths.get("artifacts", {}).items():
            logger.info(f"[stage_evaluate] [{season}][{model_name}] Reloading + evaluating...")
            model = joblib.load(Path(artifact_path) / "model.joblib")
            out[season][model_name] = evaluator.evaluate(
                model, X_test[feature_cols], y_test, season, model_name,
                ref_path, fire_test_gdf, x_coords, y_coords,
            )
    return out