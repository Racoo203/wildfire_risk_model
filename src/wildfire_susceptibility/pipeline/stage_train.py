"""Stage: hyperparameter search + final refit.

Loads stage_preprocessing's cleaned datasets, runs ModelTrainer.train_one
with post-training evaluation SKIPPED (that's stage_evaluate's job now),
and writes model + params + manifest to models/<season>/<model>/<cfg_sig>/.

    stage_train(config, input_paths) -> {
        "<season>": {"<model_name>": Path},  # artifact_dir per model
        ...
    }

`input_paths` must contain:
    "ref_path": Path
    "<season>": {"train": Path, "test": Path, "fire_train": Path, "fire_test": Path}
"""
import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from ..config.signature import compute_cfg_sig
from ..modeling.training.trainer import ModelTrainer
from ..modeling.training.artifacts import artifact_dir
from ..modeling.dataset_prep import DatasetPrep
from ..utils.logger import setup_logger


def stage_train(config: dict, input_paths: dict) -> Dict[str, Dict[str, Path]]:
    # logger = setup_logger(log_file=config["logging"]["log_path"], level=config["logging"]["level"])
    logger = setup_logger()
    prep = DatasetPrep(config)
    trainer = ModelTrainer(config)
    block_size = config["modeling"].get("spatial_block_size_m", 5000.0)

    out: Dict[str, Dict[str, Path]] = {}
    for season, season_paths in input_paths.items():
        if season == "ref_path":
            continue

        logger.info(f"[stage_train] [{season}] Loading cleaned datasets...")
        df_train = pd.read_csv(season_paths["train"])
        df_test = pd.read_csv(season_paths["test"])

        X_train, y_train = df_train.drop(columns=["label"]), df_train["label"]
        X_test, y_test = df_test.drop(columns=["label"]), df_test["label"]

        excluded = set(config["modeling"].get("excluded_features", []))
        feature_cols = [c for c in X_train.columns if not c.startswith("_") and c not in excluded]

        groups_train = (
            prep.assign_spatial_blocks(X_train, block_size_m=block_size)
            if config["modeling"].get("cv_strategy") in ("spatial", "both")
            else None
        )

        out[season] = {}
        for model_name in config["modeling"]["models"]:
            logger.info(f"[stage_train] [{season}][{model_name}] Search + refit...")
            result = trainer.train_one(
                season, model_name,
                X_train[feature_cols], y_train,
                X_test[feature_cols], y_test,
                groups_train=groups_train,
                run_post_training_evaluation=False,  # deferred to stage_evaluate
            )
            out[season][model_name] = _write_artifact(config, season, model_name, result)

    return out


def _write_artifact(config: dict, season: str, model_name: str, result: dict) -> Path:
    out_dir = artifact_dir(config, season, model_name)

    joblib.dump(result["model"], out_dir / "model.joblib")
    (out_dir / "best_params.json").write_text(json.dumps(result["best_params"], indent=2))

    manifest = {
        "season": season,
        "model_name": model_name,
        "cfg_sig": compute_cfg_sig(config),
        "cv_auc_standard": result.get("cv_auc_standard"),
        "cv_auc_spatial": result.get("cv_auc_spatial"),
        "cv_auc_spatial_folds": result.get("cv_auc_spatial_folds"),
        "val_auc": result.get("val_auc"),
        "val_f1": result.get("val_f1"),
        "config_snapshot": config,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return out_dir