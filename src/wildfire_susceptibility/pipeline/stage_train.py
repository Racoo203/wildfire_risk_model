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
from ..modeling.cv import requires_spatial_groups
from ..modeling.training.artifacts import artifact_dir
from ..modeling.dataset_prep import DatasetPrep
from ..utils.logger import setup_logger


def stage_train(config: dict, input_paths: dict) -> Dict[str, Dict[str, Path]]:
    logger = setup_logger()
    prep = DatasetPrep(config)
    trainer = ModelTrainer(config)
    block_size = config["modeling"].get("spatial_block_size_m", 5000.0)

    # Whether spatial blocks need to be assigned at all is now determined
    # by the *active* CV strategy on the trainer (which already resolved
    # cv_strategy="both" down to a concrete primary strategy), not by
    # re-deriving it from the raw config string here.
    needs_groups = requires_spatial_groups(trainer.cv_strategy.name) or config["modeling"].get("cv_strategy") == "both"

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

        cv_groups = (
            prep.assign_spatial_blocks(X_train, block_size_m=block_size)
            if needs_groups
            else None
        )

        out[season] = {}
        for model_name in config["modeling"]["models"]:
            logger.info(f"[stage_train] [{season}][{model_name}] Search + refit...")
            result = trainer.train_one(
                season, model_name,
                X_train[feature_cols], y_train,
                X_test[feature_cols], y_test,
                groups_train=cv_groups,
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