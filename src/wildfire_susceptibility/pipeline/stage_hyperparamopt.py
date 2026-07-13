"""Stage: model training.

Loads assembled per-season train/test CSVs and fire-incident gpkgs from
disk — never carries them in-memory across a stage boundary — runs
DatasetPrep (imputation + label cleaning) and ModelTrainer per
(season, model), and writes artifacts to
models/<season>/<model_name>/<cfg_sig>/. This is the actual fix for the
resumability gap: previously ModelTrainer only ran in-process from
pipeline/train.py with live DataFrames passed in directly.

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

import geopandas as gpd
import joblib
import pandas as pd

from ..config.signature import compute_cfg_sig
from ..modeling.dataset_prep import DatasetPrep
from ..modeling.training.trainer import ModelTrainer
from ..modeling.training.artifacts import artifact_dir
from ..utils.logger import setup_logger


def stage_train(config: dict, input_paths: dict) -> Dict[str, Dict[str, Path]]:
    # logger = setup_logger(
    #     log_file=config["logging"]["log_path"],
    #     level=config["logging"]["level"],
    # )
    logger = setup_logger()
    ref_path = input_paths["ref_path"]
    prep = DatasetPrep(config)
    trainer = ModelTrainer(config)
    climate_vars = tuple(config["data_sources"]["haduk"]["sources"]) + ("diurnal_range",)
    block_size = config["modeling"].get("spatial_block_size_m", 5000.0)

    out: Dict[str, Dict[str, Path]] = {}
    for season, season_paths in input_paths.items():
        if season == "ref_path":
            continue

        logger.info(f"[stage_train] [{season}] Loading + preparing datasets...")
        df_train_raw = pd.read_csv(season_paths["train"])
        df_test_raw = pd.read_csv(season_paths["test"])

        df_train = prep.prepare_train(df_train_raw, season, ref_path, climate_vars)
        df_test = prep.prepare_test(df_test_raw, season, climate_vars)

        X_train, y_train = df_train.drop(columns=["label"]), df_train["label"]
        X_test, y_test = df_test.drop(columns=["label"]), df_test["label"]

        excluded = set(config["modeling"].get("excluded_features", []))
        feature_cols = [c for c in X_train.columns if not c.startswith("_") and c not in excluded]

        groups_train = (
            prep.assign_spatial_blocks(X_train, block_size_m=block_size)
            if config["modeling"].get("cv_strategy") in ("spatial", "both")
            else None
        )

        fire_test_gdf = gpd.read_file(season_paths["fire_test"]) if "fire_test" in season_paths else None
        x_coords = df_test["_x"].to_numpy() if "_x" in df_test.columns else None
        y_coords = df_test["_y"].to_numpy() if "_y" in df_test.columns else None

        out[season] = {}
        for model_name in config["modeling"]["models"]:
            logger.info(f"[stage_train] [{season}][{model_name}] Training...")
            result = trainer.train_one(
                season, model_name,
                X_train[feature_cols], y_train,
                X_test[feature_cols], y_test,
                groups_train=groups_train,
                ref_path=ref_path,
                fire_test_gdf=fire_test_gdf,
                x_coords=x_coords,
                y_coords=y_coords,
            )
            out[season][model_name] = _write_artifact(config, season, model_name, result)

    return out


def _write_artifact(config: dict, season: str, model_name: str, result: dict) -> Path:
    """Persist model + best_params + manifest (with full config_snapshot)
    to models/<season>/<model_name>/<cfg_sig>/."""
    out_dir = artifact_dir(config, season, model_name)

    joblib.dump(result["model"], out_dir / "model.joblib")
    (out_dir / "best_params.json").write_text(json.dumps(result["best_params"], indent=2))

    manifest = {
        "season": season,
        "model_name": model_name,
        "cfg_sig": compute_cfg_sig(config),
        "cv_auc_standard": result.get("cv_auc_standard"),
        "cv_auc_spatial": result.get("cv_auc_spatial"),
        "val_auc": result.get("val_auc"),
        "val_f1": result.get("val_f1"),
        "config_snapshot": config,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    return out_dir