# wildfire_susceptibility/pipeline/train.py
"""Top-level training orchestration: loops over active seasons and
configured models, delegating all per-model mechanics to ModelTrainer
(modeling/training/trainer.py). This file has no training logic of its
own — it only wires dataset prep output into ModelTrainer calls."""

from typing import Callable, Dict, Optional
from pathlib import Path
import logging

import pandas as pd

from ..modeling.dataset_prep import DatasetPrep
from ..modeling.training.trainer import ModelTrainer

logger = logging.getLogger(__name__)


def run_training_pipeline(
    config: dict,
    dataset_paths: Dict[str, Dict[str, object]],
    ref_path: Path,
    progress_callback: Optional[Callable[[str, str, int, float], None]] = None,
) -> Dict[str, Dict[str, dict]]:
    """
    dataset_paths: {season: {"train": path, "test": path,
                              "fire_train": gdf, "fire_test": gdf}}.

    Runs feature prep, hyperparameter search, final refit, and post-
    training evaluation for every active season and configured model.
    Designed to be started once and left running unattended.
    """
    prep = DatasetPrep(config)
    climate_vars = tuple(config["data_sources"]["haduk"]["sources"])
    trainer = ModelTrainer(config)
    block_size = config["modeling"].get("spatial_block_size_m", 5000.0)

    all_results: Dict[str, Dict[str, dict]] = {}

    for season, paths in dataset_paths.items():
        raw = prep.load_train_test({"train": paths["train"], "test": paths["test"]})
        df_train = prep.prepare_train(raw["train"], season, ref_path, climate_vars)
        df_test = prep.prepare_test(raw["test"], season, climate_vars)

        x_coords = df_test["_x"].to_numpy() if "_x" in df_test.columns else None
        y_coords = df_test["_y"].to_numpy() if "_y" in df_test.columns else None

        X_train, y_train = df_train.drop(columns=["label"]), df_train["label"]
        X_test, y_test = df_test.drop(columns=["label"]), df_test["label"]

        feature_cols = [c for c in X_train.columns if not c.startswith("_")]
        groups_train = (
            prep.assign_spatial_blocks(X_train, block_size_m=block_size)
            if config["modeling"].get("cv_strategy") in ("spatial", "both")
            else None
        )

        fire_test_gdf = paths.get("fire_test")

        season_results = {}
        for model_name in config["modeling"]["models"]:
            cb = (lambda t, v, _s=season, _m=model_name: progress_callback(_s, _m, t, v)) if progress_callback else None
            season_results[model_name] = trainer.train_one(
                season, model_name,
                X_train[feature_cols], y_train,
                X_test[feature_cols], y_test,
                groups_train=groups_train,
                progress_callback=cb,
                ref_path=ref_path,
                fire_test_gdf=fire_test_gdf,
                x_coords=x_coords,
                y_coords=y_coords,
            )

        all_results[season] = season_results
        logger.info(f"[{season}] all models trained and evaluated.")

    return all_results


def results_to_dataframe(all_results: Dict[str, Dict[str, dict]]) -> pd.DataFrame:
    """Flatten run_training_pipeline()'s nested result dict into one comparison table."""
    rows = []
    for season, models in all_results.items():
        for model_name, r in models.items():
            tf = r.get("time_forward_validation") or {}
            rows.append({
                "season": season,
                "model": model_name,
                "cv_auc_standard": r["cv_auc_standard"],
                "cv_auc_spatial": r["cv_auc_spatial"],
                "val_auc": r["val_auc"],
                "val_f1": r["val_f1"],
                "tf_pct_medium_plus": tf.get("pct_medium_plus"),
                "susceptibility_map": r.get("susceptibility_map_path"),
            })
    return pd.DataFrame(rows).sort_values(["season", "cv_auc_standard"], ascending=[True, False])