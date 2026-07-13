"""Stage: preprocessing — imputation + label cleaning.

Reads stage_integration's raw CSVs, applies the shared (model-agnostic)
cleanup steps via DatasetPrep, and writes cleaned CSVs to the gold
layer. Per-model scaling is intentionally NOT done here — it depends
on each model's needs_scaling() and stays inside stage_train.

    stage_preprocessing(config, input_paths) -> {
        "<season>": {"train": Path, "test": Path},
        ...
    }

`input_paths` must contain:
    "ref_path": Path
    "<season>": {"train": Path, "test": Path}   # from stage_integration
"""
from pathlib import Path
from typing import Dict

import pandas as pd

from ..modeling.dataset_prep import DatasetPrep
from ..utils.logger import setup_logger


def stage_preprocessing(config: dict, input_paths: dict) -> Dict[str, dict]:
    logger = setup_logger()
    ref_path = input_paths["ref_path"]
    prep = DatasetPrep(config)
    climate_vars = tuple(config["data_sources"]["haduk"]["sources"]) + ("diurnal_range",)

    print(config)
    gold_dir = Path(config["base"]["gold_dir"])
    gold_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, dict] = {}
    for season, splits in input_paths.items():
        if season == "ref_path":
            continue

        train_out = gold_dir / f"dataset_train_{season}_clean.csv"
        test_out = gold_dir / f"dataset_test_{season}_clean.csv"

        if not config["processing"]["force_recompute"] and train_out.exists() and test_out.exists():
            logger.info(f"[stage_preprocessing] [CACHED] {season} -> {train_out.name}, {test_out.name}")
            out[season] = {"train": train_out, "test": test_out}
            continue

        logger.info(f"[stage_preprocessing] [{season}] Imputing + cleaning labels...")
        df_train_raw = pd.read_csv(splits["train"])
        df_test_raw = pd.read_csv(splits["test"])

        df_train = prep.prepare_train(df_train_raw, season, ref_path, climate_vars)
        df_test = prep.prepare_test(df_test_raw, season, climate_vars)

        df_train.to_csv(train_out, index=False)
        df_test.to_csv(test_out, index=False)
        out[season] = {"train": train_out, "test": test_out}

    return out