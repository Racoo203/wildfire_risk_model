"""Stage: exploratory data analysis.

Reproduces eda.ipynb's figures as a pipeline stage: NaN coverage
(pre-imputation), VIF/Spearman correlation (post-imputation), and
class balance before/after label cleaning. All figures land in
figures/manifest.json via viz/.

    stage_eda(config, input_paths) -> {
        "<season>": {"vif": Path, "spearman": Path, "class_balance": Path, "nan_coverage": Path},
        ...
    }

`input_paths` must contain:
    "raw": {"<season>": {"train": Path, "test": Path}}     # stage_integration output
    "clean": {"<season>": {"train": Path, "test": Path}}   # stage_preprocessing output
"""
from typing import Dict

import pandas as pd

from .. import viz
from ..utils.logger import setup_logger


def stage_eda(config: dict, input_paths: dict) -> Dict[str, dict]:
    """
    NOTE: class_balance compares train split only (raw vs. cleaned) —
    label cleaning (LabelCleaner.clean_flat) only ever runs on train per
    the documented no-test-leakage principle (see modeling/dataset_prep.py
    docstring), so a train-vs-clean comparison is the only one that's
    meaningful; test labels are never mutated and would show 0% change.
    """
    # logger = setup_logger(log_file=config["logging"]["log_path"], level=config["logging"]["level"])
    logger = setup_logger()
    figures_dir = config["base"]["figures_dir"]

    raw = input_paths["raw"]
    clean = input_paths["clean"]

    out: Dict[str, dict] = {}
    for season in raw:
        logger.info(f"[stage_eda] [{season}] Generating EDA figures...")
        df_raw = pd.read_csv(raw[season]["train"])
        df_clean = pd.read_csv(clean[season]["train"])

        feature_cols = [c for c in df_clean.columns if c not in ("label", "_x", "_y", "tas", "tasmin")]

        nan_arrays = {c: df_raw[c].to_numpy() for c in df_raw.columns if c != "label"}
        nan_path = viz.plot_nan_coverage(nan_arrays, figures_dir, season=season)

        vif_paths = viz.plot_vif_correlation(df_clean, feature_cols, figures_dir, season=season)

        # Class balance: raw label column pre-cleaning vs clean label column
        # post-cleaning. plot_class_balance's internal counting is shape-
        # agnostic, so flat 1-D arrays work the same as the old raster inputs.
        class_balance_path = viz.plot_class_balance(
            df_raw["label"].to_numpy(), df_clean["label"].to_numpy(), figures_dir, season=season
        )

        out[season] = {
            "nan_coverage": nan_path,
            "vif": vif_paths["vif"],
            "spearman": vif_paths["spearman"],
            "class_balance": class_balance_path,
        }

    return out