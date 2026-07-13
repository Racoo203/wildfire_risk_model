"""Stage: dataset integration.

Stacks static + seasonal features + raw labels into one row-per-pixel
CSV per (season, split). Output is RAW — no imputation or label
cleaning yet; that's stage_preprocessing's job.

    stage_integration(config, input_paths) -> {
        "<season>": {"train": Path, "test": Path},
        ...
    }

`input_paths` must contain:
    "ref_path": Path
    "static": {...}                  # from stage_static
    "seasonal": {"<season>": {...}}  # from stage_seasonal
    "labels": {"<season>": {...}}    # from stage_labels
"""
from pathlib import Path
from typing import Dict

from ..core.raster import RasterManager
from ..utils.logger import setup_logger

_NON_RASTER_STATIC_KEYS = {"ref_path", "boundary"}


def stage_integration(config: dict, input_paths: dict) -> Dict[str, dict]:
    # logger = setup_logger(log_file=config["logging"]["log_path"], level=config["logging"]["level"])
    logger = setup_logger()
    ref_path = input_paths["ref_path"]
    static_features = {k: v for k, v in input_paths["static"].items() if k not in _NON_RASTER_STATIC_KEYS}
    seasonal = input_paths["seasonal"]
    labels = input_paths["labels"]

    model_data_dir = Path(config["base"]["model_data_dir"])
    model_data_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, dict] = {}
    for season in seasonal:
        out[season] = {}
        for split in ("train", "test"):
            seasonal_paths = dict(seasonal[season][split])
            label_paths = dict(labels[season][split])
            raw_labels = label_paths.pop("raw_labels")

            all_features = {**static_features, **seasonal_paths, **label_paths}  # d_fires, if present, lives in label_paths
            out_csv = model_data_dir / f"dataset_{split}_{season}.csv"

            if not config["processing"]["force_recompute"] and out_csv.exists():
                logger.info(f"[stage_integration] [CACHED] {season}/{split} -> {out_csv.name}")
                out[season][split] = out_csv
                continue

            RasterManager.stack_to_dataframe({**all_features, "label": raw_labels}, ref_path, out_csv)
            logger.info(f"[stage_integration] [{season}][{split}] Assembled -> {out_csv.name}")
            out[season][split] = out_csv

    return out