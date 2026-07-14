"""Pipeline orchestration entry point.

Chains the five stage functions together, persisting each stage's output
path-dict to a JSON state file on disk between invocations — so
`--stage train` run tomorrow can pick up exactly where
`--stage dataset_assembly` left off today, without recomputing anything
or holding results in memory across process boundaries.

Usage:
    python -m wildfire_susceptibility.pipeline.run_pipeline --stage preprocess
    python -m wildfire_susceptibility.pipeline.run_pipeline --stage all
    python -m wildfire_susceptibility.pipeline.run_pipeline --stage train --config-file _working.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
# import sys
from pathlib import Path
from typing import List, Optional

from wildfire_susceptibility.config.loader import ConfigLoader, DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_FILES
from wildfire_susceptibility.pipeline.stage_static import stage_static
from wildfire_susceptibility.pipeline.stage_seasonal import stage_seasonal
from wildfire_susceptibility.pipeline.stage_labels import stage_labels
from wildfire_susceptibility.pipeline.stage_integration import stage_integration

from wildfire_susceptibility.pipeline.stage_preprocessing import stage_preprocessing
from wildfire_susceptibility.pipeline.stage_eda import stage_eda
from wildfire_susceptibility.pipeline.stage_train import stage_train
from wildfire_susceptibility.pipeline.stage_evaluate import stage_evaluate

from wildfire_susceptibility.pipeline.stage_selection import stage_selection

from wildfire_susceptibility.utils.logger import setup_logger


logger = logging.getLogger()

STAGE_ORDER = [
    "static", "seasonal", "labels", "integration",
    "preprocessing", "eda", "train", "evaluate", "selection",
]

GEN_ONLY = [
    "static", "seasonal", "labels", "integration",
]

TRAIN_ONLY = [
    "preprocessing", "train", "evaluate", "selection",
]

STATE_PATH = Path("models/.pipeline_state.json")

def _load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


def run_stage(stage: str, config: dict, state: dict) -> dict:
    if stage == "static":
        state["static"] = stage_static(config, {})

    elif stage == "seasonal":
        state["seasonal"] = stage_seasonal(config, {"ref_path": state["static"]["ref_path"]})

    elif stage == "labels":
        state["labels"] = stage_labels(config, {"ref_path": state["static"]["ref_path"]})

    elif stage == "integration":
        state["integration"] = stage_integration(config, {
            "ref_path": state["static"]["ref_path"],
            "static": state["static"],
            "seasonal": state["seasonal"],
            "labels": state["labels"],
        })

    elif stage == "preprocessing":
        state["preprocessing"] = stage_preprocessing(config, {
            "ref_path": state["static"]["ref_path"],
            **state["integration"],
        })

    elif stage == "eda":
        state["eda"] = stage_eda(config, {"raw": state["integration"], "clean": state["preprocessing"]})

    elif stage == "train":
        train_input = {"ref_path": state["static"]["ref_path"]}
        for season, splits in state["preprocessing"].items():
            train_input[season] = {
                **splits,
                "fire_train": state["labels"][season]["fire_train"],
                "fire_test": state["labels"][season]["fire_test"],
            }
        state["train"] = stage_train(config, train_input)

    elif stage == "evaluate":
        eval_input = {"ref_path": state["static"]["ref_path"]}
        for season, splits in state["preprocessing"].items():
            eval_input[season] = {
                "test": splits["test"],
                "fire_test": state["labels"][season]["fire_test"],
                "artifacts": state["train"][season],
            }
        state["evaluate"] = stage_evaluate(config, eval_input)

    elif stage == "selection":
        state["selection"] = stage_selection(config, {"train": state["train"], "evaluate": state["evaluate"]})

    else:
        raise ValueError(f"Unknown stage '{stage}'")

    return state


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=[*STAGE_ORDER, "dataset", "train", "all"], default="all")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--config-file", action="append", dest="config_files", default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logger = setup_logger()

    if args.experiment:
        cfg_obj = ConfigLoader.load_experiment(name=args.experiment)
        config = cfg_obj.model_dump(mode="python")
    else:
        cfg_obj = ConfigLoader.load(config_dir=args.config_dir, files=args.config_files or DEFAULT_CONFIG_FILES)
        config = cfg_obj.model_dump(mode="python")

    state = _load_state()

    if args.stage == "all":
        stages = STAGE_ORDER
    elif args.stage == "dataset":
        stages = GEN_ONLY
    elif args.stage == "train":
        stages = GEN_ONLY + TRAIN_ONLY
    else: 
        stages = [args.stage]

    for stage in stages:
        logger.info(f"=== Running stage: {stage} ===")
        state = run_stage(stage, config, state)
        _save_state(state)
        logger.info(f"=== Stage complete: {stage} ===")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())