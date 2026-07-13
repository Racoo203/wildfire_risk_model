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
import sys
from pathlib import Path
from typing import List, Optional

from ..config.loader import ConfigLoader, DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_FILES
from .stage_static import stage_static
from .stage_seasonal import stage_seasonal
from .stage_labels import stage_labels
from .stage_integration import stage_integration

from .stage_preprocessing import stage_preprocessing
from .stage_eda import stage_eda
from .stage_train import stage_train
from .stage_evaluate import stage_evaluate

from .stage_selection import stage_selection

logger = logging.getLogger("wildfire_susceptibility.pipeline.run_pipeline")

STAGE_ORDER = [
    "static", "seasonal", "labels", "integration",
    "preprocessing", "eda", "train", "evaluate", "selection",
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
    parser.add_argument("--stage", choices=[*STAGE_ORDER, "all"], default="all")
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--config-file", action="append", dest="config_files", default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout,
    )

    cfg_obj = ConfigLoader.load(config_dir=args.config_dir, files=args.config_files or DEFAULT_CONFIG_FILES)
    config = cfg_obj.model_dump(mode="python")

    state = _load_state()
    stages = STAGE_ORDER if args.stage == "all" else [args.stage]

    for stage in stages:
        logger.info(f"=== Running stage: {stage} ===")
        state = run_stage(stage, config, state)
        _save_state(state)
        logger.info(f"=== Stage complete: {stage} ===")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())