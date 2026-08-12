"""Regression coverage for the class-4 loss incident: stage_preprocessing's
gold-layer cache was keyed on output-file existence alone, so when
stage_integration regenerated its raw CSVs with a new label scheme
(dataset_train_summer.csv going from 4 to 5 classes), the stale 4-class
gold CSV from a prior run kept getting served untouched -- class 4 never
even reached stage_train, let alone got dropped there.

These tests exercise stage_preprocessing itself (not just the checkpoint
helper unit tests) to confirm the fix at the level the bug was actually
observed: a second call with regenerated raw input must produce a clean
CSV reflecting the new data, not silently reuse the old one.
"""
import time

import pandas as pd
import pytest

from wildfire_susceptibility.pipeline.stage_preprocessing import stage_preprocessing


def _write_raw_dataset(path, labels):
    n = len(labels)
    df = pd.DataFrame({
        "elevation": [10.0] * n,
        "ndvi": [0.5] * n,
        "tas": [15.0] * n,
        "tasmax": [20.0] * n,
        "tasmin": [10.0] * n,
        "rainfall": [2.0] * n,
        "sfcWind": [3.0] * n,
        "hurs": [70.0] * n,
        "diurnal_range": [10.0] * n,
        "_x": [float(i) for i in range(n)],
        "_y": [float(i) for i in range(n)],
        "label": labels,
    })
    df.to_csv(path, index=False)


@pytest.fixture
def preprocessing_config(tmp_path, minimal_modeling_config):
    cfg = minimal_modeling_config
    cfg["base"]["gold_dir"] = str(tmp_path / "gold")
    cfg["labels"]["clean_labels"] = False  # isolate caching behavior from LabelCleaner
    cfg["processing"]["force_recompute"] = False
    return cfg


def test_stage_preprocessing_regenerates_when_raw_input_changes(tmp_path, preprocessing_config):
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    train_csv = silver_dir / "dataset_train_summer.csv"
    test_csv = silver_dir / "dataset_test_summer.csv"

    # First run: old raw data, 4 classes (0-3) -- matches the pre-incident state.
    _write_raw_dataset(train_csv, [0, 1, 2, 3] * 5)
    _write_raw_dataset(test_csv, [0, 1, 2, 3] * 5)

    input_paths = {
        "ref_path": tmp_path / "ref.tif",  # unused when clean_labels is False
        "summer": {"train": train_csv, "test": test_csv},
    }

    out = stage_preprocessing(preprocessing_config, input_paths)
    first_labels = set(pd.read_csv(out["summer"]["train"])["label"].unique())
    assert first_labels == {0, 1, 2, 3}

    # Raw data regenerated upstream with a 5th class -- exactly what
    # stage_integration did in the incident. mtime must advance.
    time.sleep(0.01)
    _write_raw_dataset(train_csv, [0, 1, 2, 3, 4] * 5)
    _write_raw_dataset(test_csv, [0, 1, 2, 3, 4] * 5)

    out = stage_preprocessing(preprocessing_config, input_paths)
    second_labels = set(pd.read_csv(out["summer"]["train"])["label"].unique())

    assert second_labels == {0, 1, 2, 3, 4}, (
        "stage_preprocessing served a stale cached clean CSV instead of "
        "regenerating from the updated raw input -- this is the exact "
        "class-4 loss bug."
    )


def test_stage_preprocessing_still_caches_when_nothing_changed(tmp_path, preprocessing_config, monkeypatch):
    """The fix must not defeat caching entirely -- an unchanged rerun
    should still skip recomputation."""
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    train_csv = silver_dir / "dataset_train_summer.csv"
    test_csv = silver_dir / "dataset_test_summer.csv"
    _write_raw_dataset(train_csv, [0, 1, 2, 3] * 5)
    _write_raw_dataset(test_csv, [0, 1, 2, 3] * 5)

    input_paths = {
        "ref_path": tmp_path / "ref.tif",
        "summer": {"train": train_csv, "test": test_csv},
    }

    stage_preprocessing(preprocessing_config, input_paths)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("prepare_train should not run on an unchanged rerun")

    monkeypatch.setattr(
        "wildfire_susceptibility.pipeline.stage_preprocessing.DatasetPrep.prepare_train",
        _fail_if_called,
    )

    stage_preprocessing(preprocessing_config, input_paths)  # must hit the cache, never call prepare_train
