"""Regression coverage: stage_preprocessing must also emit a "full" gold
CSV alongside train/test -- the same feature domain as "test" but without
dropping rows for a missing ground-truth label. This is what lets
stage_evaluate predict a susceptibility class for every in-domain pixel
rather than only ones with a density label."""

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
    cfg["labels"]["clean_labels"] = False  # isolate from LabelCleaner
    cfg["processing"]["force_recompute"] = False
    return cfg


def test_stage_preprocessing_full_domain_keeps_unlabeled_pixels(tmp_path, preprocessing_config):
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    train_csv = silver_dir / "dataset_train_summer.csv"
    test_csv = silver_dir / "dataset_test_summer.csv"

    # 5 of the 25 in-domain pixels have no density label -- e.g. trimmed
    # to NaN by the bottom trim_bottom_pct of KernelDensityClassifier.
    labels = [0, 1, 2, 3] * 5 + [float("nan")] * 5
    _write_raw_dataset(train_csv, labels)
    _write_raw_dataset(test_csv, labels)

    input_paths = {
        "ref_path": tmp_path / "ref.tif",  # unused when clean_labels is False
        "summer": {"train": train_csv, "test": test_csv},
    }

    out = stage_preprocessing(preprocessing_config, input_paths)

    assert "full" in out["summer"]
    df_test = pd.read_csv(out["summer"]["test"])
    df_full = pd.read_csv(out["summer"]["full"])

    assert len(df_test) == 20, "prepare_test should still drop the 5 NaN-label rows"
    assert len(df_full) == 25, "prepare_full_domain must keep every in-domain pixel"
    assert df_full["label"].isna().sum() == 5


def test_stage_preprocessing_full_domain_survives_cache_hit(tmp_path, preprocessing_config):
    """The cached-rerun path must also return a 'full' key, not just the
    first-run path -- otherwise stage_evaluate breaks on a warm cache."""
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    train_csv = silver_dir / "dataset_train_summer.csv"
    test_csv = silver_dir / "dataset_test_summer.csv"
    labels = [0, 1, 2, 3] * 5
    _write_raw_dataset(train_csv, labels)
    _write_raw_dataset(test_csv, labels)

    input_paths = {
        "ref_path": tmp_path / "ref.tif",
        "summer": {"train": train_csv, "test": test_csv},
    }

    stage_preprocessing(preprocessing_config, input_paths)
    out = stage_preprocessing(preprocessing_config, input_paths)  # cache hit

    assert "full" in out["summer"]
    assert out["summer"]["full"].exists()
