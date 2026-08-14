"""Orchestration wiring tests for run_pipeline.py::run_stage().

These deliberately mock every stage_* function — the goal is to verify
run_stage() assembles the correct input_paths dict for each stage from
prior state, not to exercise real feature-building logic (that's what
the integration-level dataset/train smoke tests are for).
"""
from pathlib import Path

import pytest

from wildfire_susceptibility.pipeline import run_pipeline as rp


@pytest.fixture
def fake_static_state():
    return {"static": {"ref_path": Path("ref.tif"), "boundary": Path("boundary.shp")}}


@pytest.fixture
def fake_seasonal_labels_state(fake_static_state):
    state = dict(fake_static_state)
    state["seasonal"] = {
        "spring": {
            "train": {"tas": Path("tas_train.tif")},
            "test": {"tas": Path("tas_test.tif")},
        }
    }
    state["labels"] = {
        "spring": {
            "train": {"raw_labels": Path("labels_train.tif")},
            "test": {"raw_labels": Path("labels_test.tif")},
            "fire_train": Path("fire_train.gpkg"),
            "fire_test": Path("fire_test.gpkg"),
        }
    }
    return state


@pytest.fixture
def fake_integration_state(fake_seasonal_labels_state):
    state = dict(fake_seasonal_labels_state)
    state["integration"] = {
        "spring": {"train": Path("dataset_train_spring.csv"), "test": Path("dataset_test_spring.csv")}
    }
    return state


@pytest.fixture
def fake_preprocessing_state(fake_integration_state):
    state = dict(fake_integration_state)
    state["preprocessing"] = {
        "spring": {
            "train": Path("dataset_train_spring_clean.csv"),
            "test": Path("dataset_test_spring_clean.csv"),
            "full": Path("dataset_full_spring_clean.csv"),
        }
    }
    return state


def test_seasonal_stage_receives_ref_path(monkeypatch, fake_static_state):
    captured = {}

    def fake_stage(config, input_paths):
        captured.update(input_paths)
        return {}

    monkeypatch.setattr(rp, "stage_seasonal", fake_stage)
    rp.run_stage("seasonal", config={}, state=fake_static_state)

    assert captured["ref_path"] == Path("ref.tif")


def test_labels_stage_receives_ref_path(monkeypatch, fake_static_state):
    captured = {}

    def fake_stage(config, input_paths):
        captured.update(input_paths)
        return {}

    monkeypatch.setattr(rp, "stage_labels", fake_stage)
    rp.run_stage("labels", config={}, state=fake_static_state)

    assert captured["ref_path"] == Path("ref.tif")


def test_integration_stage_receives_static_seasonal_labels(monkeypatch, fake_seasonal_labels_state):
    captured = {}

    def fake_stage(config, input_paths):
        captured.update(input_paths)
        return {}

    monkeypatch.setattr(rp, "stage_integration", fake_stage)
    rp.run_stage("integration", config={}, state=fake_seasonal_labels_state)

    assert captured["ref_path"] == Path("ref.tif")
    assert captured["static"] == fake_seasonal_labels_state["static"]
    assert captured["seasonal"] == fake_seasonal_labels_state["seasonal"]
    assert captured["labels"] == fake_seasonal_labels_state["labels"]


def test_preprocessing_stage_receives_ref_path_and_integration(monkeypatch, fake_integration_state):
    captured = {}

    def fake_stage(config, input_paths):
        captured.update(input_paths)
        return {}

    monkeypatch.setattr(rp, "stage_preprocessing", fake_stage)
    rp.run_stage("preprocessing", config={}, state=fake_integration_state)

    assert captured["ref_path"] == Path("ref.tif")
    assert captured["spring"] == fake_integration_state["integration"]["spring"]


def test_eda_stage_receives_raw_and_clean(monkeypatch, fake_preprocessing_state):
    captured = {}

    def fake_stage(config, input_paths):
        captured.update(input_paths)
        return {}

    monkeypatch.setattr(rp, "stage_eda", fake_stage)
    rp.run_stage("eda", config={}, state=fake_preprocessing_state)

    assert captured["raw"] == fake_preprocessing_state["integration"]
    assert captured["clean"] == fake_preprocessing_state["preprocessing"]


def test_temporal_eda_stage_receives_config_and_stores_result(monkeypatch):
    captured = {}

    def fake_stage(config):
        captured["config"] = config
        return {"stationarity_summary": Path("stationarity.csv")}

    monkeypatch.setattr(rp, "stage_temporal_eda", fake_stage)
    fake_cfg = {"base": {"figures_dir": "figures"}}
    state = rp.run_stage("temporal_eda", config=fake_cfg, state={})

    assert captured["config"] is fake_cfg
    assert state["temporal_eda"] == {"stationarity_summary": Path("stationarity.csv")}


def test_train_stage_merges_preprocessing_paths_with_fire_gpkgs(monkeypatch, fake_preprocessing_state):
    captured = {}

    def fake_stage(config, input_paths):
        captured.update(input_paths)
        return {}

    monkeypatch.setattr(rp, "stage_train", fake_stage)
    rp.run_stage("train", config={}, state=fake_preprocessing_state)

    assert captured["ref_path"] == Path("ref.tif")
    assert captured["spring"]["train"] == Path("dataset_train_spring_clean.csv")
    assert captured["spring"]["test"] == Path("dataset_test_spring_clean.csv")
    assert captured["spring"]["fire_train"] == Path("fire_train.gpkg")
    assert captured["spring"]["fire_test"] == Path("fire_test.gpkg")


def test_evaluate_stage_wires_artifacts_from_train_state(monkeypatch, fake_preprocessing_state):
    state = dict(fake_preprocessing_state)
    state["train"] = {"spring": {"random_forest": Path("models/spring/random_forest/abc12345")}}

    captured = {}

    def fake_stage(config, input_paths):
        captured.update(input_paths)
        return {}

    monkeypatch.setattr(rp, "stage_evaluate", fake_stage)
    rp.run_stage("evaluate", config={}, state=state)

    assert captured["ref_path"] == Path("ref.tif")
    assert captured["spring"]["test"] == Path("dataset_test_spring_clean.csv")
    assert captured["spring"]["full"] == Path("dataset_full_spring_clean.csv")
    assert captured["spring"]["fire_test"] == Path("fire_test.gpkg")
    assert captured["spring"]["artifacts"] == {"random_forest": Path("models/spring/random_forest/abc12345")}


def test_selection_stage_receives_train_and_evaluate(monkeypatch, fake_preprocessing_state):
    state = dict(fake_preprocessing_state)
    state["train"] = {"spring": {"random_forest": Path("models/spring/random_forest/abc12345")}}
    state["evaluate"] = {"spring": {"random_forest": {"val_auc": 0.8}}}

    captured = {}

    def fake_stage(config, input_paths):
        captured.update(input_paths)
        return {}

    monkeypatch.setattr(rp, "stage_selection", fake_stage)
    rp.run_stage("selection", config={}, state=state)

    assert captured["train"] == state["train"]
    assert captured["evaluate"] == state["evaluate"]


def test_unknown_stage_raises():
    with pytest.raises(ValueError, match="Unknown stage"):
        rp.run_stage("not_a_real_stage", config={}, state={})