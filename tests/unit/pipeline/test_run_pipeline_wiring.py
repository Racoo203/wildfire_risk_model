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
def fake_preprocess_state():
    return {"preprocess": {"ref_path": Path("ref.tif"), "boundary": Path("boundary.shp")}}


@pytest.fixture
def fake_features_labels_state(fake_preprocess_state):
    state = dict(fake_preprocess_state)
    state["features_labels"] = {
        "spring": {
            "train": {"tas": Path("tas_train.tif"), "raw_labels": Path("labels_train.tif")},
            "test": {"tas": Path("tas_test.tif"), "raw_labels": Path("labels_test.tif")},
            "fire_train": Path("fire_train.gpkg"),
            "fire_test": Path("fire_test.gpkg"),
        }
    }
    return state


@pytest.fixture
def fake_dataset_assembly_state(fake_features_labels_state):
    state = dict(fake_features_labels_state)
    state["dataset_assembly"] = {
        "spring": {"train": Path("dataset_train_spring.csv"), "test": Path("dataset_test_spring.csv")}
    }
    return state


def test_features_labels_stage_receives_ref_path(monkeypatch, fake_preprocess_state):
    captured = {}

    def fake_stage(config, input_paths):
        captured.update(input_paths)
        return {}

    monkeypatch.setattr(rp, "stage_features_labels", fake_stage)
    rp.run_stage("features_labels", config={}, state=fake_preprocess_state)

    assert captured["ref_path"] == Path("ref.tif")


def test_dataset_assembly_stage_receives_static_and_seasonal(monkeypatch, fake_features_labels_state):
    captured = {}

    def fake_stage(config, input_paths):
        captured.update(input_paths)
        return {}

    monkeypatch.setattr(rp, "stage_dataset_assembly", fake_stage)
    rp.run_stage("dataset_assembly", config={}, state=fake_features_labels_state)

    assert captured["ref_path"] == Path("ref.tif")
    assert captured["static"] == fake_features_labels_state["preprocess"]
    assert captured["seasonal"] == fake_features_labels_state["features_labels"]


def test_train_stage_merges_dataset_paths_with_fire_gpkgs(monkeypatch, fake_dataset_assembly_state):
    captured = {}

    def fake_stage(config, input_paths):
        captured.update(input_paths)
        return {}

    monkeypatch.setattr(rp, "stage_train", fake_stage)
    rp.run_stage("train", config={}, state=fake_dataset_assembly_state)

    assert captured["ref_path"] == Path("ref.tif")
    assert captured["spring"]["train"] == Path("dataset_train_spring.csv")
    assert captured["spring"]["test"] == Path("dataset_test_spring.csv")
    assert captured["spring"]["fire_train"] == Path("fire_train.gpkg")
    assert captured["spring"]["fire_test"] == Path("fire_test.gpkg")


def test_evaluate_stage_wires_artifacts_from_train_state(monkeypatch, fake_dataset_assembly_state):
    state = dict(fake_dataset_assembly_state)
    state["train"] = {"spring": {"random_forest": Path("models/spring/random_forest/abc12345")}}

    captured = {}

    def fake_stage(config, input_paths):
        captured.update(input_paths)
        return {}

    monkeypatch.setattr(rp, "stage_evaluate", fake_stage)
    rp.run_stage("evaluate", config={}, state=state)

    assert captured["ref_path"] == Path("ref.tif")
    assert captured["spring"]["test"] == Path("dataset_test_spring.csv")
    assert captured["spring"]["fire_test"] == Path("fire_test.gpkg")
    assert captured["spring"]["artifacts"] == {"random_forest": Path("models/spring/random_forest/abc12345")}


def test_unknown_stage_raises():
    with pytest.raises(ValueError, match="Unknown stage"):
        rp.run_stage("not_a_real_stage", config={}, state={})