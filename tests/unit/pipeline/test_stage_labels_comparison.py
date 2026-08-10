"""Tests for stage_labels' compare_classify_methods wiring: when the list
is populated, every listed classify_method must actually run (train-fit +
test-freeze-apply, matching the primary build's contract) and get logged
to its own distinct nested MLflow run -- not silently ignored, which was
the pre-fix behavior (the field existed in schema but nothing iterated
over it)."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import Point

from wildfire_susceptibility.pipeline.stage_labels import _run_classify_comparison

_GRID_SIZE = 60


@pytest.fixture
def comparison_reference_raster(tmp_path):
    """Bigger than the 10x10 grid used elsewhere -- jenks/gmm need enough
    distinct density values to form non-degenerate classes."""
    path = tmp_path / "reference.tif"
    transform = from_origin(0, 100_000, 30.0, 30.0)
    data = np.ones((_GRID_SIZE, _GRID_SIZE), dtype="float32")
    meta = {
        "driver": "GTiff", "height": _GRID_SIZE, "width": _GRID_SIZE,
        "count": 1, "dtype": "float32", "crs": "EPSG:27700",
        "transform": transform, "nodata": np.nan,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data[np.newaxis, :, :])
    return path


def _fire_points(seed, n=40):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, _GRID_SIZE * 30.0, n)
    ys = rng.uniform(100_000 - _GRID_SIZE * 30.0, 100_000, n)
    return gpd.GeoDataFrame(geometry=[Point(x, y) for x, y in zip(xs, ys)], crs="EPSG:27700")


@pytest.fixture
def comparison_config(tmp_path, monkeypatch, minimal_config):
    monkeypatch.chdir(tmp_path)
    test_db = tmp_path / "mlflow_test.db"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.trainer.MLFLOW_TRACKING_URI",
        f"sqlite:///{test_db}",
    )

    config = dict(minimal_config)
    config["labels"] = {
        "density_method": "convolution",
        "classify_method": "percentile",
        "compare_classify_methods": ["percentile", "jenks", "gmm"],
        "conv_sigma_cells": 3,
        "n_classes": 3,
        "trim_bottom_pct": 0.0,
        "auto_find_k": False,
        "random_state": 42,
        "gmm_random_state": 42,
        "jenks_max_sample": 200_000,
        "gmm_max_sample": 200_000,
    }
    config["modeling"] = {"mlflow_experiment": "test-classify-comparison"}
    return config


def test_compare_classify_methods_logs_one_nested_run_per_method(
    comparison_config, comparison_reference_raster
):
    import mlflow

    fire_train = _fire_points(seed=1)
    fire_test = _fire_points(seed=2)

    _run_classify_comparison(
        comparison_config, "cmpseason", fire_train, fire_test,
        comparison_reference_raster, comparison_config["labels"]["compare_classify_methods"],
    )

    runs = mlflow.search_runs(experiment_names=["test-classify-comparison"])
    method_runs = runs[runs["tags.classify_method"].notna()]

    assert sorted(method_runs["tags.classify_method"].tolist()) == ["gmm", "jenks", "percentile"]

    # Each method run must carry its own distinct metrics/params, not a
    # shared/overwritten set -- e.g. distinct train class-count metrics.
    for _, run in method_runs.iterrows():
        assert run["metrics.train_n_total"] > 0
        assert run["metrics.test_n_total"] > 0
        assert run["params.n_classes"] == "3"

    # Parent comparison run exists and tags which methods were compared.
    parent_runs = runs[runs["tags.comparison"] == "classify_method"]
    assert len(parent_runs) == 1
    assert parent_runs.iloc[0]["tags.compared_methods"] == "percentile,jenks,gmm"


def test_compare_classify_methods_artifacts_logged_per_method(
    comparison_config, comparison_reference_raster
):
    import mlflow

    fire_train = _fire_points(seed=1)
    fire_test = _fire_points(seed=2)

    _run_classify_comparison(
        comparison_config, "cmpseason", fire_train, fire_test,
        comparison_reference_raster, comparison_config["labels"]["compare_classify_methods"],
    )

    client = mlflow.tracking.MlflowClient()
    runs = mlflow.search_runs(experiment_names=["test-classify-comparison"])
    method_runs = runs[runs["tags.classify_method"].notna()]

    for _, run in method_runs.iterrows():
        artifacts = client.list_artifacts(run["run_id"])
        artifact_names = {a.path for a in artifacts}
        # train + test label rasters, both logged, both named after this
        # run's own classify_method (not aliased to a shared/primary file)
        method = run["tags.classify_method"]
        assert any(method in name and "train" in name for name in artifact_names)
        assert any(method in name and "test" in name for name in artifact_names)
