"""Regression tests for stage_labels always computing d_fires, regardless of
the now-removed `include_d_fires_as_feature` flag. That flag used to gate
whether FireProximityBuilder ran at all, meaning d_fires never existed as a
column anywhere on disk under the reported (false) configs -- which blocked
any diagnostic wanting to look at it. d_fires is now unconditionally
computed in `_build_season`; its exclusion from the trained feature set is
expressed via the existing generic `modeling.excluded_features` mechanism
(see stage_train.py / stage_evaluate.py) instead."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import Point

from wildfire_susceptibility.labels.fire_incidents import FireBuilder
from wildfire_susceptibility.pipeline.stage_labels import _build_season

_GRID_SIZE = 60


@pytest.fixture
def d_fires_reference_raster(tmp_path):
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


def _labels_config(minimal_config, extra_labels=None):
    config = dict(minimal_config)
    config["labels"] = {
        "density_method": "convolution",
        "classify_method": "percentile",
        "conv_sigma_cells": 3,
        "n_classes": 3,
        "trim_bottom_pct": 0.0,
        "auto_find_k": False,
        "random_state": 42,
        **(extra_labels or {}),
    }
    return config


@pytest.mark.parametrize("extra_labels", [
    pytest.param({}, id="no-leftover-key"),
    pytest.param({"include_d_fires_as_feature": False}, id="stale-key-ignored"),
])
def test_build_season_always_computes_d_fires(
    minimal_config, d_fires_reference_raster, monkeypatch, extra_labels
):
    fire_train = _fire_points(seed=1)
    fire_test = _fire_points(seed=2)
    monkeypatch.setattr(
        FireBuilder, "process",
        lambda self, months, season=None: (fire_train, fire_test),
    )

    config = _labels_config(minimal_config, extra_labels)

    out = _build_season(config, "cmpseason", (1, 2, 3), d_fires_reference_raster)

    assert "d_fires" in out["train"]
    assert "d_fires" in out["test"]
    assert out["train"]["d_fires"].exists()
    # Built once from fire_train, shared unmodified across both splits.
    assert out["train"]["d_fires"] == out["test"]["d_fires"]
