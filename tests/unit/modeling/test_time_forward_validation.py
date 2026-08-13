"""Regression coverage for _try_time_forward_validation under n_classes=5:
before class_labels.py existed, the function enumerated a hardcoded
4-name CLASS_NAMES list, so any fire point whose nearest pixel was
predicted into the 5th class (index 4) was silently dropped from
`counts`/`pct` -- `total` still counted it, so the reported percentages
quietly failed to sum to 100%. This is the paper's headline validation
figure (see the function's own docstring), so the drop is consequential."""

import numpy as np
import geopandas as gpd
import pytest
from shapely.geometry import Point

from wildfire_susceptibility.modeling.evaluate import _try_time_forward_validation


class _FixedProbaModel:
    """predict_proba ignores X and returns a fixed, per-row one-hot-ish
    probability matrix so each pixel's argmax class is known up front."""

    def __init__(self, proba: np.ndarray):
        self._proba = proba

    def predict_proba(self, X):
        return self._proba


@pytest.fixture
def five_class_fixture():
    n_classes = 5
    proba = np.eye(n_classes) * 0.9 + 0.1 / n_classes
    proba /= proba.sum(axis=1, keepdims=True)
    model = _FixedProbaModel(proba)

    x_coords = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    y_coords = np.zeros(5)
    X_test = np.zeros((5, 1))  # unused by _FixedProbaModel

    fire_test_gdf = gpd.GeoDataFrame(
        geometry=[Point(x, 0.0) for x in x_coords], crs="EPSG:27700"
    )
    return model, X_test, x_coords, y_coords, fire_test_gdf


def test_every_class_counted_including_highest(five_class_fixture):
    model, X_test, x_coords, y_coords, fire_test_gdf = five_class_fixture

    result = _try_time_forward_validation(
        model, X_test, x_coords, y_coords, fire_test_gdf,
        season="summer", model_name="catboost", config={},
    )

    assert result is not None
    assert result["n_fires"] == 5
    assert result["counts"]["Very High"] == 1


def test_reported_percentages_sum_to_total(five_class_fixture):
    model, X_test, x_coords, y_coords, fire_test_gdf = five_class_fixture

    result = _try_time_forward_validation(
        model, X_test, x_coords, y_coords, fire_test_gdf,
        season="summer", model_name="catboost", config={},
    )

    per_class_pct = {
        k: v for k, v in result.items()
        if k.startswith("pct_") and k != "pct_medium_plus"
    }
    assert sum(per_class_pct.values()) == pytest.approx(100.0)
