"""Regression coverage: PostTrainingEvaluator must generate the
susceptibility raster from the full in-domain feature set (X_full/
full_x_coords/full_y_coords) when supplied, not the label-filtered
X_val/x_coords/y_coords used for metrics computation -- so pixels
without a ground-truth density label still get a predicted risk class
on the output map, instead of being left as NaN gaps."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio

from wildfire_susceptibility.modeling.training.evaluation import PostTrainingEvaluator


class _FakeModel:
    n_classes = 4

    def predict_proba(self, X):
        n = len(X)
        proba = np.zeros((n, self.n_classes))
        proba[np.arange(n), np.arange(n) % self.n_classes] = 1.0
        return proba


def _n_valid_pixels(path):
    with rasterio.open(path) as src:
        data = src.read(1)
    return int(np.sum(~np.isnan(data)))


@pytest.fixture
def evaluator(minimal_modeling_config):
    Path(minimal_modeling_config["base"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    return PostTrainingEvaluator(minimal_modeling_config)


def _labeled_subset(reference_transform, n=8):
    """A small label-filtered subset -- what the test split looked like
    before this fix: coordinates only for pixels with a ground-truth label."""
    rows, cols = np.arange(n), np.arange(n)
    xs, ys = rasterio.transform.xy(reference_transform, rows, cols)
    X = pd.DataFrame({"f1": np.zeros(n), "f2": np.zeros(n)})
    y = pd.Series(np.arange(n) % 4)
    return X, y, np.array(xs), np.array(ys)


def _full_domain(synthetic_reference_raster):
    with rasterio.open(synthetic_reference_raster) as ref:
        data = ref.read(1)
        transform = ref.transform
    valid_rows, valid_cols = np.where(~np.isnan(data))
    xs, ys = rasterio.transform.xy(transform, valid_rows, valid_cols)
    # A DataFrame, matching X_val's contract — PostTrainingEvaluator.evaluate()
    # expects X_full pre-encoded but not yet converted to a model array (that
    # conversion, via to_model_array, happens inside evaluate() itself).
    X_full = pd.DataFrame({"f1": np.zeros(len(xs)), "f2": np.zeros(len(xs))})
    return X_full, np.array(xs), np.array(ys), int((~np.isnan(data)).sum())


def test_raster_falls_back_to_labeled_subset_without_full_domain_args(
    evaluator, synthetic_reference_raster, reference_transform
):
    """Prior behavior, preserved: if the caller doesn't supply X_full, the
    raster is scoped to the label-filtered X_val/x_coords/y_coords."""
    X_val, y_val, x_coords, y_coords = _labeled_subset(reference_transform)

    result = evaluator.evaluate(
        _FakeModel(), X_val, y_val, "summer", "fake_model",
        synthetic_reference_raster, None, x_coords, y_coords,
    )

    assert result["susceptibility_map_path"] is not None
    assert _n_valid_pixels(result["susceptibility_map_path"]) == len(x_coords)


def test_raster_covers_full_domain_when_full_domain_args_supplied(
    evaluator, synthetic_reference_raster, reference_transform
):
    """The fix: when X_full/full_x_coords/full_y_coords are supplied, the
    raster must cover every in-domain pixel, not just the smaller
    label-filtered subset used for metrics."""
    X_val, y_val, x_coords, y_coords = _labeled_subset(reference_transform)
    X_full, full_x, full_y, n_domain_pixels = _full_domain(synthetic_reference_raster)

    assert n_domain_pixels > len(x_coords), "test fixture must exercise full > labeled-subset coverage"

    result = evaluator.evaluate(
        _FakeModel(), X_val, y_val, "summer", "fake_model",
        synthetic_reference_raster, None, x_coords, y_coords,
        X_full=X_full, full_x_coords=full_x, full_y_coords=full_y,
    )

    assert result["susceptibility_map_path"] is not None
    assert _n_valid_pixels(result["susceptibility_map_path"]) == n_domain_pixels
