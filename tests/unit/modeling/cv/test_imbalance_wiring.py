# tests/unit/modeling/cv/test_imbalance_wiring.py
"""Regression coverage for CVStrategy.fit_and_score_full's model_name-aware
imbalance handling (cv/base.py, modeling/imbalance.py): confirms
'cost_weighted' and 'smote' are genuinely different code paths reaching
model.fit() — not two config values that silently collapse to the same
behavior — and that sample_weight is computed from the real per-fold
training labels, not some placeholder."""

import numpy as np
import pandas as pd
import pytest

from sklearn.utils.class_weight import compute_sample_weight

from wildfire_susceptibility.modeling.cv.standard import StandardKFoldCV
from wildfire_susceptibility.modeling.resampling import SMOTEResampler


class _CapturingModel:
    """Records exactly what fit() was called with, so a test can assert on
    the real call rather than inferring it from predict_proba output."""

    last_fit_calls = []

    def __init__(self, **_params):
        pass

    def fit(self, X, y, sample_weight=None):
        _CapturingModel.last_fit_calls.append({"X": X, "y": y, "sample_weight": sample_weight})
        return self

    def predict_proba(self, X):
        # 4-class, uniform — good enough, this test doesn't assert on metrics.
        return np.full((len(X), 4), 0.25)


@pytest.fixture(autouse=True)
def _reset_capture():
    _CapturingModel.last_fit_calls = []
    yield
    _CapturingModel.last_fit_calls = []


def _dataset():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=["f1", "f2", "f3"])
    y = pd.Series([0, 1, 2, 3] * 10)
    train_idx = np.arange(30)
    test_idx = np.arange(30, 40)
    return X, y, train_idx, test_idx


def test_cost_weighted_model_gets_balanced_sample_weight_and_no_resampling(minimal_modeling_config):
    config = minimal_modeling_config
    config["modeling"]["use_smote"] = True  # would enable SMOTE under the legacy/default path
    config["modeling"]["imbalance_strategy_by_model"] = {"catboost": "cost_weighted"}
    resampler = SMOTEResampler(config)
    strategy = StandardKFoldCV(config, resampler)

    X, y, train_idx, test_idx = _dataset()
    strategy.fit_and_score_full(_CapturingModel, {}, X, y, train_idx, test_idx, model_name="catboost")

    call = _CapturingModel.last_fit_calls[0]
    assert call["sample_weight"] is not None
    np.testing.assert_allclose(call["sample_weight"], compute_sample_weight("balanced", call["y"]))
    # No resampling: fit sees exactly the untouched training fold.
    assert len(call["y"]) == len(train_idx)


def test_smote_model_gets_no_sample_weight(minimal_modeling_config):
    config = minimal_modeling_config
    config["modeling"]["use_smote"] = True
    # minimal_modeling_config dumps the schema's real default (None) for
    # smote_sampling_strategy explicitly, which imblearn's SMOTE rejects as
    # a sampling_strategy value — set the legacy "auto" string.
    config["modeling"]["smote_sampling_strategy"] = "auto"
    config["modeling"]["imbalance_strategy"] = "smote"
    resampler = SMOTEResampler(config)
    strategy = StandardKFoldCV(config, resampler)

    X, y, train_idx, test_idx = _dataset()
    strategy.fit_and_score_full(_CapturingModel, {}, X, y, train_idx, test_idx, model_name="random_forest")

    call = _CapturingModel.last_fit_calls[0]
    assert call["sample_weight"] is None


def test_none_strategy_model_gets_no_sample_weight_and_no_resampling(minimal_modeling_config):
    config = minimal_modeling_config
    config["modeling"]["use_smote"] = True  # must still be overridden off by imbalance_strategy='none'
    config["modeling"]["imbalance_strategy"] = "none"
    resampler = SMOTEResampler(config)
    strategy = StandardKFoldCV(config, resampler)

    X, y, train_idx, test_idx = _dataset()
    strategy.fit_and_score_full(_CapturingModel, {}, X, y, train_idx, test_idx, model_name="ordinal_lr")

    call = _CapturingModel.last_fit_calls[0]
    assert call["sample_weight"] is None
    assert len(call["y"]) == len(train_idx)  # untouched, not resampled
