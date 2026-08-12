"""Regression tests for modeling/evaluate.py's SHAP plotting: both the
TreeExplainer path (random_forest/catboost) and the KernelExplainer
fallback path used for ordinal_lr/neural_net. The fallback path had only
ever been exercised historically against old neural_net/xgboost rosters —
never against ordinal_lr specifically, and mord.LogisticAT has had
numpy-compat issues before (see modeling/models/ordinal_logistic.py's
np.int shim), so it needs its own coverage."""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from wildfire_susceptibility.modeling.evaluate import _try_shap_plots
from wildfire_susceptibility.modeling.models.ordinal_logistic import OrdinalLogisticModel


@pytest.fixture
def synthetic_tabular_4class():
    rng = np.random.default_rng(0)
    n = 60
    X = rng.normal(size=(n, 3))
    score = X[:, 0] + 0.5 * X[:, 1]
    y = np.digitize(score, np.quantile(score, [0.25, 0.5, 0.75]))
    return X, y, ["elevation", "slope", "ndvi"]


@pytest.fixture
def shap_config(tmp_path):
    return {"base": {"figures_dir": str(tmp_path / "figures")}}


def _manifest_categories(shap_config) -> set:
    manifest_path = Path(shap_config["base"]["figures_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    return {entry["category"] for entry in manifest}


@pytest.mark.slow
def test_tree_explainer_path_writes_summary_and_dependence_plots(synthetic_tabular_4class, shap_config):
    X, y, feature_names = synthetic_tabular_4class
    model = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)

    result = _try_shap_plots(model, X, feature_names, "summer", "random_forest", shap_config)

    assert result is not None
    assert result["shap_summary_path"].exists()
    assert result["shap_dependence_path"].exists()
    assert {"shap_summary", "shap_dependence"} <= _manifest_categories(shap_config)


@pytest.mark.slow
def test_kernel_explainer_path_works_for_ordinal_lr(synthetic_tabular_4class, shap_config):
    """Confirms the KernelExplainer fallback (used for every non-tree model,
    via the shared predict_proba contract) actually works against
    ordinal_lr — previously untested against this specific model."""
    X, y, feature_names = synthetic_tabular_4class
    model = OrdinalLogisticModel().fit(X, y)

    result = _try_shap_plots(model, X, feature_names, "summer", "ordinal_lr", shap_config)

    assert result is not None
    assert result["shap_summary_path"].exists()
    assert result["shap_dependence_path"].exists()
    assert {"shap_summary", "shap_dependence"} <= _manifest_categories(shap_config)


@pytest.mark.slow
def test_shap_plots_failure_is_caught_and_returns_none(shap_config):
    """A model with no predict_proba at all should fail soft (matching the
    stage's documented fail-soft contract: a partial evaluation beats
    losing the whole training run), not raise."""
    class BrokenModel:
        pass

    result = _try_shap_plots(BrokenModel(), np.zeros((5, 3)), ["a", "b", "c"], "summer", "broken", shap_config)

    assert result is None
