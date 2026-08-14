"""Regression tests for modeling/evaluate.py's SHAP plotting: both the
TreeExplainer path (random_forest/catboost) and the KernelExplainer
fallback path used for ordinal_lr/neural_net. The fallback path had only
ever been exercised historically against old neural_net/xgboost rosters —
never against ordinal_lr specifically, and mord.LogisticAT has had
numpy-compat issues before (see modeling/models/ordinal_logistic.py's
np.int shim), so it needs its own coverage.

CatBoost and neural_net (MLP) were confirmed working against these same
paths during the post-merge reporting audit via a standalone smoke
script, but were never ported into the permanent suite — the two tests
below close that gap (RandomForest/ordinal_lr above already covered
each explainer path in the abstract; these confirm the two remaining
roster models concretely, since "same code path" isn't a substitute for
"actually exercised against this model's predict_proba output shape and
serialization details")."""

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from wildfire_susceptibility.modeling.evaluate import _try_shap_plots
from wildfire_susceptibility.modeling.models.ordinal_logistic import OrdinalLogisticModel
from wildfire_susceptibility.modeling.models.catboost_model import CatBoostModel
from wildfire_susceptibility.modeling.models.ann import NeuralNetModel


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
def test_tree_explainer_path_works_for_catboost(synthetic_tabular_4class, shap_config):
    """catboost is the other model routed through the TreeExplainer branch
    (`model_name in ("random_forest", "xgboost", "catboost")`), and is now
    the finalized-roster tree model replacing xgboost — confirmed working
    empirically during the reporting audit against a real trained model,
    ported here as permanent coverage."""
    X, y, feature_names = synthetic_tabular_4class
    model = CatBoostModel(iterations=20).fit(X, y)

    result = _try_shap_plots(model, X, feature_names, "summer", "catboost", shap_config)

    assert result is not None
    assert result["shap_summary_path"].exists()
    assert result["shap_dependence_path"].exists()
    assert {"shap_summary", "shap_dependence"} <= _manifest_categories(shap_config)


@pytest.mark.slow
def test_kernel_explainer_path_works_for_neural_net(synthetic_tabular_4class, shap_config):
    """neural_net (the PyTorch MLP) goes through the same KernelExplainer
    fallback as ordinal_lr, via the shared predict_proba contract — but
    had never been exercised against a torch model specifically, where
    tensor conversion / eval-mode handling could plausibly break
    KernelExplainer's repeated background-perturbation calls."""
    X, y, feature_names = synthetic_tabular_4class
    model = NeuralNetModel(epochs=5, hidden_dim=16, n_layers=1).fit(X, y)

    result = _try_shap_plots(model, X, feature_names, "summer", "neural_net", shap_config)

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
