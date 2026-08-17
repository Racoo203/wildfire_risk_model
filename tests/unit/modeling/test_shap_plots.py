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
def synthetic_tabular_4class_with_categorical():
    """Same shape as synthetic_tabular_4class, plus a native-categorical
    landuse_class column at the last position — object-dtype array with
    real int codes there (mirrors modeling.categorical.to_model_array's
    output), not a plain float column. This is what actually reproduces
    the CatBoost/SHAP crash below: a model with zero categorical splits
    (the old fixture) never exercises the tree-parsing code path that
    crashes."""
    rng = np.random.default_rng(0)
    n = 60
    numeric = rng.normal(size=(n, 3))
    score = numeric[:, 0] + 0.5 * numeric[:, 1]
    y = np.digitize(score, np.quantile(score, [0.25, 0.5, 0.75]))
    X = np.empty((n, 4), dtype=object)
    X[:, :3] = numeric
    X[:, 3] = rng.integers(0, 4, size=n)
    return X, y, ["elevation", "slope", "ndvi", "landuse_class"]


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
def test_native_shap_path_works_for_catboost_with_categorical_feature(
    synthetic_tabular_4class_with_categorical, shap_config
):
    """catboost is routed through its own native SHAP computation
    (`model.get_feature_importance(type="ShapValues")` via a catboost.Pool),
    not shap.TreeExplainer — shap.TreeExplainer(catboost_model) segfaults
    (access violation inside shap's _cext extension, confirmed against
    shap==0.52.0/catboost==1.2.10) as soon as it's constructed against a
    CatBoost model carrying a native categorical feature, because its
    tree-parsing internals don't correctly represent CatBoost's
    oblivious/CTR-based categorical splits.

    This must use a fixture with a real cat_features-bound categorical
    column: the previous version of this test trained CatBoost on purely
    numeric features with no cat_features at all, so it never exercised
    the code path that actually crashes — it passed even against the
    crashing shap.TreeExplainer code, which is why the bug shipped
    unnoticed."""
    X, y, feature_names = synthetic_tabular_4class_with_categorical
    model = CatBoostModel(iterations=20, cat_features=[3]).fit(X, y)

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
