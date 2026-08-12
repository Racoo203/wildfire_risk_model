"""Shared contract test for every BaseWildfireModel implementation —
this is what makes the registry pattern safe to extend (Section 12)."""

import numpy as np
import pytest

from wildfire_susceptibility.core.registry import MODELS
from wildfire_susceptibility import modeling  # noqa: F401
from wildfire_susceptibility.modeling import models as _models  # noqa: F401 — registers wrappers


@pytest.fixture
def synthetic_classification_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 5)).astype("float32")
    y = rng.integers(0, 4, size=120)
    return X, y


@pytest.mark.parametrize(
    "model_name", ["random_forest", "svm", "xgboost", "catboost", "ordinal_lr"]
)
def test_model_contract(model_name, synthetic_classification_data):
    X, y = synthetic_classification_data
    model_cls = MODELS[model_name]
    model = model_cls()

    fitted = model.fit(X, y)
    assert fitted is model

    proba = model.predict_proba(X)
    assert proba.shape[0] == X.shape[0]
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-3)
    assert (proba >= 0).all() and (proba <= 1).all()

    assert isinstance(model.needs_scaling(), bool)


@pytest.mark.parametrize("model_name", ["catboost", "ordinal_lr"])
def test_new_model_proba_shape_matches_4class_metrics_contract(model_name, synthetic_classification_data):
    """metrics.py's compute_full_metrics (PR-AUC/F1-macro/QWK, and the
    Optuna HPO objective built on top of it) all index y_proba as an
    (n_samples, 4) array — confirm both new models produce exactly that
    shape, not just "some" 2D array."""
    X, y = synthetic_classification_data
    model = MODELS[model_name]().fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], 4)


def test_neural_net_contract(synthetic_classification_data):
    torch = pytest.importorskip("torch")
    X, y = synthetic_classification_data
    model_cls = MODELS["neural_net"]
    model = model_cls(epochs=3)  # keep the test fast

    fitted = model.fit(X, y)
    assert fitted is model

    proba = model.predict_proba(X)
    assert proba.shape[0] == X.shape[0]
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-2)


@pytest.mark.parametrize(
    "model_name", ["random_forest", "svm", "xgboost", "catboost", "ordinal_lr"]
)
def test_fit_accepts_sample_weight_and_it_actually_changes_the_fit(model_name, synthetic_classification_data):
    """Every wrapper's fit() must accept sample_weight (used by
    imbalance_strategy='cost_weighted', see modeling/imbalance.py) and
    actually use it, not just silently accept-and-ignore it — this is a
    behavioral check (different weighting -> different predict_proba), not
    just a signature/crash check."""
    X, y = synthetic_classification_data

    heavy_on_class_0 = np.where(y == 0, 100.0, 1.0)
    heavy_on_class_1 = np.where(y == 1, 100.0, 1.0)

    model_a = MODELS[model_name]().fit(X, y, sample_weight=heavy_on_class_0)
    model_b = MODELS[model_name]().fit(X, y, sample_weight=heavy_on_class_1)

    proba_a = model_a.predict_proba(X)
    proba_b = model_b.predict_proba(X)

    assert not np.allclose(proba_a, proba_b), (
        f"{model_name}: predict_proba was identical under two very different "
        f"sample_weight arrays — sample_weight is likely being silently dropped."
    )


def test_neural_net_fit_accepts_sample_weight_and_it_actually_changes_the_fit(synthetic_classification_data):
    torch = pytest.importorskip("torch")
    X, y = synthetic_classification_data

    heavy_on_class_0 = np.where(y == 0, 100.0, 1.0)
    heavy_on_class_1 = np.where(y == 1, 100.0, 1.0)

    torch.manual_seed(0)
    model_a = MODELS["neural_net"](epochs=5).fit(X, y, sample_weight=heavy_on_class_0)
    torch.manual_seed(0)
    model_b = MODELS["neural_net"](epochs=5).fit(X, y, sample_weight=heavy_on_class_1)

    proba_a = model_a.predict_proba(X)
    proba_b = model_b.predict_proba(X)
    assert not np.allclose(proba_a, proba_b), (
        "neural_net: predict_proba was identical under two very different "
        "sample_weight arrays — sample_weight is likely being silently dropped."
    )


def test_random_forest_cost_weighted_pins_class_weight_to_none(synthetic_classification_data):
    """RF's class_weight is an Optuna-searched hyperparameter (None/'balanced').
    When an external sample_weight is supplied (cost_weighted strategy),
    class_weight must be forced to None so sklearn doesn't multiply
    class_weight-derived weights by sample_weight and silently compound the
    two imbalance-correction mechanisms."""
    X, y = synthetic_classification_data
    model = MODELS["random_forest"](class_weight="balanced")
    weight = np.ones(len(y))

    model.fit(X, y, sample_weight=weight)

    assert model.model.class_weight is None


def test_param_space_returns_dict_optuna_can_consume(synthetic_classification_data):
    optuna = pytest.importorskip("optuna")
    X, y = synthetic_classification_data

    for name in ["random_forest", "svm", "xgboost", "catboost", "ordinal_lr"]:
        model = MODELS[name]()
        study = optuna.create_study()
        trial = study.ask()
        space = model.param_space(trial)
        assert isinstance(space, dict) and len(space) > 0