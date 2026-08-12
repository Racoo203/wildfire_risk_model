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


def test_param_space_returns_dict_optuna_can_consume(synthetic_classification_data):
    optuna = pytest.importorskip("optuna")
    X, y = synthetic_classification_data

    for name in ["random_forest", "svm", "xgboost", "catboost", "ordinal_lr"]:
        model = MODELS[name]()
        study = optuna.create_study()
        trial = study.ask()
        space = model.param_space(trial)
        assert isinstance(space, dict) and len(space) > 0