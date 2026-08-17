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


def test_neural_net_fit_is_deterministic_without_external_seeding(synthetic_classification_data):
    """Every other model wrapper fixes random_state=42; neural_net had no
    equivalent, so weight init and the DataLoader's shuffle order varied
    run-to-run purely from torch's global RNG state — meaning HPO search
    and the final refit for this model alone weren't reproducible, unlike
    every other model in the roster. fit() must seed internally now, not
    rely on a caller to torch.manual_seed() first."""
    torch = pytest.importorskip("torch")
    X, y = synthetic_classification_data

    model_a = MODELS["neural_net"](epochs=5).fit(X, y)
    model_b = MODELS["neural_net"](epochs=5).fit(X, y)

    proba_a = model_a.predict_proba(X)
    proba_b = model_b.predict_proba(X)
    np.testing.assert_allclose(proba_a, proba_b)


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


def test_random_forest_param_space_excludes_class_weight(synthetic_classification_data):
    """class_weight must never be an Optuna-tunable dimension: imbalance
    handling is a resolver-level config choice (modeling.imbalance_strategy),
    not something a trial should be free to pick. Before this test existed,
    a trial could sample class_weight="balanced" while imbalance_strategy
    resolved to "smote" for random_forest, stacking class weighting on top
    of SMOTE-resampled training data — the fit()-level guard in
    RandomForestModel.fit() only neutralizes the cost_weighted case (see
    test_random_forest_cost_weighted_pins_class_weight_to_none above), not
    this one."""
    optuna = pytest.importorskip("optuna")
    study = optuna.create_study()
    trial = study.ask()

    space = MODELS["random_forest"]().param_space(trial)

    assert "class_weight" not in space


def test_ordinal_lr_fits_float_labels(synthetic_classification_data):
    """Real pipeline labels are float (raster stacking in core/raster.py's
    stack_to_dataframe casts every column, including labels, to float32),
    not the int64 labels synthetic_classification_data otherwise supplies.
    mord.LogisticAT derives n_class_ from y's own dtype rather than casting
    it, so a float64 y previously made n_class_ a numpy.float64 and crashed
    mord's internal np.zeros((n_class_ - 1, ...)) allocation with
    TypeError: 'numpy.float64' object cannot be interpreted as an integer."""
    X, y = synthetic_classification_data
    y_float = y.astype("float64")

    model = MODELS["ordinal_lr"]().fit(X, y_float)

    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], 4)


def test_param_space_returns_dict_optuna_can_consume(synthetic_classification_data):
    optuna = pytest.importorskip("optuna")
    X, y = synthetic_classification_data

    for name in ["random_forest", "svm", "xgboost", "catboost", "ordinal_lr"]:
        model = MODELS[name]()
        study = optuna.create_study()
        trial = study.ask()
        space = model.param_space(trial)
        assert isinstance(space, dict) and len(space) > 0


def test_random_forest_search_space_bounds_tightened_against_overfitting():
    """Deployed RF models were consistently landing at/near the old
    max_depth=25 ceiling and min_samples_leaf=1 floor (depth=23, leaf=2 in
    one case), with standard-CV AUC ~0.99 collapsing to ~0.5-0.55 on spatial
    CV and true validation. Pins the tightened range so a future edit can't
    silently widen it back toward that overfitting regime."""
    optuna = pytest.importorskip("optuna")
    study = optuna.create_study()
    trial = study.ask()

    space = MODELS["random_forest"]().param_space(trial)

    assert "max_samples" in space, "row-subsampling knob must stay in the search space"
    assert (trial.distributions["max_depth"].low, trial.distributions["max_depth"].high) == (3, 12)
    assert (trial.distributions["min_samples_leaf"].low, trial.distributions["min_samples_leaf"].high) == (5, 30)
    assert (trial.distributions["min_samples_split"].low, trial.distributions["min_samples_split"].high) == (10, 40)
    assert (trial.distributions["max_samples"].low, trial.distributions["max_samples"].high) == (0.3, 0.8)


def test_random_forest_fits_with_max_samples(synthetic_classification_data):
    """max_samples is a new param_space() key that flows straight into
    RandomForestClassifier(**best_params) at final-refit time — confirm the
    kwarg name is one sklearn actually accepts, not just that param_space()
    returns a plausible-looking dict."""
    X, y = synthetic_classification_data
    model = MODELS["random_forest"](
        n_estimators=50, max_depth=5, min_samples_leaf=5,
        min_samples_split=10, max_features="sqrt", max_samples=0.5,
    ).fit(X, y)

    assert model.model.max_samples == 0.5


def test_catboost_search_space_bounds_tightened_against_overfitting():
    """CatBoost's optimism gap was more moderate than RF's, but the old
    l2_leaf_reg floor (1e-2) allowed near-zero-regularization trials and the
    space had no min_data_in_leaf/random_strength — CatBoost's two standard
    anti-overfitting knobs beyond depth/l2_leaf_reg."""
    optuna = pytest.importorskip("optuna")
    study = optuna.create_study()
    trial = study.ask()

    space = MODELS["catboost"]().param_space(trial)

    assert "min_data_in_leaf" in space
    assert "random_strength" in space
    assert (trial.distributions["depth"].low, trial.distributions["depth"].high) == (3, 8)
    assert (trial.distributions["l2_leaf_reg"].low, trial.distributions["l2_leaf_reg"].high) == (1.0, 10.0)
    assert (trial.distributions["min_data_in_leaf"].low, trial.distributions["min_data_in_leaf"].high) == (5, 50)
    assert (trial.distributions["random_strength"].low, trial.distributions["random_strength"].high) == (0.0, 10.0)


def test_catboost_fits_with_new_regularization_params(synthetic_classification_data):
    """min_data_in_leaf/random_strength are new param_space() keys that flow
    straight into CatBoostClassifier(**best_params) at final-refit time —
    confirm the kwarg names are ones CatBoost actually accepts."""
    X, y = synthetic_classification_data
    model = MODELS["catboost"](
        iterations=50, depth=4, learning_rate=0.1, l2_leaf_reg=2.0,
        bagging_temperature=0.5, min_data_in_leaf=5, random_strength=1.0,
    ).fit(X, y)

    proba = model.predict_proba(X)
    assert proba.shape[0] == X.shape[0]