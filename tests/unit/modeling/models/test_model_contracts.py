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


def test_random_forest_native_balanced_uses_balanced_random_forest_classifier(synthetic_classification_data):
    """imbalance_strategy='native_balanced' must switch the underlying
    estimator to imbalanced-learn's BalancedRandomForestClassifier (each
    tree bootstraps a class-balanced sample), not plain
    RandomForestClassifier with a differently-set knob — sklearn's RF
    sample_weight only reweights the split criterion, not which rows get
    bootstrapped, so it structurally cannot reproduce this behavior (see
    scripts/experiment_imbalance_native_vs_costweighted.py: production RF
    predicted "Very High" for 3 of 858,106 holdout pixels under
    cost_weighted vs. ~35,500 of 3.85M full-domain pixels under this)."""
    from imblearn.ensemble import BalancedRandomForestClassifier

    X, y = synthetic_classification_data
    model = MODELS["random_forest"](native_balanced=True, n_estimators=10, max_depth=3).fit(X, y)

    assert isinstance(model.model, BalancedRandomForestClassifier)
    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], 4)


def test_random_forest_native_balanced_false_keeps_plain_random_forest(synthetic_classification_data):
    """native_balanced is bound onto model_cls via functools.partial in
    trainer.py (same mechanism as cat_features) — it is never a real
    sklearn/imblearn constructor kwarg. Confirms the default/False path is
    completely unaffected (still plain RandomForestClassifier) and that
    popping the kwarg doesn't itself break anything."""
    from sklearn.ensemble import RandomForestClassifier

    X, y = synthetic_classification_data
    model = MODELS["random_forest"](native_balanced=False, n_estimators=10, max_depth=3).fit(X, y)

    assert isinstance(model.model, RandomForestClassifier)


def test_catboost_native_balanced_sets_auto_class_weights(synthetic_classification_data):
    """imbalance_strategy='native_balanced' must set CatBoost's own
    auto_class_weights='Balanced' — the variant validated (scripts/
    experiment_catboost_weighting_variants.py) to engage the rare classes
    more than the external cost_weighted sample_weight (High recall
    0.170->0.195, Very High recall 0.001->0.020) without the collapse
    dampened (SqrtBalanced) or manually over-weighted variants showed."""
    X, y = synthetic_classification_data
    model = MODELS["catboost"](native_balanced=True, iterations=10, depth=3).fit(X, y)

    assert model.model.get_params().get("auto_class_weights") == "Balanced"


def test_catboost_native_balanced_false_leaves_auto_class_weights_unset(synthetic_classification_data):
    """Same rationale as the random_forest version above — native_balanced
    is a trainer.py-level binding, not a real CatBoostClassifier kwarg;
    confirms the default/False path doesn't set auto_class_weights at all."""
    X, y = synthetic_classification_data
    model = MODELS["catboost"](native_balanced=False, iterations=10, depth=3).fit(X, y)

    assert model.model.get_params().get("auto_class_weights") is None


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


def test_random_forest_search_space_widened_for_deeper_learning():
    """Supersedes the old test_random_forest_search_space_bounds_tightened_
    against_overfitting pin (08/16/2026: max_depth capped at 12,
    min_samples_leaf/min_samples_split floors raised to 5/10, after
    deployed RF models were landing at/near the old max_depth=25 ceiling
    and min_samples_leaf=1 floor with standard-CV AUC ~0.99 collapsing to
    ~0.5-0.55 on spatial CV/true validation).

    Re-widened 08/17/2026 (max_depth 12->20, min_samples_leaf floor 5->2,
    min_samples_split floor 10->5): that tightening was diagnosed under
    labels.classify_method="jenks"/n_classes=4, where the rarest class
    was ~0.1-0.2% of the data. Under classify_method="gmm"/n_classes=3
    (configs/experiment/baseline.yaml), the rarest class is ~9.8% of
    training data, and the standard-vs-spatial optimism gap that
    motivated the original tightening independently collapsed from
    ~0.35 to ~0.02-0.03 PR-AUC-macro at the OLD tight bounds — evidence
    the specific overfitting exploit (a deep tree carving one leaf around
    a handful of memorized rare-class points) may no longer be reachable
    under the new label balance. This widening is the deliberate,
    user-requested empirical test of that hypothesis, not an assumption
    it's true — a rerun's optimism gap staying low at the new bounds
    would support it; the gap reopening would refute it.
    max_features/max_samples deliberately left untouched, to isolate the
    depth/leaf effect from other regularization knobs.
    Pins the NEW range so a future edit can't silently drift it (in
    either direction) without an equally deliberate decision."""
    optuna = pytest.importorskip("optuna")
    study = optuna.create_study()
    trial = study.ask()

    space = MODELS["random_forest"]().param_space(trial)

    assert "max_samples" in space, "row-subsampling knob must stay in the search space"
    assert (trial.distributions["max_depth"].low, trial.distributions["max_depth"].high) == (3, 20)
    assert (trial.distributions["min_samples_leaf"].low, trial.distributions["min_samples_leaf"].high) == (2, 30)
    assert (trial.distributions["min_samples_split"].low, trial.distributions["min_samples_split"].high) == (5, 40)
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


def test_neural_net_search_space_bounds_tightened_against_overfitting():
    """neural_net was untouched by the RF/CatBoost overfitting-tightening
    pass despite having the most capacity of the four models (up to 4
    layers x 256 units) and a regularization floor that let trials disable
    both dropout (0.0) and weight_decay (1e-6) at once. Pins the tightened
    range so a future edit can't silently widen it back."""
    optuna = pytest.importorskip("optuna")
    study = optuna.create_study()
    trial = study.ask()

    space = MODELS["neural_net"]().param_space(trial)

    assert set(trial.distributions["hidden_dim"].choices) == {32, 64, 128}
    assert (trial.distributions["n_layers"].low, trial.distributions["n_layers"].high) == (1, 3)
    assert (trial.distributions["dropout"].low, trial.distributions["dropout"].high) == (0.1, 0.5)
    assert (trial.distributions["weight_decay"].low, trial.distributions["weight_decay"].high) == (1e-4, 1e-2)


def test_neural_net_early_stopping_halts_before_epoch_ceiling(synthetic_classification_data):
    """fit() previously had no validation split and no stopping mechanism —
    it always trained for exactly `epochs` regardless of overfitting. With
    lr=0 the model can't improve after its first epoch, so a non-trivial
    early_stopping_patience should trigger well before the (deliberately
    high) epoch ceiling is reached."""
    torch = pytest.importorskip("torch")
    X, y = synthetic_classification_data

    model = MODELS["neural_net"](
        epochs=50, lr=0.0, early_stopping_patience=2,
    ).fit(X, y)

    assert model.n_epochs_trained < 50, (
        "neural_net trained for the full epoch ceiling — early stopping "
        "did not trigger even with a stalled loss"
    )


def test_neural_net_early_stopping_restores_best_weights(synthetic_classification_data):
    """Confirm fit() reloads the best-val-loss checkpoint rather than
    whatever weights happen to be live when training stops — the whole
    point of early stopping is to not keep the post-overfitting weights."""
    torch = pytest.importorskip("torch")
    X, y = synthetic_classification_data

    model = MODELS["neural_net"](epochs=10, early_stopping_patience=3).fit(X, y)

    restored = {k: v.clone() for k, v in model.model.state_dict().items()}
    proba_a = model.predict_proba(X)
    proba_b = model.predict_proba(X)
    np.testing.assert_allclose(proba_a, proba_b)
    for k, v in model.model.state_dict().items():
        assert torch.equal(v, restored[k])