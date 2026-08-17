# tests/unit/modeling/training/test_search_objective.py
"""Coverage for HyperparamSearch._make_objective (search.py): the Optuna
objective shared by every model (RF/XGBoost/SVM/neural_net). Locks down two
things that motivated this change:

1. The value Optuna actually optimizes/prunes on is PR-AUC-macro, not AUC,
   F1-macro, or QWK — AUC tolerates majority-class collapse in a way
   PR-AUC/F1-macro don't (the original AUC~0.65-0.73 vs F1~0.32-0.38 symptom
   that motivated switching the objective), so it matters that the tuned
   value is really PR-AUC and not silently still AUC.
2. F1-macro, AUC, and QWK are computed every trial (via fit_and_score_full,
   not the auc-only scalar fit_and_score) and stashed as trial user_attrs
   for reporting/model comparison, without influencing the tuned value.

Also locks down that get_or_create_study's study name changes when
OBJECTIVE_VERSION changes, mirroring test_param_space_versioning.py's
coverage for param_space-shape changes — the study name's param-space hash
alone doesn't capture objective-function changes, so a bare param_space
hash bump would silently mix old AUC-scored trials with new PR-AUC-scored
ones in the same Optuna study."""

import logging

import numpy as np
import optuna
import pytest


class _StubCVStrategy:
    """Returns the same fixed, all-distinct metric dict for every fold, so
    a test can tell exactly which value ends up as the tuned objective vs.
    which end up as user_attrs just by their distinct values."""

    def __init__(self, scores: dict):
        self.scores = scores
        self.calls = 0

    def fit_and_score_full(self, model_cls, params, X, y, train_idx, test_idx, context="", model_name=None):
        self.calls += 1
        return dict(self.scores)


def _dummy_model_cls():
    class _Dummy:
        def param_space(self, trial):
            return {"x": trial.suggest_float("x", 0.0, 1.0)}
    return _Dummy


def _make_search_with_stub(scores: dict):
    from wildfire_susceptibility.modeling.training.search import HyperparamSearch

    search = HyperparamSearch.__new__(HyperparamSearch)  # skip __init__: no config/db needed
    stub_strategy = _StubCVStrategy(scores)
    search.cv_strategy = stub_strategy
    return search, stub_strategy


class _StubCVStrategySequence:
    """Like _StubCVStrategy, but returns a different fixed dict per call in
    sequence (cycling if there are more calls than entries) — needed to
    exercise per-fold NaN aggregation, which a single constant dict per
    call can't represent."""

    def __init__(self, scores_by_call: list):
        self.scores_by_call = scores_by_call
        self.calls = 0

    def fit_and_score_full(self, model_cls, params, X, y, train_idx, test_idx, context="", model_name=None):
        scores = self.scores_by_call[self.calls % len(self.scores_by_call)]
        self.calls += 1
        return dict(scores)


def _make_search_with_sequence_stub(scores_by_call: list):
    from wildfire_susceptibility.modeling.training.search import HyperparamSearch

    search = HyperparamSearch.__new__(HyperparamSearch)
    stub_strategy = _StubCVStrategySequence(scores_by_call)
    search.cv_strategy = stub_strategy
    return search, stub_strategy


def test_objective_value_is_pr_auc_macro_not_auc_f1_or_qwk():
    fixed_scores = {"auc": 0.5, "f1_macro": 0.3, "pr_auc_macro": 0.7, "qwk": 0.2}
    search, stub_strategy = _make_search_with_stub(fixed_scores)

    folds = [(np.array([0, 1]), np.array([2, 3])), (np.array([4, 5]), np.array([6, 7]))]
    objective = search._make_objective(
        _dummy_model_cls(), X_search=None, y_search=None, folds=folds, season="s", model_name="m",
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    assert study.best_value == pytest.approx(0.7), (
        "Optuna's tuned value must be PR-AUC-macro (0.7 in this fixture) — "
        f"got {study.best_value}, which matches a different metric in the fixture "
        "if this assertion fails, meaning the objective is scoring the wrong metric."
    )
    assert stub_strategy.calls == len(folds), "objective must score every fold via fit_and_score_full"


def test_objective_logs_auc_f1_macro_and_qwk_as_user_attrs_not_objective():
    fixed_scores = {"auc": 0.5, "f1_macro": 0.3, "pr_auc_macro": 0.7, "qwk": 0.2}
    search, _ = _make_search_with_stub(fixed_scores)

    folds = [(np.array([0, 1]), np.array([2, 3]))]
    objective = search._make_objective(
        _dummy_model_cls(), X_search=None, y_search=None, folds=folds, season="s", model_name="m",
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    attrs = study.best_trial.user_attrs
    assert attrs["auc"] == pytest.approx(0.5)
    assert attrs["f1_macro"] == pytest.approx(0.3)
    assert attrs["qwk"] == pytest.approx(0.2)
    # None of the secondary metrics leaked into what Optuna optimized.
    assert study.best_value not in (
        pytest.approx(attrs["auc"]), pytest.approx(attrs["f1_macro"]), pytest.approx(attrs["qwk"]),
    )


def test_objective_prunes_on_pr_auc_running_mean():
    """trial.report (which MedianPruner reads) must track PR-AUC-macro's
    running mean across folds, not AUC's/F1's/QWK's — otherwise pruning
    decisions would be made on a metric different from the one actually
    being optimized. auc/f1_macro/qwk are set to a much higher value than
    pr_auc_macro in this fixture, so reading the wrong metric would show up
    immediately as reported intermediate values near 0.9 instead of 0.1."""
    fixed_scores = {"auc": 0.9, "f1_macro": 0.9, "pr_auc_macro": 0.1, "qwk": 0.9}
    search, _ = _make_search_with_stub(fixed_scores)

    folds = [(np.array([0, 1]), np.array([2, 3])), (np.array([4, 5]), np.array([6, 7]))]
    objective = search._make_objective(
        _dummy_model_cls(), X_search=None, y_search=None, folds=folds, season="s", model_name="m",
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    # Every fold in this fixture reports the same pr_auc_macro (0.1), so the
    # running mean reported at every step must equal it exactly.
    trial = study.trials[0]
    assert trial.intermediate_values, "trial.report was never called"
    assert all(v == pytest.approx(0.1) for v in trial.intermediate_values.values())


def test_objective_excludes_a_partial_nan_fold_via_nanmean():
    """fix-spatial-cv-auc-missing-classes: self.cv_strategy is the
    PRODUCTION strategy (often spatial/stratified_spatial_block), so this
    objective can hit the same missing-class scoring path as
    trainer.py's _run_full_metrics_pass — fit_and_score_full's underlying
    scorer now returns NaN (not 0.0) for a genuinely degenerate fold. One
    NaN fold out of several must not poison the whole trial's tuned value
    (or its user_attrs) to NaN via a plain np.mean; the objective must
    exclude it (np.nanmean), same aggregation-safety decision as
    trainer.py's _run_full_metrics_pass."""
    scores_by_call = [
        {"auc": 0.8, "f1_macro": 0.6, "pr_auc_macro": 0.7, "qwk": 0.5},
        {"auc": float("nan"), "f1_macro": 0.4, "pr_auc_macro": float("nan"), "qwk": 0.3},
    ]
    search, stub_strategy = _make_search_with_sequence_stub(scores_by_call)

    folds = [(np.array([0, 1]), np.array([2, 3])), (np.array([4, 5]), np.array([6, 7]))]
    objective = search._make_objective(
        _dummy_model_cls(), X_search=None, y_search=None, folds=folds, season="s", model_name="m",
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    assert study.best_value == pytest.approx(0.7), "PR-AUC-macro must exclude the NaN fold, not average it in"
    attrs = study.best_trial.user_attrs
    assert attrs["auc"] == pytest.approx(0.8)
    assert attrs["qwk"] == pytest.approx((0.5 + 0.3) / 2)
    assert stub_strategy.calls == len(folds)


def test_objective_prunes_the_trial_when_every_fold_is_nan(caplog):
    """If every fold's pr_auc_macro is NaN (a genuinely degenerate search
    subsample/fold split), the objective must NOT return NaN directly —
    Optuna rejects a NaN objective return value outright ('The value nan
    is not acceptable'), which would mark the trial FAILED, and if every
    trial in a study hit this, study.best_value would raise ValueError
    ('No trials are completed yet'), crashing run_search_and_report
    entirely. Pruning (Optuna's own idiom for 'this trial can't be scored')
    keeps the trial out of best_value consideration without crashing the
    study, and a clear warning is logged either way."""
    scores_by_call = [
        {"auc": float("nan"), "f1_macro": 0.4, "pr_auc_macro": float("nan"), "qwk": 0.3},
    ]
    search, _ = _make_search_with_sequence_stub(scores_by_call)

    folds = [(np.array([0, 1]), np.array([2, 3]))]
    objective = search._make_objective(
        _dummy_model_cls(), X_search=None, y_search=None, folds=folds, season="s", model_name="m",
    )

    study = optuna.create_study(direction="maximize")
    with caplog.at_level(logging.WARNING):
        study.optimize(objective, n_trials=1)  # must not raise

    assert study.trials[0].state == optuna.trial.TrialState.PRUNED
    assert any(
        "every fold" in r.message and "pr_auc_macro" in r.message for r in caplog.records
    ), f"expected an explicit all-NaN warning, got: {[r.message for r in caplog.records]}"


def test_study_name_carries_the_objective_version_token(tmp_path, monkeypatch, fast_modeling_config):
    """The study name's param-space hash (test_param_space_versioning.py)
    only captures param_space() shape, not _make_objective's scoring logic
    — so switching the objective from AUC to PR-AUC-macro needed a
    separate, explicit version token in the study name (OBJECTIVE_VERSION)
    to avoid resuming/appending new PR-AUC trials into an old study full of
    AUC-scored trials via load_if_exists=True."""
    from wildfire_susceptibility.modeling.training.search import HyperparamSearch, OBJECTIVE_VERSION
    from wildfire_susceptibility.modeling.cv import get_cv_strategy
    from wildfire_susceptibility.modeling.resampling import SMOTEResampler
    from wildfire_susceptibility.modeling import models  # noqa: F401 — registers svm

    db_path = tmp_path / "optuna_test.db"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.search.OPTUNA_STORAGE",
        f"sqlite:///{db_path}",
    )

    config = fast_modeling_config
    config["modeling"]["cv_strategy"] = "standard"
    resampler = SMOTEResampler(config)
    cv_strategy = get_cv_strategy("standard", config, resampler)
    search = HyperparamSearch(config, cv_strategy)

    study = search.get_or_create_study("test_season", "svm")

    assert OBJECTIVE_VERSION in study.study_name, (
        f"study_name={study.study_name!r} must carry OBJECTIVE_VERSION so a future "
        f"objective-scoring change (without a param_space change) can't silently mix "
        f"trials scored under two different metrics in the same Optuna study."
    )
