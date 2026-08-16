# tests/unit/modeling/training/test_optimism_gap_end_to_end.py
"""End-to-end coverage for optimism-gap logging through ModelTrainer.train_one
with cv_strategy="both". Optuna's objective is PR-AUC-macro (search.py); AUC/
F1-macro/QWK for the searched side come from the best trial's user_attrs, and
the diagnostic side always runs a full metrics pass, so all four metrics
(AUC, F1-macro, PR-AUC-macro, QWK) are available on both sides regardless of
log_full_cv_diagnostics:

- With modeling.log_full_cv_diagnostics=True (the dissertation's
  baseline.yaml setting): confirms the standard-vs-spatial full metrics
  passes run without error, populate the returned dict's
  cv_f1_macro_*/cv_pr_auc_macro_*/cv_qwk_*/cv_optimism_gap keys with
  full-training-set values (rather than the search subsample) on the
  searched side, fix the pre-existing bug where cv_auc_spatial_folds stayed
  None under this config (breaking stage_selection.py's Kruskal-Wallis
  test), and actually reach MLflow.
- With the flag left at its default (False): confirms the expensive
  primary-side full-training-set re-scoring pass is skipped entirely (not
  just its results discarded), while every metric's optimism gap is still
  logged correctly using the search subsample's cheap values on the
  searched side — only cv_auc_spatial_folds (which needs a full pass on the
  searched side specifically) stays unpopulated.

_run_full_metrics_pass's per-fold mean is np.nanmean, not np.mean (see
trainer.py) — a fold that cv/base.py's score_multiclass_fold scores as NaN
(fix-spatial-cv-auc-missing-classes: genuinely degenerate, fewer than 2
observed classes) is excluded from the mean rather than poisoning it."""

import logging
import math

import numpy as np
import pandas as pd
import pytest


N_BLOCKS = 24
ROWS_PER_BLOCK = 20
N_CLASSES = 4
CV_FOLDS = 3


@pytest.fixture
def spatial_dataset():
    """Feature values cleanly separate by class (so fitted models score
    meaningfully above chance) with every class present in every block, so
    stratified_spatial_block's fold assignment doesn't degenerate."""
    rng = np.random.default_rng(0)
    rows, labels, blocks = [], [], []
    for b in range(N_BLOCKS):
        for _ in range(ROWS_PER_BLOCK):
            cls = int(rng.integers(0, N_CLASSES))
            rows.append([cls * 10.0 + rng.normal(0, 0.5), rng.normal(0, 0.5)])
            labels.append(cls)
            blocks.append(f"blk_{b}")
    X = pd.DataFrame(rows, columns=["f1", "f2"])
    y = pd.Series(labels)
    groups = pd.Series(blocks)
    return X, y, groups


@pytest.fixture
def spatial_dataset_with_rare_class_confined_to_one_block():
    """Regression fixture for fix-spatial-cv-auc-missing-classes: unlike
    spatial_dataset above (every class present in every block, by design —
    see its docstring), class 3 here exists in exactly ONE spatial block.
    With CV_FOLDS=3, stratified_spatial_block's fold assignment
    (_seed_every_fold_with_rarest_class_coverage) puts that single block
    into exactly one fold's VALIDATION split, so that fold's TRAINING
    split has zero rows of class 3 (the only block carrying it is held
    out) while its validation split still contains it — the exact shape
    that broke AUC/PR-AUC scoring before the fix (predict_proba returning
    fewer columns than the full label space). The other two outer folds
    see zero rows of class 3 anywhere at all — degenerate for that one
    class specifically, though not for the fold as a whole, since classes
    0-2 are present in every block."""
    rng = np.random.default_rng(1)
    rows, labels, blocks = [], [], []
    for b in range(N_BLOCKS):
        block_id = f"blk_{b}"
        classes_this_block = [3] if b == 0 else [0, 1, 2]
        for _ in range(ROWS_PER_BLOCK):
            cls = int(rng.choice(classes_this_block))
            rows.append([cls * 10.0 + rng.normal(0, 0.5), rng.normal(0, 0.5)])
            labels.append(cls)
            blocks.append(block_id)
    X = pd.DataFrame(rows, columns=["f1", "f2"])
    y = pd.Series(labels)
    groups = pd.Series(blocks)
    return X, y, groups


def _make_trainer(tmp_path, monkeypatch, fast_modeling_config, experiment_name, log_full_cv_diagnostics):
    from wildfire_susceptibility.modeling.training import ModelTrainer

    test_db = tmp_path / "mlflow_test.db"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.trainer.MLFLOW_TRACKING_URI",
        f"sqlite:///{test_db}",
    )

    config = fast_modeling_config
    config["base"]["figures_dir"] = tmp_path
    config["modeling"]["mlflow_experiment"] = experiment_name
    config["modeling"]["cv_strategy"] = "both"
    config["modeling"]["cv_folds"] = CV_FOLDS
    config["modeling"].setdefault("spatial_block_size_m", 5000.0)
    config["modeling"]["log_full_cv_diagnostics"] = log_full_cv_diagnostics
    return ModelTrainer(config)


@pytest.fixture
def full_diagnostics_trainer(tmp_path, monkeypatch, fast_modeling_config):
    return _make_trainer(tmp_path, monkeypatch, fast_modeling_config, "test-optimism-gap-full", True)


@pytest.fixture
def cheap_default_trainer(tmp_path, monkeypatch, fast_modeling_config):
    return _make_trainer(tmp_path, monkeypatch, fast_modeling_config, "test-optimism-gap-cheap", False)


def test_train_one_populates_optimism_gap_metrics(full_diagnostics_trainer, spatial_dataset):
    X, y, groups = spatial_dataset

    result = full_diagnostics_trainer.train_one(
        season="test_season",
        model_name="random_forest",
        X_train=X, y_train=y, X_val=X, y_val=y,
        groups_train=groups,
        ref_path=None,
        run_post_training_evaluation=False,
    )

    # standard-vs-spatial AUC gap already existed pre-this-change; F1-macro,
    # PR-AUC-macro, and QWK are new.
    assert result["cv_optimism_gap"] is not None
    assert set(result["cv_optimism_gap"]) == {"auc", "f1_macro", "pr_auc_macro", "qwk"}
    for v in result["cv_optimism_gap"].values():
        assert isinstance(v, float)

    for key in (
        "cv_auc_standard", "cv_auc_spatial",
        "cv_f1_macro_standard", "cv_f1_macro_spatial",
        "cv_pr_auc_macro_standard", "cv_pr_auc_macro_spatial",
        "cv_qwk_standard", "cv_qwk_spatial",
    ):
        assert isinstance(result[key], float), f"{key} was not populated: {result[key]!r}"

    # Regression coverage for the pre-existing bug where cv_auc_spatial_folds
    # stayed None under cv_strategy="both" (primary always resolves to
    # stratified_spatial_block, which requires groups) — this list feeds
    # stage_selection.py's Kruskal-Wallis cross-model test, which silently
    # no-op'd every season as a result.
    assert result["cv_auc_spatial_folds"] is not None
    assert len(result["cv_auc_spatial_folds"]) == CV_FOLDS
    assert all(isinstance(v, float) for v in result["cv_auc_spatial_folds"])


def test_train_one_logs_optimism_gap_metrics_to_mlflow(full_diagnostics_trainer, spatial_dataset):
    import mlflow

    X, y, groups = spatial_dataset
    full_diagnostics_trainer.train_one(
        season="test_season",
        model_name="random_forest",
        X_train=X, y_train=y, X_val=X, y_val=y,
        groups_train=groups,
        ref_path=None,
        run_post_training_evaluation=False,
    )

    runs = mlflow.search_runs(experiment_names=["test-optimism-gap-full"])
    assert len(runs) == 1
    row = runs.iloc[0]

    for metric in (
        "metrics.cv_f1_macro_optimism_gap",
        "metrics.cv_pr_auc_macro_optimism_gap",
        "metrics.cv_auc_optimism_gap",
        "metrics.cv_qwk_optimism_gap",
        "metrics.cv_f1_macro_standard",
        "metrics.cv_f1_macro_spatial",
        "metrics.cv_pr_auc_macro_standard",
        "metrics.cv_pr_auc_macro_spatial",
        "metrics.cv_qwk_standard",
        "metrics.cv_qwk_spatial",
    ):
        assert metric in row.index, f"{metric} missing from logged run"
        assert not pd.isna(row[metric]), f"{metric} was logged as NaN"


def test_log_full_cv_diagnostics_defaults_off_and_skips_the_expensive_pass(
    cheap_default_trainer, spatial_dataset, monkeypatch,
):
    """With the flag left at its schema default (False), the primary
    strategy's full-training-set re-scoring pass — ModelTrainer's own
    _run_full_metrics_pass, called with role="primary" — must not run at
    all (not just have its results discarded), since that call is the ~2x
    cost this flag exists to gate."""
    from wildfire_susceptibility.modeling.training.trainer import ModelTrainer

    seen_roles = []
    original = ModelTrainer._run_full_metrics_pass

    def _spy(self, strategy, model_cls, best_params, X_tr, y_train, groups_train, season, model_name, role):
        seen_roles.append(role)
        return original(self, strategy, model_cls, best_params, X_tr, y_train, groups_train, season, model_name, role)

    monkeypatch.setattr(ModelTrainer, "_run_full_metrics_pass", _spy)

    X, y, groups = spatial_dataset
    result = cheap_default_trainer.train_one(
        season="test_season",
        model_name="random_forest",
        X_train=X, y_train=y, X_val=X, y_val=y,
        groups_train=groups,
        ref_path=None,
        run_post_training_evaluation=False,
    )

    assert seen_roles == ["diagnostic"], (
        f"expected only the diagnostic pass to run when log_full_cv_diagnostics=False (default), "
        f"got roles={seen_roles}"
    )

    # Every metric's gap is still logged correctly (Bug A fix applies
    # unconditionally, and AUC/F1-macro/PR-AUC-macro/QWK are all available
    # cheaply on both sides even without the primary-side full pass: the
    # diagnostic side from its own full pass, the searched side from
    # Optuna's best_value/user_attrs).
    assert set(result["cv_optimism_gap"]) == {"auc", "f1_macro", "pr_auc_macro", "qwk"}
    assert result["cv_optimism_gap"]["auc"] == pytest.approx(
        result["cv_auc_standard"] - result["cv_auc_spatial"]
    )
    assert result["cv_optimism_gap"]["pr_auc_macro"] == pytest.approx(
        result["cv_pr_auc_macro_standard"] - result["cv_pr_auc_macro_spatial"]
    )
    for key in (
        "cv_auc_standard", "cv_auc_spatial",
        "cv_f1_macro_standard", "cv_f1_macro_spatial",
        "cv_pr_auc_macro_standard", "cv_pr_auc_macro_spatial",
        "cv_qwk_standard", "cv_qwk_spatial",
    ):
        assert isinstance(result[key], float), f"{key} should be populated cheaply, got {result[key]!r}"

    # ...but cv_auc_spatial_folds specifically needs a full pass on the
    # *searched* side (for stage_selection's Kruskal-Wallis) and stays None.
    assert result["cv_auc_spatial_folds"] is None


def test_train_one_survives_a_class_confined_to_a_single_spatial_block(
    full_diagnostics_trainer, spatial_dataset_with_rare_class_confined_to_one_block,
):
    """Regression test for fix-spatial-cv-auc-missing-classes, run through
    the full train_one -> _run_full_metrics_pass -> mlflow pipeline (not
    just the pure scorer unit tests in test_score_multiclass_fold.py).
    Before the fix, the fold whose training split lost class 3 entirely
    made AUC/PR-AUC scoring raise and get silently reported as 0.0 — this
    must now complete without crashing or raising, produce real (not
    all-identical-0.0) cv_auc_spatial_folds, and log real, non-NaN
    aggregate metrics to mlflow (the aggregate must stay a real number as
    long as at least one fold was scorable, per _run_full_metrics_pass's
    np.nanmean aggregation — only a fold missing EVERY class would poison
    it)."""
    import mlflow

    X, y, groups = spatial_dataset_with_rare_class_confined_to_one_block

    result = full_diagnostics_trainer.train_one(
        season="test_season",
        model_name="random_forest",
        X_train=X, y_train=y, X_val=X, y_val=y,
        groups_train=groups,
        ref_path=None,
        run_post_training_evaluation=False,
    )

    assert result["cv_auc_spatial_folds"] is not None
    assert len(result["cv_auc_spatial_folds"]) == CV_FOLDS
    folds = result["cv_auc_spatial_folds"]

    # Classes 0-2 are present in every block, so no outer fold can be
    # degenerate as a whole — every fold's AUC must be a real, finite
    # score, not the old bug's constant 0.0 fallback.
    assert all(not math.isnan(v) for v in folds), f"expected every fold scorable, got {folds}"
    assert not all(v == 0.0 for v in folds), "fold AUCs collapsing to the old constant-0.0 bug"

    for key in ("cv_auc_spatial", "cv_f1_macro_spatial", "cv_pr_auc_macro_spatial", "cv_qwk_spatial"):
        assert isinstance(result[key], float)
        assert not math.isnan(result[key])

    runs = mlflow.search_runs(experiment_names=["test-optimism-gap-full"])
    row = runs.iloc[-1]
    for metric in ("metrics.cv_auc_spatial", "metrics.cv_pr_auc_macro_spatial"):
        assert metric in row.index
        assert not pd.isna(row[metric]), f"{metric} was logged as NaN"


# ----------------------------------------------------------------------
# fix-spatial-cv-auc-missing-classes: _run_full_metrics_pass's mean
# aggregation must be NaN-aware, since cv/base.py::score_multiclass_fold
# now returns NaN (not 0.0) for a genuinely degenerate fold. Exercised
# directly (not through train_one) via a stub CVStrategy so the per-fold
# metric values are exactly controlled.
# ----------------------------------------------------------------------

class _StubFoldStrategy:
    """Minimal CVStrategy stand-in for testing _run_full_metrics_pass's
    mean-aggregation in isolation, independent of any real fold-building
    or model-fitting machinery."""

    name = "standard"

    def __init__(self, scores_by_fold):
        self._scores_by_fold = scores_by_fold
        self._n = 0

    def make_folds(self, X, y, groups):
        return [(None, None)] * len(self._scores_by_fold)

    def fit_and_score_full(self, model_cls, params, X, y, train_idx, test_idx, context="", model_name=None):
        scores = dict(self._scores_by_fold[self._n])
        self._n += 1
        return scores


def test_run_full_metrics_pass_mean_excludes_a_partial_nan_fold(full_diagnostics_trainer):
    """One fold's AUC/PR-AUC are NaN (as if that fold alone were genuinely
    degenerate); the other two are real numbers. The reported mean must be
    the mean of the real folds only (np.nanmean), not NaN itself — a
    single unscorable fold shouldn't poison the whole reported estimate
    when other folds are perfectly fine."""
    strategy = _StubFoldStrategy([
        {"auc": 0.8, "f1_macro": 0.6, "pr_auc_macro": 0.7, "qwk": 0.5},
        {"auc": float("nan"), "f1_macro": 0.4, "pr_auc_macro": float("nan"), "qwk": 0.3},
        {"auc": 0.6, "f1_macro": 0.5, "pr_auc_macro": 0.5, "qwk": 0.4},
    ])
    X = pd.DataFrame({"f1": [0.0]})
    y = pd.Series([0])

    mean, per_fold = full_diagnostics_trainer._run_full_metrics_pass(
        strategy, object, {}, X, y, None, "test_season", "random_forest", role="test",
    )

    assert len(per_fold) == 3
    assert mean["auc"] == pytest.approx((0.8 + 0.6) / 2), "NaN fold must be excluded, not averaged in"
    assert mean["pr_auc_macro"] == pytest.approx((0.7 + 0.5) / 2)
    # f1_macro/qwk had no NaN folds — plain mean, unaffected by the change.
    assert mean["f1_macro"] == pytest.approx((0.6 + 0.4 + 0.5) / 3)
    assert mean["qwk"] == pytest.approx((0.5 + 0.3 + 0.4) / 3)


def test_run_full_metrics_pass_mean_is_nan_and_warns_when_every_fold_is_nan(
    full_diagnostics_trainer, caplog,
):
    """If every fold for a metric is NaN (all genuinely degenerate), the
    mean must stay NaN — silently averaging to some other number would
    misrepresent a config where the reported estimate is entirely
    unscorable — and a clear warning must be logged rather than relying on
    numpy's own silent 'Mean of empty slice' RuntimeWarning as the only
    signal."""
    strategy = _StubFoldStrategy([
        {"auc": float("nan"), "f1_macro": 0.6, "pr_auc_macro": float("nan"), "qwk": 0.5},
        {"auc": float("nan"), "f1_macro": 0.4, "pr_auc_macro": float("nan"), "qwk": 0.3},
        {"auc": float("nan"), "f1_macro": 0.5, "pr_auc_macro": float("nan"), "qwk": 0.4},
    ])
    X = pd.DataFrame({"f1": [0.0]})
    y = pd.Series([0])

    with caplog.at_level(logging.WARNING):
        mean, per_fold = full_diagnostics_trainer._run_full_metrics_pass(
            strategy, object, {}, X, y, None, "test_season", "random_forest", role="test",
        )

    assert np.isnan(mean["auc"])
    assert np.isnan(mean["pr_auc_macro"])
    assert mean["f1_macro"] == pytest.approx((0.6 + 0.4 + 0.5) / 3)
    assert any(
        "every fold" in r.message and "auc" in r.message for r in caplog.records
    ), f"expected an explicit all-NaN warning, got: {[r.message for r in caplog.records]}"
