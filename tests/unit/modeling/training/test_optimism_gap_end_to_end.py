# tests/unit/modeling/training/test_optimism_gap_end_to_end.py
"""End-to-end coverage for optimism-gap logging through ModelTrainer.train_one
with cv_strategy="both", post the refactor-nested-cv-optuna-schratz-method
branch. Under cv_strategy="both", __init__ always resolves the primary
strategy to stratified_spatial_block (spatial), so the PRIMARY side's
genuinely nested CV pass (nested_cv.run_nested_cv) is now MANDATORY and
unconditional — there is no more non-nested/cheap fallback for it, since
that used to be exactly the Varma & Simon (2006) bias this branch removes.

log_full_cv_diagnostics now gates only the DIAGNOSTIC (standard, comparison-
only) side's own nested CV pass — this is the flip from the pre-nested-CV
behavior (where the diagnostic side always ran and the flag gated the
PRIMARY side's full pass instead):

- log_full_cv_diagnostics=True (the dissertation's baseline.yaml setting):
  both sides run genuinely nested CV, cv_optimism_gap is populated from two
  real nested estimates, and cv_auc_spatial_folds (consumed by
  stage_selection's Kruskal-Wallis test) is populated from the primary
  (spatial) side's outer-fold AUCs.
- log_full_cv_diagnostics=False (schema default): only the primary
  (spatial) side runs nested CV — cv_auc_spatial/cv_f1_macro_spatial/
  cv_pr_auc_macro_spatial/cv_qwk_spatial and cv_auc_spatial_folds are still
  populated (mandatory, unconditional), but the standard side and
  cv_optimism_gap stay None, since the diagnostic pass never runs."""

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

    # cv_auc_spatial_folds feeds stage_selection.py's Kruskal-Wallis
    # cross-model test — these are now genuinely nested outer-fold AUCs
    # (refit-best-inner-params-on-outer-train, score-on-outer-test), one
    # per outer fold of the primary (spatial) strategy.
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


def test_log_full_cv_diagnostics_defaults_off_and_skips_the_diagnostic_side(
    cheap_default_trainer, spatial_dataset, monkeypatch,
):
    """With the flag left at its schema default (False), the DIAGNOSTIC
    (standard) side's nested CV pass — ModelTrainer's own _run_nested_cv,
    called a second time for the diagnostic strategy — must not run at
    all (not just have its results discarded): that's the ~half of
    'both' mode's nested-CV cost this flag exists to gate. The PRIMARY
    (spatial) side's nested CV pass is unconditional and must still run
    exactly once."""
    from wildfire_susceptibility.modeling.training.trainer import ModelTrainer

    call_strategy_names = []
    original = ModelTrainer._run_nested_cv

    def _spy(self, strategy, *args, **kwargs):
        call_strategy_names.append(strategy.name)
        return original(self, strategy, *args, **kwargs)

    monkeypatch.setattr(ModelTrainer, "_run_nested_cv", _spy)

    X, y, groups = spatial_dataset
    result = cheap_default_trainer.train_one(
        season="test_season",
        model_name="random_forest",
        X_train=X, y_train=y, X_val=X, y_val=y,
        groups_train=groups,
        ref_path=None,
        run_post_training_evaluation=False,
    )

    assert call_strategy_names == ["stratified_spatial_block"], (
        f"expected only the primary (spatial) side's nested CV to run when "
        f"log_full_cv_diagnostics=False (default), got {call_strategy_names}"
    )

    # Primary (spatial) side is mandatory and unbiased regardless of the flag.
    for key in (
        "cv_auc_spatial", "cv_f1_macro_spatial", "cv_pr_auc_macro_spatial", "cv_qwk_spatial",
    ):
        assert isinstance(result[key], float), f"{key} should be populated, got {result[key]!r}"
    assert result["cv_auc_spatial_folds"] is not None
    assert len(result["cv_auc_spatial_folds"]) == CV_FOLDS

    # Diagnostic (standard) side never ran, so its fields and the gap stay None.
    for key in ("cv_auc_standard", "cv_f1_macro_standard", "cv_pr_auc_macro_standard", "cv_qwk_standard"):
        assert result[key] is None, f"{key} should stay None when the diagnostic side is skipped, got {result[key]!r}"
    assert result["cv_optimism_gap"] is None
