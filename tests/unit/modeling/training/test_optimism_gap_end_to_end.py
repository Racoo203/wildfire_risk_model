# tests/unit/modeling/training/test_optimism_gap_end_to_end.py
"""End-to-end coverage for optimism-gap logging through ModelTrainer.train_one
with cv_strategy="both":

- With modeling.log_full_cv_diagnostics=True (the dissertation's
  baseline.yaml setting): confirms the standard-vs-spatial full metrics
  passes run without error, populate the returned dict's new
  cv_f1_macro_*/cv_pr_auc_macro_*/cv_optimism_gap keys, fix the pre-existing
  bug where cv_auc_spatial_folds stayed None under this config (breaking
  stage_selection.py's Kruskal-Wallis test), and actually reach MLflow.
- With the flag left at its default (False): confirms the expensive
  primary-side full-training-set re-scoring pass is skipped entirely (not
  just its results discarded), while the pre-existing AUC-only optimism gap
  still gets logged correctly (now numerically fixed vs. the old multiclass
  bug, just not the new F1-macro/PR-AUC-macro/fold-level additions)."""

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

    # standard-vs-spatial AUC gap already existed pre-this-change; F1-macro
    # and PR-AUC-macro are new.
    assert result["cv_optimism_gap"] is not None
    assert set(result["cv_optimism_gap"]) == {"auc", "f1_macro", "pr_auc_macro"}
    for v in result["cv_optimism_gap"].values():
        assert isinstance(v, float)

    for key in (
        "cv_f1_macro_standard", "cv_f1_macro_spatial",
        "cv_pr_auc_macro_standard", "cv_pr_auc_macro_spatial",
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
        "metrics.cv_f1_macro_standard",
        "metrics.cv_f1_macro_spatial",
        "metrics.cv_pr_auc_macro_standard",
        "metrics.cv_pr_auc_macro_spatial",
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

    # AUC-only gap still logged correctly (Bug A fix applies unconditionally)...
    assert result["cv_optimism_gap"] == {"auc": pytest.approx(result["cv_auc_standard"] - result["cv_auc_spatial"])}
    # ...but nothing that requires the primary-side pass is populated.
    for key in (
        "cv_f1_macro_standard", "cv_f1_macro_spatial",
        "cv_pr_auc_macro_standard", "cv_pr_auc_macro_spatial",
        "cv_auc_spatial_folds",
    ):
        assert result[key] is None, f"{key} should stay None when log_full_cv_diagnostics is False, got {result[key]!r}"
