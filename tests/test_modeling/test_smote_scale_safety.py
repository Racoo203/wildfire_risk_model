# tests/test_modeling/test_smote_scale_safety.py
"""Regression test for SMOTE targets sized for the full dataset
exploding a small search subsample — the 4,800 -> 703,090 row bug."""

import numpy as np
import pytest
from wildfire_susceptibility.modeling.training.resampling import SMOTEResampler


def test_smote_caps_targets_to_batch_majority_size():
    resampler = SMOTEResampler({
        "modeling": {
            "use_smote": True,
            "smote_k_neighbors": 5,
            "smote_sampling_strategy": {1: 400000, 2: 200000, 3: 100000},
        }
    })
    rng = np.random.default_rng(0)
    X = rng.normal(size=(4800, 3))
    y = np.array([0] * 3090 + [1] * 1342 + [2] * 320 + [3] * 48)

    X_res, y_res = resampler.resample(X, y, context="test")

    # Must never exceed the batch's own majority class count (3090) per class —
    # NOT the config's absolute 400000/200000/100000 targets.
    _, counts = np.unique(y_res, return_counts=True)
    assert counts.max() <= 3090
    assert len(X_res) <= 3090 * 4  # sane upper bound, nowhere near 703,090


def test_trainer_disables_smote_for_search_but_not_final_refit(tmp_path, monkeypatch):
    from wildfire_susceptibility.modeling.training import ModelTrainer

    test_db = tmp_path / "mlflow_test.db"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.trainer.MLFLOW_TRACKING_URI",
        f"sqlite:///{test_db}",
    )
    config = {
        "modeling": {
            "mlflow_experiment": "test-smote-split",
            "cv_folds": 2, "optuna_n_trials": 1,
            "use_smote": True, "smote_k_neighbors": 5,
            "smote_sampling_strategy": {1: 400000},
        },
    }
    trainer = ModelTrainer(config)

    assert trainer.fold_strategy.resampler.enabled is False  # search/CV: SMOTE off
    assert trainer.final_resampler.enabled is True            # final refit: SMOTE on