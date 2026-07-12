"""Guards against SVM hyperparameter search silently taking hours: a
kernel SVM fit on the per-model subsample cap must complete quickly, and
the trainer must actually apply the per-model override rather than the
generic optuna_search_subsample."""

import time
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def big_dataset():
    rng = np.random.default_rng(0)
    n = 20000
    X = pd.DataFrame({
        "elevation": rng.normal(size=n),
        "slope": rng.normal(size=n),
        "ndvi": rng.normal(size=n),
    })
    y = pd.Series(rng.integers(0, 4, size=n))
    return X, y


def _make_trainer(tmp_path, monkeypatch, fast_modeling_config, subsample_by_model=None):
    from wildfire_susceptibility.modeling.training import ModelTrainer

    test_db = tmp_path / "mlflow_test.db"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.trainer.MLFLOW_TRACKING_URI",
        f"sqlite:///{test_db}",
    )
    config = fast_modeling_config
    config["modeling"]["mlflow_experiment"] = "test-svm-scaling"
    config["modeling"]["optuna_search_subsample"] = 50000
    config["modeling"]["optuna_search_subsample_by_model"] = subsample_by_model or {"svm": 2000}
    return ModelTrainer(config)


def test_svm_uses_per_model_subsample_override(tmp_path, monkeypatch, big_dataset, fast_modeling_config):
    X, y = big_dataset
    trainer = _make_trainer(tmp_path, monkeypatch, fast_modeling_config, subsample_by_model={"svm": 2000})

    X_search, y_search = trainer._subsample_for_search(X, y, "test_season", "svm")
    assert len(X_search) == 2000  # not the generic 50000



def test_random_forest_ignores_svm_override(tmp_path, monkeypatch, big_dataset, fast_modeling_config):
    X, y = big_dataset
    trainer = _make_trainer(tmp_path, monkeypatch, fast_modeling_config, subsample_by_model={"svm": 2000})

    X_search, y_search = trainer._subsample_for_search(X, y, "test_season", "random_forest")
    assert len(X_search) == len(X)  # falls back to generic (bigger than dataset, so no-op)


def test_svm_calibrated_fit_completes_quickly_at_capped_size(tmp_path, monkeypatch, big_dataset):
    """Not a hard perf assertion (CI machines vary), but catches a
    regression back to SVC(probability=True)'s uncontrolled internal CV,
    which would make even this tiny case noticeably slower."""
    from wildfire_susceptibility.core.registry import MODELS
    from wildfire_susceptibility.modeling import models  # noqa: F401

    X, y = big_dataset
    X_small, y_small = X.iloc[:2000].values, y.iloc[:2000].values

    model = MODELS["svm"]()
    start = time.time()
    model.fit(X_small, y_small)
    elapsed = time.time() - start

    proba = model.predict_proba(X_small)
    assert proba.shape == (2000, 4)
    assert elapsed < 30, f"SVM fit on 2000 rows took {elapsed:.1f}s — check CalibratedClassifierCV cv= wasn't reverted"


def test_smote_application_is_logged_at_info(caplog):
    """Regression test for the 'was SMOTE even applied?' confusion — the
    resample summary must be visible at INFO, not buried at DEBUG."""
    import logging
    from wildfire_susceptibility.modeling.training.resampling import SMOTEResampler

    caplog.set_level(logging.INFO)
    resampler = SMOTEResampler({"modeling": {"use_smote": True, "smote_k_neighbors": 3}})

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = np.array([0] * 70 + [1] * 20 + [2] * 8 + [3] * 2)

    resampler.resample(X, y, context="[test]")
    assert "SMOTE applied" in caplog.text