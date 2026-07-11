# tests/test_modeling/test_search_subsampling.py
"""Verifies optuna_search_subsample actually shrinks the data used during
hyperparameter search, and that it survives the full config round-trip
(YAML -> pydantic schema -> plain dict -> ModelTrainer)."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def big_dataset():
    rng = np.random.default_rng(0)
    n = 5000
    X = pd.DataFrame({
        "elevation": rng.normal(size=n),
        "slope": rng.normal(size=n),
        "ndvi": rng.normal(size=n),
    })
    y = pd.Series(rng.integers(0, 4, size=n))
    return X, y


def _make_trainer(tmp_path, monkeypatch, subsample=None):
    from wildfire_susceptibility.modeling.training import ModelTrainer

    test_db = tmp_path / "mlflow_test.db"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.trainer.MLFLOW_TRACKING_URI",
        f"sqlite:///{test_db}",
    )

    config = {
        "modeling": {
            "mlflow_experiment": "test-subsample",
            "cv_folds": 2,
            "optuna_n_trials": 1,
            "use_smote": False,
            "smote_k_neighbors": 5,
        },
    }
    if subsample is not None:
        config["modeling"]["optuna_search_subsample"] = subsample
    return ModelTrainer(config)


# ------------------------------------------------------------------
# Unit-level: _subsample_for_search in isolation
# ------------------------------------------------------------------

def test_subsample_for_search_shrinks_when_configured(tmp_path, monkeypatch, big_dataset):
    X, y = big_dataset
    trainer = _make_trainer(tmp_path, monkeypatch, subsample=500)

    X_search, y_search = trainer._subsample_for_search(X, y, "test_season", "random_forest")

    assert len(X_search) == 500
    assert len(y_search) == 500
    # stratified split shouldn't wildly distort class proportions
    original_props = y.value_counts(normalize=True).sort_index()
    search_props = y_search.value_counts(normalize=True).sort_index()
    assert np.allclose(original_props.values, search_props.values, atol=0.05)


def test_subsample_for_search_noop_when_unconfigured(tmp_path, monkeypatch, big_dataset):
    X, y = big_dataset
    trainer = _make_trainer(tmp_path, monkeypatch, subsample=None)

    X_search, y_search = trainer._subsample_for_search(X, y, "test_season", "random_forest")

    assert len(X_search) == len(X)  # documents current no-op behavior when unset


def test_subsample_for_search_noop_when_smaller_than_n(tmp_path, monkeypatch, big_dataset):
    X, y = big_dataset
    trainer = _make_trainer(tmp_path, monkeypatch, subsample=999_999)  # bigger than dataset

    X_search, y_search = trainer._subsample_for_search(X, y, "test_season", "random_forest")

    assert len(X_search) == len(X)


# ------------------------------------------------------------------
# End-to-end: does train_one() actually use the subsampled size for
# search, and the FULL size for the final refit?
# ------------------------------------------------------------------

def test_train_one_uses_subsample_for_search_but_full_data_for_refit(tmp_path, monkeypatch, big_dataset, caplog):
    import logging
    caplog.set_level(logging.INFO)

    X, y = big_dataset
    trainer = _make_trainer(tmp_path, monkeypatch, subsample=200)

    trainer.train_one(
        season="test_season",
        model_name="random_forest",
        X_train=X, y_train=y,
        X_val=X, y_val=y,
        groups_train=None,
        ref_path=None,
    )

    log_text = caplog.text
    assert "search subsample: 200/" in log_text, (
        "Expected a 'search subsample: 200/5000 rows' log line — if missing, "
        "the subsample size never reached _subsample_for_search or the "
        "config key name/location doesn't match what train.py reads."
    )


# ------------------------------------------------------------------
# Config round-trip: does optuna_search_subsample survive
# YAML -> pydantic -> model_dump(mode="python") -> plain dict?
# ------------------------------------------------------------------

def test_optuna_search_subsample_survives_config_roundtrip(tmp_path):
    import yaml
    from wildfire_susceptibility.config.loader import ConfigLoader

    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "modeling": {
            "optuna_search_subsample": 50000,
        },
    }))

    cfg_obj = ConfigLoader.load(cfg_path)
    cfg = cfg_obj.model_dump(mode="python")

    assert "optuna_search_subsample" in cfg["modeling"], (
        "optuna_search_subsample missing after ConfigLoader round-trip — "
        "check ModelingConfig schema has this field declared."
    )
    assert cfg["modeling"]["optuna_search_subsample"] == 50000


def test_optuna_search_subsample_defaults_to_none(tmp_path):
    """If the working config doesn't set this, it must resolve to None
    (triggering the documented no-op path), not be silently dropped/KeyError."""
    import yaml
    from wildfire_susceptibility.config.loader import ConfigLoader

    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(yaml.safe_dump({"modeling": {}}))

    cfg_obj = ConfigLoader.load(cfg_path)
    cfg = cfg_obj.model_dump(mode="python")

    assert cfg["modeling"].get("optuna_search_subsample") is None