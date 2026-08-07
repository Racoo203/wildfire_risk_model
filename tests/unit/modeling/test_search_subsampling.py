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


def _make_trainer(tmp_path, monkeypatch, fast_modeling_config, subsample=None):
    from wildfire_susceptibility.modeling.training import ModelTrainer

    test_db = tmp_path / "mlflow_test.db"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.trainer.MLFLOW_TRACKING_URI",
        f"sqlite:///{test_db}",
    )

    config = fast_modeling_config
    config["base"]["figures_dir"] = tmp_path
    config["modeling"]["mlflow_experiment"] = "test-subsample"
    config["modeling"]["cv_strategy"] = "standard"   # test is about subsampling, not spatial CV
    if subsample is not None:
        config["modeling"]["optuna_search_subsample"] = subsample
    else:
        config["modeling"].pop("optuna_search_subsample", None)
    return ModelTrainer(config)


def test_subsample_for_search_shrinks_when_configured(tmp_path, monkeypatch, big_dataset, fast_modeling_config):
    X, y = big_dataset
    trainer = _make_trainer(tmp_path, monkeypatch, fast_modeling_config, subsample=500)

    n = trainer.search._resolve_subsample_size("random_forest", population_size=len(X))
    X_search, y_search = trainer.search._subsample_rows(X, y, n, "test_season", "random_forest")

    assert len(X_search) == 500
    assert len(y_search) == 500
    original_props = y.value_counts(normalize=True).sort_index()
    search_props = y_search.value_counts(normalize=True).sort_index()
    assert np.allclose(original_props.values, search_props.values, atol=0.05)



def test_subsample_for_search_noop_when_unconfigured(tmp_path, monkeypatch, big_dataset, fast_modeling_config):
    X, y = big_dataset
    trainer = _make_trainer(tmp_path, monkeypatch, fast_modeling_config, subsample=None)

    n = trainer.search._resolve_subsample_size("random_forest", population_size=len(X))
    X_search, y_search = trainer.search._subsample_rows(X, y, n, "test_season", "random_forest")

    assert len(X_search) == len(X)


def test_subsample_for_search_noop_when_smaller_than_n(tmp_path, monkeypatch, big_dataset, fast_modeling_config):
    X, y = big_dataset
    trainer = _make_trainer(tmp_path, monkeypatch, fast_modeling_config, subsample=999_999)

    n = trainer.search._resolve_subsample_size("random_forest", population_size=len(X))
    X_search, y_search = trainer.search._subsample_rows(X, y, n, "test_season", "random_forest")

    assert len(X_search) == len(X)


def test_train_one_uses_subsample_for_search_but_full_data_for_refit(tmp_path, monkeypatch, big_dataset, fast_modeling_config, caplog):
    import logging
    caplog.set_level(logging.INFO)

    X, y = big_dataset
    trainer = _make_trainer(tmp_path, monkeypatch, fast_modeling_config, subsample=200)

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


def test_optuna_search_subsample_survives_config_roundtrip(tmp_path):
    import yaml
    from wildfire_susceptibility.config.loader import ConfigLoader

    cfg_path = tmp_path / "test_config.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "modeling": {
            "optuna_search_subsample": 50000,
        },
    }))

    cfg_obj = ConfigLoader.load(config_dir=tmp_path, files=[cfg_path.name])
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

    cfg_obj = ConfigLoader.load(config_dir=tmp_path, files=[cfg_path.name])
    cfg = cfg_obj.model_dump(mode="python")

    assert cfg["modeling"].get("optuna_search_subsample") is None