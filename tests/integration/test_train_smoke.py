# tests/test_pipeline/test_train_smoke.py
"""Fast smoke test for the full train_one() call path, using tiny
synthetic data — catches registry, scaling, CV, and mlflow wiring bugs
in seconds instead of after a multi-hour real run."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_training_config(tmp_path, fast_modeling_config):
    config = fast_modeling_config
    config["base"]["figures_dir"] = str(tmp_path / "figures")
    config["base"]["output_dir"] = str(tmp_path / "layers")
    config["modeling"]["models"] = ["random_forest"]
    config["modeling"]["mlflow_experiment"] = "test-smoke"
    config["modeling"]["cv_strategy"] = "standard"
    config["modeling"]["spatial_block_size_m"] = 5000.0
    return config

@pytest.fixture
def tiny_dataset():
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame({
        "elevation": rng.normal(size=n),
        "slope": rng.normal(size=n),
        "ndvi": rng.normal(size=n),
    })
    y = pd.Series(rng.integers(0, 4, size=n))
    return X, y

@pytest.mark.slow
def test_train_one_runs_end_to_end(tiny_training_config, tiny_dataset, tmp_path, monkeypatch, fast_modeling_config):
    """train_one() must complete without raising, and MODELS must resolve
    'random_forest' — this is the exact call path that failed with
    'Model random_forest not found in MODELS registry: []'."""
    import mlflow

    # Isolate this test's mlflow tracking from the real project DB, and
    # from ModelTrainer's own _ensure_mlflow_backend() call — monkeypatch
    # the tracking URI it points to so tests never touch data/silver/dbs/.
    test_db = tmp_path / "mlflow_test.db"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.trainer.MLFLOW_TRACKING_URI",
        f"sqlite:///{test_db}",
    )

    from wildfire_susceptibility.modeling.training import ModelTrainer

    X, y = tiny_dataset
    trainer = ModelTrainer(tiny_training_config)

    result = trainer.train_one(
        season="test_season",
        model_name="random_forest",
        X_train=X, y_train=y,
        X_val=X, y_val=y,
        groups_train=None,
        ref_path=None,
    )

    assert result["model"] is not None
    assert 0.0 <= result["cv_auc_standard"] <= 1.0
    assert result["val_auc"] is not None
    assert result["val_pr_auc_macro"] is not None

@pytest.mark.slow
def test_train_one_raises_clear_error_for_unregistered_model(tiny_training_config, tiny_dataset, tmp_path, monkeypatch, fast_modeling_config):
    """If someone breaks the registry import again, this should fail loudly
    with the model name, not silently produce garbage."""
    import mlflow

    test_db = tmp_path / "mlflow_test.db"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.trainer.MLFLOW_TRACKING_URI",
        f"sqlite:///{test_db}",
    )

    from wildfire_susceptibility.modeling.training import ModelTrainer

    X, y = tiny_dataset
    tiny_training_config["modeling"]["models"] = ["totally_not_a_model"]
    trainer = ModelTrainer(tiny_training_config)

    with pytest.raises(ValueError, match="not found in MODELS registry"):
        trainer.train_one(
            season="test_season", model_name="totally_not_a_model",
            X_train=X, y_train=y, X_val=X, y_val=y,
        )