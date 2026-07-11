import pytest

def test_model_trainer_uses_sqlite_backend(tmp_path, monkeypatch, fast_modeling_config):
    import mlflow

    test_db = tmp_path / "mlflow_test.db"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.trainer.MLFLOW_TRACKING_URI",
        f"sqlite:///{test_db}",
    )

    from wildfire_susceptibility.modeling.training import ModelTrainer

    fast_modeling_config["modeling"]["mlflow_experiment"] = "test-backend-check"
    ModelTrainer(fast_modeling_config)

    assert mlflow.get_tracking_uri().startswith("sqlite:///")