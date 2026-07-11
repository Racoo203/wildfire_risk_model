# tests/test_modeling/test_mlflow_backend.py

import pytest


def test_model_trainer_uses_sqlite_backend(tmp_path, monkeypatch):
    import mlflow

    test_db = tmp_path / "mlflow_test.db"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.trainer.MLFLOW_TRACKING_URI",
        f"sqlite:///{test_db}",
    )

    from wildfire_susceptibility.modeling.training import ModelTrainer

    config = {
        "modeling": {
            "mlflow_experiment": "test-backend-check",
            "cv_folds": 2,
            "optuna_n_trials": 2,
            "use_smote": False,
            "smote_k_neighbors": 5,
        },
    }
    ModelTrainer(config)

    assert mlflow.get_tracking_uri().startswith("sqlite:///")