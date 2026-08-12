# tests/unit/modeling/training/test_final_refit_class_weight.py
"""Regression coverage for the HPO-audit leak reaching the shipped model:
random_forest's class_weight used to be an Optuna-tunable categorical, so
a trial could win the search with class_weight="balanced" and that value
would flow straight through ModelTrainer._fit_final_and_validate into the
final refit — stacking with SMOTE-resampled training data under
imbalance_strategy="smote", or silently riding along even under
"cost_weighted" (where the fit()-level guard nulls it, but only after
Optuna wasted a whole search dimension on it). Now that class_weight is
gone from RandomForestModel.param_space() (see
tests/unit/modeling/models/test_model_contracts.py and
tests/unit/modeling/cv/test_imbalance_wiring.py for the search-space and
CV-integration coverage), this confirms the same guarantee holds
end-to-end through the real ModelTrainer.train_one() path, including the
final refit that actually gets logged/deployed."""

import numpy as np
import pandas as pd
import pytest


N_ROWS_PER_CLASS = 30
N_CLASSES = 4


@pytest.fixture
def classification_dataset():
    rng = np.random.default_rng(0)
    rows, labels = [], []
    for cls in range(N_CLASSES):
        for _ in range(N_ROWS_PER_CLASS):
            rows.append([cls * 10.0 + rng.normal(0, 0.5), rng.normal(0, 0.5)])
            labels.append(cls)
    X = pd.DataFrame(rows, columns=["f1", "f2"])
    y = pd.Series(labels)
    return X, y


def _make_trainer(tmp_path, monkeypatch, fast_modeling_config, experiment_name):
    from wildfire_susceptibility.modeling.training import ModelTrainer

    test_db = tmp_path / "mlflow_test.db"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.trainer.MLFLOW_TRACKING_URI",
        f"sqlite:///{test_db}",
    )

    config = fast_modeling_config
    config["base"]["figures_dir"] = tmp_path
    config["modeling"]["mlflow_experiment"] = experiment_name
    config["modeling"]["cv_strategy"] = "standard"
    config["modeling"]["cv_folds"] = 2
    config["modeling"]["use_smote"] = True
    config["modeling"]["smote_sampling_strategy"] = "auto"
    config["modeling"]["imbalance_strategy"] = "smote"
    return ModelTrainer(config)


def test_final_refit_best_params_never_carries_class_weight(
    tmp_path, monkeypatch, fast_modeling_config, classification_dataset,
):
    trainer = _make_trainer(tmp_path, monkeypatch, fast_modeling_config, "test-final-refit-class-weight")
    X, y = classification_dataset

    result = trainer.train_one(
        season="test_season",
        model_name="random_forest",
        X_train=X, y_train=y, X_val=X, y_val=y,
        groups_train=None,
        ref_path=None,
        run_post_training_evaluation=False,
    )

    assert "class_weight" not in result["best_params"], (
        "class_weight leaked into Optuna's best_params — it should no "
        "longer be a sampled dimension for random_forest at all."
    )

    # The final refit's class_weight must be sklearn's own default (None),
    # never an Optuna-selected value smuggled through best_params.
    final_model = result["model"]
    assert final_model.model.class_weight is None
