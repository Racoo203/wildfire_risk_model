# tests/unit/modeling/training/test_native_balanced_wiring.py
"""Regression coverage for imbalance_strategy='native_balanced' being
correctly wired through ModelTrainer.train_one(): both cat_features and
native_balanced get bound onto model_cls via functools.partial
(trainer.py) — the actual bug this file guards is that they must land in
a SINGLE partial call, not two nested ones. Nesting would make
cat_features invisible to every cat_features_of() caller
(modeling/categorical.py reads model_cls.keywords, which only exposes the
OUTERMOST partial layer's own kwargs, not an inner partial's), silently
breaking CatBoost's categorical handling for landuse_class the moment
native_balanced is also active for that model — with no exception raised
anywhere (see test_categorical_encoding.py's module docstring for what
this failure mode looks like when nothing catches it)."""

import numpy as np
import pandas as pd
import pytest


def _dataset_with_categorical(n_per_class=30, n_classes=4, seed=0):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for cls in range(n_classes):
        for _ in range(n_per_class):
            rows.append([
                cls * 10.0 + rng.normal(0, 0.5),
                rng.normal(0, 0.5),
                rng.choice([0, 1, 2, 5]),  # landuse_class — non-contiguous codes, mirrors real data
            ])
            labels.append(cls)
    X = pd.DataFrame(rows, columns=["elevation", "d_roads", "landuse_class"])
    y = pd.Series(labels)
    return X, y


def _make_trainer(tmp_path, monkeypatch, fast_modeling_config, experiment_name, model_name):
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
    config["modeling"]["imbalance_strategy_by_model"] = {model_name: "native_balanced"}
    return ModelTrainer(config)


def test_catboost_native_balanced_and_cat_features_both_survive_together(
    tmp_path, monkeypatch, fast_modeling_config,
):
    from wildfire_susceptibility.modeling.categorical import cat_features_of

    trainer = _make_trainer(
        tmp_path, monkeypatch, fast_modeling_config, "test-native-balanced-catboost", "catboost",
    )
    X, y = _dataset_with_categorical()

    result = trainer.train_one(
        season="test_season",
        model_name="catboost",
        X_train=X, y_train=y, X_val=X, y_val=y,
        groups_train=None,
        ref_path=None,
        run_post_training_evaluation=False,
    )

    assert result["categorical_encoding"]["kind"] == "native"

    final_model = result["model"]
    assert cat_features_of(final_model) != [], (
        "cat_features_of() returned no categorical positions for the final "
        "fitted CatBoost model — landuse_class encoding was lost. This is "
        "exactly the failure mode of binding native_balanced and "
        "cat_features via two nested functools.partial calls instead of "
        "one combined call."
    )
    assert final_model.model.get_params().get("auto_class_weights") == "Balanced"


def test_random_forest_native_balanced_end_to_end(tmp_path, monkeypatch, fast_modeling_config):
    """random_forest never binds cat_features (it one-hot-encodes
    landuse_class instead — see dataset_prep.py), so this is a simpler
    single-binding check: native_balanced alone must reach the final
    refit and produce a BalancedRandomForestClassifier."""
    from imblearn.ensemble import BalancedRandomForestClassifier

    trainer = _make_trainer(
        tmp_path, monkeypatch, fast_modeling_config, "test-native-balanced-rf", "random_forest",
    )
    X, y = _dataset_with_categorical()

    result = trainer.train_one(
        season="test_season",
        model_name="random_forest",
        X_train=X, y_train=y, X_val=X, y_val=y,
        groups_train=None,
        ref_path=None,
        run_post_training_evaluation=False,
    )

    assert isinstance(result["model"].model, BalancedRandomForestClassifier)
