"""CatBoost wrapper: native categorical-feature support (landuse_class needs
no one-hot encoding) and, under imbalance_strategy="native_balanced",
auto_class_weights="Balanced" instead of an external sample_weight.
"""
from catboost import CatBoostClassifier
import numpy as np

from ...core.registry import MODELS


@MODELS.register("catboost")
class CatBoostModel:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model: CatBoostClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "CatBoostModel":
        params = dict(self.params)
        # native_balanced is bound onto model_cls via functools.partial in
        # trainer.py (same mechanism as cat_features), not an Optuna-tuned
        # hyperparameter — pop it before it reaches CatBoostClassifier,
        # which has no such constructor kwarg.
        native_balanced = params.pop("native_balanced", False)

        ctor_kwargs = dict(
            random_state=42,
            verbose=False,
            thread_count=-1,
            loss_function="MultiClass",
            allow_writing_files=False,  # skip catboost_info/ training-log dir — unused here
            **params,
        )
        if native_balanced:
            # imbalance_strategy="native_balanced" (modeling/imbalance.py):
            # CatBoost's own undamped auto_class_weights="Balanced" scheme.
            # Empirically engaged the rare classes more than the external
            # cost_weighted sample_weight without the collapse seen from
            # the dampened SqrtBalanced or manual extra-weighted variants
            # (scripts/experiment_catboost_weighting_variants.py).
            # ImbalanceStrategy.sample_weight_for() never returns a weight
            # for 'native_balanced', so sample_weight is expected to be
            # None here — no double-correction risk to guard against.
            ctor_kwargs["auto_class_weights"] = "Balanced"

        self.model = CatBoostClassifier(**ctor_kwargs)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def param_space(self, trial) -> dict:
        # Range tightened 08/16/2026 alongside random_forest.py's param_space
        # (see that file's comment for the full overfitting diagnosis).
        # CatBoost's optimism gap was more moderate than RF's and its
        # deployed depth/learning_rate weren't at an extreme, but
        # l2_leaf_reg's old floor (1e-2) allowed near-zero regularization
        # trials, and the space had no min_data_in_leaf/random_strength —
        # CatBoost's two standard anti-overfitting knobs beyond
        # depth/l2_leaf_reg. depth ceiling lowered 10 -> 8 as a matching
        # precaution to random_forest.py's max_depth cut.
        return {
            "iterations": trial.suggest_int("iterations", 100, 800),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
        }

    def needs_scaling(self) -> bool:
        return False

    def native_categorical_support(self) -> bool:
        return True
