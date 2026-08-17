from sklearn.ensemble import RandomForestClassifier
import numpy as np

from ...core.registry import MODELS


@MODELS.register("random_forest")
class RandomForestModel:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model: RandomForestClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> "RandomForestModel":
        params = dict(self.params)
        if sample_weight is not None:
            # An externally-computed sample_weight (cost_weighted imbalance
            # strategy) takes over class balancing — sklearn multiplies
            # class_weight-derived weights by sample_weight elementwise, so
            # leaving class_weight="balanced" here (Optuna's HPO choice, see
            # param_space below) would silently compound the two.
            params["class_weight"] = None
        self.model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            criterion="gini",
            **params
        )
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def param_space(self, trial) -> dict:
        # class_weight is deliberately NOT tunable here: imbalance handling
        # is a resolver-level config choice (modeling.imbalance_strategy),
        # not something Optuna should pick. Leaving it in the search space
        # meant a trial could sample class_weight="balanced" while
        # imbalance_strategy resolved to "smote" for this model, stacking
        # class weighting on top of SMOTE-resampled training data with
        # nothing preventing it (the fit()-level guard below only fires for
        # the "cost_weighted" sample_weight path).
        # Range tightened 08/16/2026: deployed models were consistently
        # landing at/near the old max_depth=25 ceiling and min_samples_leaf=1
        # floor (e.g. depth=23/leaf=2), with standard-CV AUC ~0.99 collapsing
        # to ~0.5-0.55 on spatial CV and true validation — a severe
        # overfitting signature the old range let Optuna reach even though
        # search is spatial-CV-scored. max_samples added as a new tunable
        # (bootstrap row-subsample fraction, default was unset ->
        # RandomForestClassifier's full-bootstrap default) since row
        # subsampling decorrelates trees more effectively than depth/leaf
        # constraints alone on spatially autocorrelated features.
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 10, 40),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "max_samples": trial.suggest_float("max_samples", 0.3, 0.8),
        }

    def needs_scaling(self) -> bool:
        return False

    def native_categorical_support(self) -> bool:
        return False