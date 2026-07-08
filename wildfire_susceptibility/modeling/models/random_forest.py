from sklearn.ensemble import RandomForestClassifier
import numpy as np

from ...core.registry import MODELS


@MODELS.register("random_forest")
class RandomForestModel:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model: RandomForestClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        self.model = RandomForestClassifier(
            random_state=42, 
            n_jobs=-1, 
            criterion="gini", 
            **self.params
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def param_space(self, trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400),
            "max_depth": trial.suggest_int("max_depth", 4, 25),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }

    def needs_scaling(self) -> bool:
        return False