import xgboost as xgb
import numpy as np

from ...core.registry import MODELS

@MODELS.register("xgboost")
class XGBoostModel:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model: xgb.XGBClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        self.model = xgb.XGBClassifier(
            random_state=42, 
            eval_metric="mlogloss", 
            objective="multi:softprob",
            n_jobs=-1, 
            **self.params,
        )
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def param_space(self, trial) -> dict:
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    def needs_scaling(self) -> bool:
        return False