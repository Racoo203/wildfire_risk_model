from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

from ...core.registry import MODELS

@MODELS.register("svm")
class SVMModel:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.model: CalibratedClassifierCV | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMModel":
        base = SVC(random_state=42, **self.params)
        self.model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def param_space(self, trial) -> dict:
        return {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["rbf", "linear"]),  # dropped poly — too slow, rarely wins here
        }

    def needs_scaling(self) -> bool:
        return True