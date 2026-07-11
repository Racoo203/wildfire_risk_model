"""Shared protocol every model wrapper implements, so train.py can treat
RF/SVM/XGBoost/NN identically."""

from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class BaseWildfireModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseWildfireModel": ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def param_space(self, trial) -> dict:
        """Return an Optuna search space dict for this model's hyperparameters."""
        ...

    def needs_scaling(self) -> bool:
        """Whether dataset_prep should feed this model a StandardScaler'd copy."""
        ...