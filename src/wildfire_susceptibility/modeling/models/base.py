"""Shared protocol every model wrapper implements, so train.py can treat
RF/SVM/XGBoost/NN identically."""

from typing import Optional, Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class BaseWildfireModel(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> "BaseWildfireModel":
        """sample_weight is None unless modeling.imbalance_strategy resolves
        to 'cost_weighted' for this model (see modeling/imbalance.py) —
        every wrapper must accept the kwarg even if it has no effect."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def param_space(self, trial) -> dict:
        """Return an Optuna search space dict for this model's hyperparameters."""
        ...

    def needs_scaling(self) -> bool:
        """Whether dataset_prep should feed this model a StandardScaler'd copy."""
        ...

    def native_categorical_support(self) -> bool:
        """Whether this model can consume a raw categorical column
        directly (via cat_features) rather than needing it one-hot
        encoded upstream by DatasetPrep.encode_categoricals_for_model_family."""
        ...