"""Importing this package registers all model wrappers into MODELS."""

from .random_forest import RandomForestModel
from .svm import SVMModel
from .xgboost import XGBoostModel
from .ann import NeuralNetModel
from .catboost_model import CatBoostModel
from .ordinal_logistic import OrdinalLogisticModel
from .base import BaseWildfireModel

__all__ = [
    "RandomForestModel", "SVMModel", "XGBoostModel", "NeuralNetModel",
    "CatBoostModel", "OrdinalLogisticModel", "BaseWildfireModel",
]