"""Importing this package registers all model wrappers into MODELS."""

from .random_forest import RandomForestModel
from .svm import SVMModel
from .xgboost import XGBoostModel
from .ann import NeuralNetModel
from .base import BaseWildfireModel

__all__ = ["RandomForestModel", "SVMModel", "XGBoostModel", "NeuralNetModel", "BaseWildfireModel"]