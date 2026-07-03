"""Optuna hyperparameter search + mlflow logging, per (season, model)."""

from typing import Dict, Optional
import logging

import numpy as np
import optuna
import mlflow
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score

from ..core.registry import MODELS
from . import models  # noqa: F401 — import side effect registers model wrappers
from .dataset_prep import DatasetPrep

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Runs an Optuna study per (season, model_name), logs each trial and the
    final refit to mlflow, and returns the best-by-CV-AUC fitted model.
    """

    def __init__(self, config: dict):
        self.config = config
        mlflow.set_experiment(config["modeling"]["mlflow_experiment"])

    def train_all(
        self,
        season: str,
        X_train, y_train,
        X_val, y_val,
    ) -> Dict[str, dict]:
        """Train every model listed in config.modeling.models for one season."""
        results = {}
        for model_name in self.config["modeling"]["models"]:
            results[model_name] = self.train_one(season, model_name, X_train, y_train, X_val, y_val)
        return results

    def train_one(
        self,
        season: str,
        model_name: str,
        X_train, y_train,
        X_val, y_val,
    ) -> dict:
        if model_name not in MODELS:
            raise ValueError(f"Model '{model_name}' not found in MODELS registry: {list(MODELS)}")

        model_cls = MODELS[model_name]
        prep = DatasetPrep(self.config)

        probe = model_cls()
        needs_scaling = probe.needs_scaling()
        X_tr, X_va, _ = prep.scale_for_model_family(X_train, X_val, needs_scaling)

        n_trials = self.config["modeling"]["optuna_n_trials"]
        cv_folds = self.config["modeling"]["cv_folds"]

        study = optuna.create_study(
            direction="maximize",
            study_name=f"{season}_{model_name}",
            load_if_exists=True,
        )

        def objective(trial):
            params = model_cls().param_space(trial)
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            aucs = []
            for train_idx, test_idx in skf.split(X_tr, y_train):
                m = model_cls(**params)
                m.fit(X_tr.iloc[train_idx].values, y_train.iloc[train_idx].values)
                proba = m.predict_proba(X_tr.iloc[test_idx].values)
                y_fold = y_train.iloc[test_idx].values
                try:
                    auc = roc_auc_score(y_fold, proba, multi_class="ovr")
                except ValueError:
                    auc = 0.0
                aucs.append(auc)
            return float(np.mean(aucs))

        with mlflow.start_run(run_name=f"{season}_{model_name}"):
            mlflow.set_tags({"season": season, "model": model_name})
            study.optimize(objective, n_trials=n_trials)

            best_params = study.best_params
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_auc", study.best_value)

            final_model = model_cls(**best_params)
            final_model.fit(X_tr.values, y_train.values)

            val_proba = final_model.predict_proba(X_va.values)
            val_auc = roc_auc_score(y_val, val_proba, multi_class="ovr")
            val_f1 = f1_score(y_val, np.argmax(val_proba, axis=1), average="macro")

            mlflow.log_metric("val_auc", val_auc)
            mlflow.log_metric("val_f1", val_f1)

            logger.info(
                f"[{season}][{model_name}] cv_auc={study.best_value:.4f} "
                f"val_auc={val_auc:.4f} val_f1={val_f1:.4f}"
            )

        return {
            "model": final_model,
            "best_params": best_params,
            "cv_auc": study.best_value,
            "val_auc": val_auc,
            "val_f1": val_f1,
        }