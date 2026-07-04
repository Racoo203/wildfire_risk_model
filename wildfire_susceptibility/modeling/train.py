# modeling/train.py

from typing import Dict, Optional, Callable
import logging

import numpy as np
import optuna
import mlflow
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score

from ..core.registry import MODELS
from . import models  # noqa: F401
from .dataset_prep import DatasetPrep

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        mlflow.set_experiment(config["modeling"]["mlflow_experiment"])

    def train_all(
        self,
        season: str,
        X_train, y_train,
        X_val, y_val,
        groups_train=None,
        progress_callback: Optional[Callable[[str, int, float], None]] = None,
    ) -> Dict[str, dict]:
        """Train every model listed in config.modeling.models for one season.

        progress_callback(model_name, trial_number, trial_value) fires after
        every completed Optuna trial, letting the caller (e.g. Streamlit)
        render live progress instead of blocking silently until the whole
        study finishes.
        """
        results = {}
        for model_name in self.config["modeling"]["models"]:
            results[model_name] = self.train_one(
                season, model_name, X_train, y_train, X_val, y_val, groups_train,
                progress_callback=progress_callback,
            )
        return results

    def train_one(
        self,
        season: str,
        model_name: str,
        X_train, y_train,
        X_val, y_val,
        groups_train=None,
        progress_callback: Optional[Callable[[str, int, float], None]] = None,
    ) -> dict:
        if model_name not in MODELS:
            raise ValueError(f"Model '{model_name}' not found in MODELS registry: {list(MODELS)}")

        model_cls = MODELS[model_name]
        prep = DatasetPrep(self.config)

        probe = model_cls()
        needs_scaling = probe.needs_scaling()
        X_tr, X_va, _ = prep.scale_for_model_family(X_train, X_val, needs_scaling)

        n_trials = self.config["modeling"]["optuna_n_trials"]
        cv_strategy = self.config["modeling"].get("cv_strategy", "both")
        cv_folds = self.config["modeling"]["cv_folds"]

        study = optuna.create_study(
            direction="maximize",
            study_name=f"{season}_{model_name}",
            load_if_exists=True,
        )

        def _make_folds(strategy):
            if strategy == "spatial":
                if groups_train is None:
                    raise ValueError("Spatial CV requested but no spatial groups provided.")
                gkf = GroupKFold(n_splits=cv_folds)
                return list(gkf.split(X_tr, y_train, groups=groups_train))
            else:
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                return list(skf.split(X_tr, y_train))

        def objective(trial, strategy="standard"):
            params = model_cls().param_space(trial)
            folds = _make_folds(strategy)
            aucs = []
            for train_idx, test_idx in folds:
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

        # Optuna callback — fires after every trial regardless of success/pruning
        def _optuna_progress(study, trial):
            logger.info(f"[{season}][{model_name}] trial {trial.number}: value={trial.value:.4f}")
            if progress_callback is not None:
                progress_callback(model_name, trial.number, trial.value)

        with mlflow.start_run(run_name=f"{season}_{model_name}"):
            mlflow.set_tags({"season": season, "model": model_name, "cv_strategy": cv_strategy})

            study.optimize(
                lambda t: objective(t, "standard"),
                n_trials=n_trials,
                callbacks=[_optuna_progress],
            )
            best_params = study.best_params
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_auc_standard", study.best_value)

            cv_auc_spatial = None
            if cv_strategy in ("spatial", "both") and groups_train is not None:
                spatial_folds = _make_folds("spatial")
                spatial_aucs = []
                for train_idx, test_idx in spatial_folds:
                    m = model_cls(**best_params)
                    m.fit(X_tr.iloc[train_idx].values, y_train.iloc[train_idx].values)
                    proba = m.predict_proba(X_tr.iloc[test_idx].values)
                    spatial_aucs.append(
                        roc_auc_score(y_train.iloc[test_idx].values, proba, multi_class="ovr")
                    )
                cv_auc_spatial = float(np.mean(spatial_aucs))
                mlflow.log_metric("cv_auc_spatial", cv_auc_spatial)

                gap = study.best_value - cv_auc_spatial
                mlflow.log_metric("cv_auc_optimism_gap", gap)
                logger.info(
                    f"[{season}][{model_name}] standard CV AUC={study.best_value:.4f} vs "
                    f"spatial CV AUC={cv_auc_spatial:.4f} (optimism gap={gap:.4f})"
                )

            final_model = model_cls(**best_params)
            final_model.fit(X_tr.values, y_train.values)
            val_proba = final_model.predict_proba(X_va.values)
            val_auc = roc_auc_score(y_val, val_proba, multi_class="ovr")
            val_f1 = f1_score(y_val, np.argmax(val_proba, axis=1), average="macro")
            mlflow.log_metric("val_auc", val_auc)
            mlflow.log_metric("val_f1", val_f1)

            # Persist the fitted model artifact so evaluate.py / the dashboard
            # can load it later without retraining in-process.
            self._log_model_artifact(model_name, final_model)

        return {
            "model": final_model, "best_params": best_params,
            "cv_auc_standard": study.best_value,
            "cv_auc_spatial": cv_auc_spatial,
            "val_auc": val_auc, "val_f1": val_f1,
            "study": study,
        }

    @staticmethod
    def _log_model_artifact(model_name: str, fitted_model) -> None:
        """Log the underlying fitted estimator via the appropriate mlflow flavor."""
        try:
            if model_name in ("random_forest", "svm"):
                mlflow.sklearn.log_model(fitted_model.model, artifact_path="model")
            elif model_name == "xgboost":
                mlflow.xgboost.log_model(fitted_model.model, artifact_path="model")
            elif model_name == "neural_net":
                mlflow.pytorch.log_model(fitted_model.model, artifact_path="model")
        except Exception as exc:
            logger.warning(f"Could not log model artifact for '{model_name}': {exc}")