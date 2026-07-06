# wildfire_susceptibility/modeling/training/trainer.py
"""ModelTrainer: coordinates dataset prep, hyperparameter search, final
refit, mlflow logging, and post-training evaluation for one (season,
model) pair. Delegates every actual mechanism to a focused collaborator
class rather than implementing it inline."""

from typing import Callable, Dict, Optional, Tuple
from pathlib import Path
import logging

import numpy as np
import geopandas as gpd
import mlflow
from sklearn.metrics import roc_auc_score, f1_score

from ...core.registry import MODELS
from ... import modeling  # noqa: F401 — registers model wrappers
from ..dataset_prep import DatasetPrep
from .resampling import SMOTEResampler
from .cv import FoldStrategy
from .search import HyperparamSearch
from .evaluation import PostTrainingEvaluator

logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        mlflow.set_experiment(config["modeling"]["mlflow_experiment"])

        resampler = SMOTEResampler(config)
        self.fold_strategy = FoldStrategy(config, resampler)
        self.search = HyperparamSearch(config, self.fold_strategy)
        self.evaluator = PostTrainingEvaluator(config)
        self.dataset_prep = DatasetPrep(config)

    def train_all(
        self,
        season: str,
        X_train, y_train, X_val, y_val,
        groups_train=None,
        progress_callback: Optional[Callable[[str, int, float], None]] = None,
        **evaluate_kwargs,
    ) -> Dict[str, dict]:
        return {
            model_name: self.train_one(
                season, model_name, X_train, y_train, X_val, y_val,
                groups_train, progress_callback=progress_callback, **evaluate_kwargs,
            )
            for model_name in self.config["modeling"]["models"]
        }

    def train_one(
        self,
        season: str,
        model_name: str,
        X_train, y_train, X_val, y_val,
        groups_train=None,
        progress_callback: Optional[Callable[[str, int, float], None]] = None,
        ref_path: Optional[Path] = None,
        fire_test_gdf: Optional[gpd.GeoDataFrame] = None,
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
    ) -> dict:
        if model_name not in MODELS:
            raise ValueError(f"Model '{model_name}' not found in MODELS registry: {list(MODELS)}")
        model_cls = MODELS[model_name]

        X_tr, X_va = self._scale_features(model_cls, X_train, X_val)
        X_search, y_search = self._subsample_for_search(X_tr, y_train, season, model_name)

        logger.info(
            f"[{season}][{model_name}] train shape={X_tr.shape} | search shape={X_search.shape} | "
            f"class counts={y_train.value_counts().to_dict()}"
        )

        study = self.search.get_or_create_study(season, model_name)
        standard_folds = self.fold_strategy.make_folds(X_search, y_search, groups_train, "standard")
        self.search.run(study, model_cls, X_search, y_search, standard_folds, season, model_name, progress_callback)

        best_params, best_value = study.best_params, study.best_value

        with mlflow.start_run(run_name=f"{season}_{model_name}"):
            self._log_run_setup(season, model_name, best_params, best_value)

            cv_auc_spatial = self._spatial_cv_check(
                model_cls, best_params, X_tr, y_train, groups_train, season, model_name, best_value,
            )

            final_model, val_auc, val_f1 = self._fit_final_and_validate(
                model_cls, best_params, X_tr, y_train, X_va, y_val, season, model_name,
            )
            mlflow.log_metric("val_auc", val_auc)
            mlflow.log_metric("val_f1", val_f1)
            self._log_model_artifact(model_name, final_model)

            eval_results = self.evaluator.evaluate(
                final_model, X_va, y_val, season, model_name,
                ref_path, fire_test_gdf, x_coords, y_coords,
            )

        return {
            "model": final_model,
            "best_params": best_params,
            "cv_auc_standard": best_value,
            "cv_auc_spatial": cv_auc_spatial,
            "val_auc": val_auc,
            "val_f1": val_f1,
            "study": study,
            **eval_results,
        }

    # -------------------------------------------------------------
    # Small private helpers — feature scaling, subsampling, logging
    # -------------------------------------------------------------

    def _scale_features(self, model_cls, X_train, X_val) -> Tuple:
        needs_scaling = model_cls().needs_scaling()
        X_tr, X_va, _ = self.dataset_prep.scale_for_model_family(X_train, X_val, needs_scaling)
        return X_tr, X_va

    def _subsample_for_search(self, X_tr, y_train, season, model_name):
        from sklearn.model_selection import train_test_split

        n = self.config["modeling"].get("optuna_search_subsample")
        if not n or len(X_tr) <= n:
            return X_tr, y_train

        X_search, _, y_search, _ = train_test_split(
            X_tr, y_train, train_size=n, stratify=y_train, random_state=42
        )
        logger.info(f"[{season}][{model_name}] search subsample: {len(X_search):,}/{len(X_tr):,} rows")
        return X_search, y_search

    def _log_run_setup(self, season, model_name, best_params, best_value):
        mlflow.set_tags({
            "season": season,
            "model": model_name,
            "cv_strategy": self.config["modeling"].get("cv_strategy", "both"),
            "use_smote": self.config["modeling"].get("use_smote", False),
        })
        mlflow.log_params(best_params)
        mlflow.log_metric("cv_auc_standard", best_value)

    def _spatial_cv_check(self, model_cls, best_params, X_tr, y_train, groups_train, season, model_name, standard_auc):
        cv_strategy = self.config["modeling"].get("cv_strategy", "both")
        if cv_strategy not in ("spatial", "both"):
            return None

        spatial_folds = self.fold_strategy.make_folds(X_tr, y_train, groups_train, "spatial")
        context = f"[{season}][{model_name}] spatial CV"
        cv_auc_spatial = self.fold_strategy.mean_auc_across_folds(
            model_cls, best_params, X_tr, y_train, spatial_folds, context=context,
        )
        gap = standard_auc - cv_auc_spatial

        mlflow.log_metric("cv_auc_spatial", cv_auc_spatial)
        mlflow.log_metric("cv_auc_optimism_gap", gap)
        logger.info(
            f"[{season}][{model_name}] standard CV AUC={standard_auc:.4f} vs "
            f"spatial CV AUC={cv_auc_spatial:.4f} (optimism gap={gap:.4f})"
        )
        return cv_auc_spatial

    def _fit_final_and_validate(self, model_cls, best_params, X_tr, y_train, X_va, y_val, season, model_name):
        resampler = self.fold_strategy.resampler
        X_fit, y_fit = resampler.resample(X_tr.values, y_train.values, context=f"[{season}][{model_name}] final refit")

        final_model = model_cls(**best_params)
        final_model.fit(X_fit, y_fit)

        val_proba = final_model.predict_proba(X_va.values)
        val_auc = roc_auc_score(y_val, val_proba, multi_class="ovr")
        val_f1 = f1_score(y_val, np.argmax(val_proba, axis=1), average="macro")
        return final_model, val_auc, val_f1

    @staticmethod
    def _log_model_artifact(model_name: str, fitted_model) -> None:
        try:
            if model_name in ("random_forest", "svm"):
                mlflow.sklearn.log_model(fitted_model.model, artifact_path="model")
            elif model_name == "xgboost":
                mlflow.xgboost.log_model(fitted_model.model, artifact_path="model")
            elif model_name == "neural_net":
                mlflow.pytorch.log_model(fitted_model.model, artifact_path="model")
        except Exception as exc:
            logger.warning(f"Could not log model artifact for '{model_name}': {exc}")