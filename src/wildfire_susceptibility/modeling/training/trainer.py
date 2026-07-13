# wildfire_susceptibility/modeling/training/trainer.py
"""ModelTrainer: coordinates dataset prep, hyperparameter search, final
refit, mlflow logging, and post-training evaluation for one (season,
model) pair. Delegates every actual mechanism to a focused collaborator
class rather than implementing it inline."""

from typing import Callable, Dict, Optional, Tuple
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
import mlflow
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

from ...core.registry import MODELS
from .. import models  # noqa: F401 — registers model wrappers (two dots: training/ -> modeling/)
from ..dataset_prep import DatasetPrep
from .resampling import SMOTEResampler
from .cv import FoldStrategy
from .search import HyperparamSearch
from .evaluation import PostTrainingEvaluator

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "sqlite:///data/silver/dbs/mlflow.db"

class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self._validate_cv_config()
        self._ensure_mlflow_backend()
        mlflow.set_experiment(config["modeling"]["mlflow_experiment"])

        no_smote_config = {
            **config,
            "modeling": {**config["modeling"], "use_smote": False},
        }
        search_resampler = SMOTEResampler(no_smote_config)
        self.final_resampler = SMOTEResampler(config)

        self.fold_strategy = FoldStrategy(config, search_resampler)
        self.search = HyperparamSearch(config, self.fold_strategy)
        self.evaluator = PostTrainingEvaluator(config)
        self.dataset_prep = DatasetPrep(config)

    def _validate_cv_config(self) -> None:
        cv_strategy = self.config["modeling"].get("cv_strategy", "both")
        if cv_strategy in ("spatial", "both"):
            block_size = self.config["modeling"].get("spatial_block_size_m", 5000.0)
            if not block_size or block_size <= 0:
                raise ValueError(
                    f"cv_strategy='{cv_strategy}' requires a positive "
                    f"modeling.spatial_block_size_m, got {block_size!r}."
                )

    @staticmethod
    def _ensure_mlflow_backend() -> None:
        db_path = Path("data/silver/dbs/mlflow.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

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
        run_post_training_evaluation: bool = True,
    ) -> dict:
        if model_name not in MODELS:
            raise ValueError(f"Model '{model_name}' not found in MODELS registry: {list(MODELS)}")
        model_cls = MODELS[model_name]

        cv_strategy = self.config["modeling"].get("cv_strategy", "both")
        if cv_strategy in ("spatial", "both") and groups_train is None:
            raise ValueError(
                f"[{season}][{model_name}] cv_strategy='{cv_strategy}' requires groups_train "
                f"(spatial blocks), but None was passed. Call DatasetPrep.assign_spatial_blocks() "
                f"first, or set cv_strategy='standard' if spatial CV isn't needed here."
            )

        X_tr, X_va = self._scale_features(model_cls, X_train, X_val)

        study = self.search.get_or_create_study(season, model_name)
        search_metric_is_spatial = cv_strategy in ("spatial", "both")

        if search_metric_is_spatial:
            X_search, y_search, groups_search = self._subsample_for_search_spatial(
                X_tr, y_train, groups_train, season, model_name
            )
            search_folds = self.fold_strategy.make_folds(X_search, y_search, groups_search, "spatial")
        else:
            X_search, y_search = self._subsample_for_search(X_tr, y_train, season, model_name)
            search_folds = self.fold_strategy.make_folds(X_search, y_search, None, "standard")

        logger.info(
            f"[{season}][{model_name}] train shape={X_tr.shape} | search shape={X_search.shape} "
            f"({'spatial' if search_metric_is_spatial else 'standard'} CV) | "
            f"class counts={y_train.value_counts().to_dict()}"
        )

        self.search.run(study, model_cls, X_search, y_search, search_folds, season, model_name, progress_callback)
        best_params, best_value = study.best_params, study.best_value

        with mlflow.start_run(run_name=f"{season}_{model_name}"):
            self._log_run_setup(season, model_name, best_params, best_value, search_metric_is_spatial)

            if search_metric_is_spatial:
                cv_auc_spatial = best_value
                cv_auc_spatial_folds = None  # per-fold detail not retained from the winning Optuna trial itself
                cv_auc_standard = self._diagnostic_cv_check(
                    model_cls, best_params, X_tr, y_train, None, "standard", season, model_name, best_value,
                )
            else:
                cv_auc_standard = best_value
                cv_auc_spatial, cv_auc_spatial_folds = self._spatial_diagnostic_with_folds(
                    model_cls, best_params, X_tr, y_train, groups_train, season, model_name, best_value,
                ) if cv_strategy == "both" else (None, None)

            final_model, val_auc, val_f1 = self._fit_final_and_validate(
                model_cls, best_params, X_tr, y_train, X_va, y_val, season, model_name,
            )
            mlflow.log_metric("val_auc", val_auc)
            mlflow.log_metric("val_f1", val_f1)
            self._log_model_artifact(model_name, final_model)

            eval_results = {}
            if run_post_training_evaluation:
                eval_results = self.evaluator.evaluate(
                    final_model, X_va, y_val, season, model_name,
                    ref_path, fire_test_gdf, x_coords, y_coords,
                )

        return {
            "model": final_model,
            "best_params": best_params,
            "cv_auc_standard": cv_auc_standard,
            "cv_auc_spatial": cv_auc_spatial,
            "cv_auc_spatial_folds": cv_auc_spatial_folds,
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
        n = self.config["modeling"].get("optuna_search_subsample_by_model", {}).get(model_name) \
            or self.config["modeling"].get("optuna_search_subsample")
        if not n or len(X_tr) <= n:
            return X_tr, y_train

        X_search, _, y_search, _ = train_test_split(
            X_tr, y_train, train_size=n, stratify=y_train, random_state=42
        )
        logger.info(f"[{season}][{model_name}] search subsample: {len(X_search):,}/{len(X_tr):,} rows")
        return X_search, y_search

    def _subsample_for_search_spatial(self, X_tr, y_train, groups_train, season, model_name):
        """
        Subsample by whole spatial block, not individual row — a plain
        row-level subsample would sever the block structure GroupKFold
        depends on, silently invalidating spatial CV on the subsample.
        Greedily fills blocks (in random order) until the row budget is
        reached, so the resulting subsample still has enough distinct
        blocks for GroupKFold to split meaningfully.
        """
        n = self.config["modeling"].get("optuna_search_subsample_by_model", {}).get(model_name) \
            or self.config["modeling"].get("optuna_search_subsample")
        if not n or len(X_tr) <= n:
            return X_tr, y_train, groups_train

        groups_series = pd.Series(groups_train, index=X_tr.index) if not isinstance(groups_train, pd.Series) else groups_train
        block_sizes = groups_series.value_counts().sample(frac=1.0, random_state=42)
        cumulative = block_sizes.cumsum()
        selected_blocks = cumulative[cumulative <= n].index

        if len(selected_blocks) == 0:
            selected_blocks = block_sizes.index[:1]  # smallest block alone exceeds n; take it anyway

        mask = groups_series.isin(selected_blocks)
        logger.info(
            f"[{season}][{model_name}] spatial search subsample: "
            f"{int(mask.sum()):,}/{len(X_tr):,} rows across {len(selected_blocks)} block(s)"
        )
        return X_tr[mask.values], y_train[mask.values], groups_series[mask.values]

    def _log_run_setup(self, season, model_name, best_params, best_value, search_metric_is_spatial):
        mlflow.set_tags({
            "season": season,
            "model": model_name,
            "cv_strategy": self.config["modeling"].get("cv_strategy", "both"),
            "search_metric": "spatial" if search_metric_is_spatial else "standard",
            "use_smote": self.config["modeling"].get("use_smote", False),
        })
        mlflow.log_params(best_params)
        metric_key = "cv_auc_spatial" if search_metric_is_spatial else "cv_auc_standard"
        mlflow.log_metric(metric_key, best_value)

    def _diagnostic_cv_check(self, model_cls, best_params, X_tr, y_train, groups_train, strategy, season, model_name, primary_value):
        """
        Post-hoc, non-selection diagnostic: score the winning (spatially-
        selected) hyperparameters under standard K-fold, purely to report
        the optimism gap. Never influences which params were chosen.
        """
        folds = self.fold_strategy.make_folds(X_tr, y_train, groups_train, strategy)
        context = f"[{season}][{model_name}] standard CV (diagnostic)"
        auc = self.fold_strategy.mean_auc_across_folds(model_cls, best_params, X_tr, y_train, folds, context=context)

        gap = auc - primary_value
        mlflow.log_metric("cv_auc_standard", auc)
        mlflow.log_metric("cv_auc_optimism_gap", gap)
        logger.info(
            f"[{season}][{model_name}] spatial CV AUC={primary_value:.4f} (search objective) vs "
            f"standard CV AUC={auc:.4f} (diagnostic; gap={gap:.4f})"
        )
        return auc

    def _spatial_diagnostic_with_folds(self, model_cls, best_params, X_tr, y_train, groups_train, season, model_name, standard_auc):
        """
        Used only when cv_strategy='standard' but 'both' diagnostics were
        requested at a higher level in the original design — kept for
        completeness, but note: when cv_strategy='standard' outright (not
        'both'), groups_train is None and this is never called (see
        train_one's branch: cv_auc_spatial stays (None, None) unless
        cv_strategy == 'both').
        """
        spatial_folds = self.fold_strategy.make_folds(X_tr, y_train, groups_train, "spatial")
        context = f"[{season}][{model_name}] spatial CV (diagnostic)"

        fold_scores = []
        for i, (train_idx, test_idx) in enumerate(spatial_folds):
            fold_context = f"{context} fold {i + 1}/{len(spatial_folds)}"
            auc = self.fold_strategy.fit_and_score(model_cls, best_params, X_tr, y_train, train_idx, test_idx, context=fold_context)
            logger.info(f"{fold_context} AUC={auc:.4f}")
            fold_scores.append(auc)

        cv_auc_spatial = float(np.mean(fold_scores))
        gap = standard_auc - cv_auc_spatial

        mlflow.log_metric("cv_auc_spatial", cv_auc_spatial)
        mlflow.log_metric("cv_auc_optimism_gap", gap)
        logger.info(
            f"[{season}][{model_name}] standard CV AUC={standard_auc:.4f} (search objective) vs "
            f"spatial CV AUC={cv_auc_spatial:.4f} (diagnostic; gap={gap:.4f})"
        )
        return cv_auc_spatial, fold_scores

    def _fit_final_and_validate(self, model_cls, best_params, X_tr, y_train, X_va, y_val, season, model_name):
        X_fit, y_fit = self.final_resampler.resample(
            X_tr.values, y_train.values, context=f"[{season}][{model_name}] final refit"
        )

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