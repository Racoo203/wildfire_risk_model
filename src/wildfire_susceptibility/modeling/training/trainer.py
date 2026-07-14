# wildfire_susceptibility/modeling/training/trainer.py
"""ModelTrainer: coordinates dataset prep, hyperparameter search, final
refit, mlflow logging, and post-training evaluation for one (season,
model) pair. Delegates every actual mechanism to a focused collaborator
class rather than implementing it inline:

    - CVStrategy (cv/)         : how folds are built and how a fold is
                                  fit/scored (standard / spatial /
                                  stratified_spatial_block)
    - HyperparamSearch         : search-time subsampling, fold building on
                                  the subsample, and the Optuna study itself
    - SMOTEResampler           : in-fold / final-refit class balancing
    - DatasetPrep              : scaling, spatial block assignment
    - PostTrainingEvaluator    : metrics, figures, susceptibility raster

trainer.py itself should only ever be orchestration glue.
"""

from typing import Dict, Optional, Tuple
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
import mlflow
from sklearn.metrics import roc_auc_score, f1_score

from ...core.registry import MODELS
from .. import models  # noqa: F401 — registers model wrappers (two dots: training/ -> modeling/)
from ..dataset_prep import DatasetPrep
from .resampling import SMOTEResampler
from ..cv import get_cv_strategy, requires_spatial_groups
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

        cv_strategy_name = config["modeling"].get("cv_strategy", "both")
        primary_strategy_name = "spatial" if cv_strategy_name == "both" else cv_strategy_name

        smote_during_search = config["modeling"].get("smote_during_search", False)
        search_target_size = config["modeling"].get("search_resample_target_size")
        no_smote_config = {**config, "modeling": {**config["modeling"], "use_smote": False}}

        # stratified_spatial_block always resamples in-fold by design; other
        # strategies only resample during search if smote_during_search is
        # explicitly enabled. Either way, if search_resample_target_size is
        # set, search resampling targets a FIXED TOTAL split evenly across
        # classes rather than majority-relative auto/dict targets — this is
        # the knob that controls "how big/balanced is the fold the model
        # actually gets fit on during HPO", independent of the final refit.
        if primary_strategy_name == "stratified_spatial_block" or smote_during_search:
            search_resampler = SMOTEResampler(config, target_size=search_target_size)
        else:
            search_resampler = SMOTEResampler(no_smote_config)

        # Final refit is untouched by target_size — it keeps using whatever
        # smote_sampling_strategy/auto behavior was configured for the
        # full-scale training set.
        self.final_resampler = SMOTEResampler(config)

        self.cv_strategy = get_cv_strategy(primary_strategy_name, config, search_resampler)
        self.search = HyperparamSearch(config, self.cv_strategy)
        self.evaluator = PostTrainingEvaluator(config)
        self.dataset_prep = DatasetPrep(config)

    def _validate_cv_config(self) -> None:
        cv_strategy = self.config["modeling"].get("cv_strategy", "both")
        if cv_strategy in ("spatial", "stratified_spatial_block", "both"):
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

    def train_one(
        self,
        season: str,
        model_name: str,
        X_train, y_train, X_val, y_val,
        groups_train=None,
        progress_callback=None,
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
        search_uses_groups = requires_spatial_groups(self.cv_strategy.name)
        if search_uses_groups and groups_train is None:
            raise ValueError(
                f"[{season}][{model_name}] cv_strategy='{cv_strategy}' requires groups_train "
                f"(spatial blocks), but None was passed. Call DatasetPrep.assign_spatial_blocks() "
                f"first, or set cv_strategy='standard' if spatial CV isn't needed here."
            )

        X_tr, X_va = self._scale_features(model_cls, X_train, X_val)

        # --- Hyperparameter search: subsampling, fold building, and the
        # Optuna study itself all live inside HyperparamSearch now. ---
        search_result = self.search.run_search_and_report(
            model_cls, model_name, season, X_tr, y_train, groups_train, progress_callback,
        )
        best_params, best_value, study = (
            search_result["best_params"], search_result["best_value"], search_result["study"]
        )

        logger.info(
            f"[{season}][{model_name}] train shape={X_tr.shape} | "
            f"search shape={search_result['X_search'].shape} ({self.cv_strategy.name} CV) | "
            f"class counts={y_train.value_counts().to_dict()}"
        )

        with mlflow.start_run(run_name=f"{season}_{model_name}"):
            self._log_run_setup(season, model_name, best_params, best_value, search_uses_groups)

            cv_auc_spatial, cv_auc_standard, cv_auc_spatial_folds = None, None, None
            if search_uses_groups:
                cv_auc_spatial = best_value
            else:
                cv_auc_standard = best_value

            if cv_strategy == "both":
                diagnostic_strategy_name = "standard" if search_uses_groups else "spatial"
                diagnostic_strategy = get_cv_strategy(
                    diagnostic_strategy_name, self.config, self.cv_strategy.resampler
                )
                diagnostic_value, diagnostic_folds = self._run_diagnostic(
                    diagnostic_strategy, model_cls, best_params, X_tr, y_train, groups_train,
                    season, model_name, best_value,
                )
                if search_uses_groups:
                    cv_auc_standard = diagnostic_value
                else:
                    cv_auc_spatial, cv_auc_spatial_folds = diagnostic_value, diagnostic_folds

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
    # Small private helpers — feature scaling, logging, diagnostics
    # -------------------------------------------------------------

    def _scale_features(self, model_cls, X_train, X_val) -> Tuple:
        needs_scaling = model_cls().needs_scaling()
        X_tr, X_va, _ = self.dataset_prep.scale_for_model_family(X_train, X_val, needs_scaling)
        return X_tr, X_va

    def _log_run_setup(self, season, model_name, best_params, best_value, search_uses_groups):
        mlflow.set_tags({
            "season": season,
            "model": model_name,
            "cv_strategy": self.config["modeling"].get("cv_strategy", "both"),
            "search_cv_strategy": self.cv_strategy.name,
            "search_metric": "spatial" if search_uses_groups else "standard",
            "use_smote": self.config["modeling"].get("use_smote", False),
        })
        mlflow.log_params(best_params)
        metric_key = "cv_auc_spatial" if search_uses_groups else "cv_auc_standard"
        mlflow.log_metric(metric_key, best_value)

    def _run_diagnostic(
        self, diagnostic_strategy, model_cls, best_params, X_tr, y_train, groups_train,
        season, model_name, primary_value,
    ) -> Tuple[float, Optional[list]]:
        """Post-hoc, non-selection diagnostic: score the winning params under
        the *other* CV strategy purely to report the optimism gap. Never
        influences which hyperparameters were chosen. Returns (mean_auc,
        per-fold scores) — per-fold scores are only retained when the
        diagnostic uses spatial groups (needed by stage_selection's
        Kruskal-Wallis test)."""
        groups_for_diag = groups_train if requires_spatial_groups(diagnostic_strategy.name) else None
        folds = diagnostic_strategy.make_folds(X_tr, y_train, groups_for_diag)
        context = f"[{season}][{model_name}] {diagnostic_strategy.name} CV (diagnostic)"

        fold_scores = [
            diagnostic_strategy.fit_and_score(
                model_cls, best_params, X_tr, y_train, train_idx, test_idx,
                context=f"{context} fold {i + 1}/{len(folds)}",
            )
            for i, (train_idx, test_idx) in enumerate(folds)
        ]
        auc = float(np.mean(fold_scores))
        gap = (
            primary_value - auc if requires_spatial_groups(diagnostic_strategy.name)
            else auc - primary_value
        )

        mlflow.log_metric(f"cv_auc_{diagnostic_strategy.name}", auc)
        mlflow.log_metric("cv_auc_optimism_gap", gap)
        logger.info(f"{context}: AUC={auc:.4f} (search primary={primary_value:.4f}, gap={gap:.4f})")

        retained_folds = fold_scores if requires_spatial_groups(diagnostic_strategy.name) else None
        return auc, retained_folds

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