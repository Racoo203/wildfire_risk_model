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
    - nested_cv.run_nested_cv  : genuinely nested outer/inner CV — the
                                  reported cv_* performance estimate
    - SMOTEResampler           : in-fold / final-refit class balancing
    - DatasetPrep              : scaling, spatial block assignment
    - PostTrainingEvaluator    : metrics, figures, susceptibility raster

trainer.py itself should only ever be orchestration glue.
"""

from typing import Dict, Optional, Tuple
from pathlib import Path
import functools
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
import mlflow
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score

from ...core.registry import MODELS
from .. import models  # noqa: F401 — registers model wrappers (two dots: training/ -> modeling/)
from ..categorical import cat_features_of, to_model_array
from ..dataset_prep import DatasetPrep
from ..resampling import SMOTEResampler
from ..imbalance import ImbalanceStrategy
from ..cv import get_cv_strategy, requires_spatial_groups
from .search import HyperparamSearch
from .nested_cv import run_nested_cv
from .evaluation import PostTrainingEvaluator
from .optimism_gap import compute_optimism_gap

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = "sqlite:///data/silver/dbs/mlflow.db"


class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        self._validate_cv_config()
        self._ensure_mlflow_backend()
        mlflow.set_experiment(config["modeling"]["mlflow_experiment"])

        cv_strategy_name = self.config["modeling"].get("cv_strategy", "stratified_spatial_block")
        primary_strategy_name = "stratified_spatial_block" if cv_strategy_name == "both" else cv_strategy_name

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
        # Also used by nested_cv.run_nested_cv to build each outer fold's
        # inner StandardKFoldCV instance — the inner loop reuses this same
        # search-time resampling policy (smote_during_search /
        # search_resample_target_size), just never the
        # stratified_spatial_block "always resample" special case, since
        # the inner strategy is never stratified_spatial_block.
        self.search_resampler = search_resampler

        # Final refit is untouched by target_size — it keeps using whatever
        # smote_sampling_strategy/auto behavior was configured for the
        # full-scale training set.
        self.final_resampler = SMOTEResampler(config, is_final = True)

        self.cv_strategy = get_cv_strategy(primary_strategy_name, config, search_resampler)
        self.search = HyperparamSearch(config, self.cv_strategy)
        self.evaluator = PostTrainingEvaluator(config)
        self.dataset_prep = DatasetPrep(config)
        self._imbalance = ImbalanceStrategy(config)

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

        X_train, categorical_encoding = self.dataset_prep.fit_categorical_encoding(X_train, model_cls)
        X_val = self.dataset_prep.apply_categorical_encoding(X_val, categorical_encoding)
        cat_feature_names = (
            categorical_encoding["columns"] if categorical_encoding["kind"] == "native" else []
        )
        X_tr, X_va, scaler, scaling_meta = self._scale_features(model_cls, X_train, X_val)

        if cat_feature_names:
            # landuse_class (or any future native-categorical column) must
            # reach this model as an actual categorical column, not a
            # one-hot/ordinal encoding — see dataset_prep.py's
            # fit_categorical_encoding/apply_categorical_encoding and
            # modeling/categorical.py. Binding cat_features onto model_cls
            # here (rather than passing it separately) means every
            # downstream model_cls(**params) call — HyperparamSearch's
            # Optuna trials, the diagnostic/primary full-metrics CV
            # passes, and the final refit below — picks it up automatically
            # with no signature changes required at any of those call sites.
            cat_positions = [X_tr.columns.get_loc(c) for c in cat_feature_names]
            model_cls = functools.partial(model_cls, cat_features=cat_positions)

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
            # best_value is the PRODUCTION search's own convergence number
            # (see _log_run_setup) — used only to pick best_params for the
            # deployed model below, never as the reported CV performance
            # estimate. That estimate now comes exclusively from genuinely
            # nested CV (run_nested_cv), since reusing the search's own
            # best-trial score for reporting is exactly the Varma & Simon
            # (2006) bias nested CV exists to remove — see branch
            # refactor-nested-cv-optuna-schratz-method.
            self._log_run_setup(season, model_name, best_params, best_value, search_uses_groups)

            cv_auc_spatial, cv_auc_standard, cv_auc_spatial_folds = None, None, None
            cv_f1_macro_standard, cv_f1_macro_spatial = None, None
            cv_pr_auc_macro_standard, cv_pr_auc_macro_spatial = None, None
            cv_qwk_standard, cv_qwk_spatial = None, None
            cv_optimism_gap = None

            # PRIMARY side: genuinely nested CV, mandatory regardless of
            # cv_strategy or log_full_cv_diagnostics — there is no more
            # non-nested fallback for this number.
            primary_mean, primary_folds = self._run_nested_cv(
                self.cv_strategy, model_cls, X_tr, y_train, groups_train,
                season, model_name, progress_callback,
            )
            if search_uses_groups:
                cv_auc_spatial = primary_mean["auc"]
                cv_f1_macro_spatial = primary_mean["f1_macro"]
                cv_pr_auc_macro_spatial = primary_mean["pr_auc_macro"]
                cv_qwk_spatial = primary_mean["qwk"]
                cv_auc_spatial_folds = [f["auc"] for f in primary_folds]
                self._log_fold_step_metrics("spatial", primary_folds)
            else:
                cv_auc_standard = primary_mean["auc"]
                cv_f1_macro_standard = primary_mean["f1_macro"]
                cv_pr_auc_macro_standard = primary_mean["pr_auc_macro"]
                cv_qwk_standard = primary_mean["qwk"]
                self._log_fold_step_metrics("standard", primary_folds)

            if cv_strategy == "both":
                # Under cv_strategy="both", __init__ always resolves the
                # primary strategy to stratified_spatial_block, so
                # search_uses_groups is always True here and the
                # diagnostic side is always "standard" — the branch below
                # is kept explicit (not hardcoded to "standard") so intent
                # stays legible if that primary-strategy mapping ever
                # changes.
                diagnostic_strategy_name = "standard" if search_uses_groups else "spatial"

                # DIAGNOSTIC (comparison-only) side: gated by
                # log_full_cv_diagnostics. The primary side's nested pass
                # above is unconditional and already answers "how good is
                # the configured strategy," so this flag now only controls
                # whether the *opposite* side also gets a genuinely nested
                # pass for the standard-vs-spatial optimism-gap comparison
                # — skipping it halves 'both' mode's nested-CV cost when
                # only the primary side's number is needed.
                if self.config["modeling"].get("log_full_cv_diagnostics", False):
                    diagnostic_strategy = get_cv_strategy(
                        diagnostic_strategy_name, self.config, self.search_resampler
                    )
                    diagnostic_mean, diagnostic_folds = self._run_nested_cv(
                        diagnostic_strategy, model_cls, X_tr, y_train, groups_train,
                        season, model_name, progress_callback,
                    )
                    self._log_fold_step_metrics(diagnostic_strategy_name, diagnostic_folds)

                    if search_uses_groups:
                        cv_auc_standard = diagnostic_mean["auc"]
                        cv_f1_macro_standard = diagnostic_mean["f1_macro"]
                        cv_pr_auc_macro_standard = diagnostic_mean["pr_auc_macro"]
                        cv_qwk_standard = diagnostic_mean["qwk"]
                        standard_metrics, spatial_metrics = diagnostic_mean, primary_mean
                    else:
                        cv_auc_spatial = diagnostic_mean["auc"]
                        cv_f1_macro_spatial = diagnostic_mean["f1_macro"]
                        cv_pr_auc_macro_spatial = diagnostic_mean["pr_auc_macro"]
                        cv_qwk_spatial = diagnostic_mean["qwk"]
                        standard_metrics, spatial_metrics = primary_mean, diagnostic_mean

                    cv_optimism_gap = self._log_optimism_gap(standard_metrics, spatial_metrics)
                else:
                    logger.info(
                        f"[{season}][{model_name}] log_full_cv_diagnostics=False: skipping "
                        f"{diagnostic_strategy_name} nested CV diagnostic side; "
                        f"cv_optimism_gap will be None"
                    )

            self._log_cv_metrics(
                cv_auc_standard, cv_auc_spatial,
                cv_f1_macro_standard, cv_f1_macro_spatial,
                cv_pr_auc_macro_standard, cv_pr_auc_macro_spatial,
                cv_qwk_standard, cv_qwk_spatial,
            )

            final_model, val_auc, val_f1, val_qwk = self._fit_final_and_validate(
                model_cls, best_params, X_tr, y_train, X_va, y_val, season, model_name,
            )
            mlflow.log_metric("val_auc", val_auc)
            mlflow.log_metric("val_f1", val_f1)
            mlflow.log_metric("val_qwk", val_qwk)
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
            "categorical_encoding": categorical_encoding,
            "scaler": scaler,
            "scaling_meta": scaling_meta,
            "cv_auc_standard": cv_auc_standard,
            "cv_auc_spatial": cv_auc_spatial,
            "cv_auc_spatial_folds": cv_auc_spatial_folds,
            "cv_f1_macro_standard": cv_f1_macro_standard,
            "cv_f1_macro_spatial": cv_f1_macro_spatial,
            "cv_pr_auc_macro_standard": cv_pr_auc_macro_standard,
            "cv_pr_auc_macro_spatial": cv_pr_auc_macro_spatial,
            "cv_qwk_standard": cv_qwk_standard,
            "cv_qwk_spatial": cv_qwk_spatial,
            "cv_optimism_gap": cv_optimism_gap,
            "val_auc": val_auc,
            "val_f1": val_f1,
            "val_qwk": val_qwk,
            "study": study,
            **eval_results,
        }

    # -------------------------------------------------------------
    # Small private helpers — feature scaling, logging, diagnostics
    # -------------------------------------------------------------

    def _scale_features(self, model_cls, X_train, X_val) -> Tuple:
        needs_scaling = model_cls().needs_scaling()
        X_tr, X_va, scaler = self.dataset_prep.scale_for_model_family(X_train, X_val, needs_scaling)
        # Columns captured post-categorical-encoding (X_train here is
        # already fit_categorical_encoding's output), matching exactly
        # what the scaler was fit on — apply_scaling at eval time needs
        # the same column set/order reproduced by apply_categorical_encoding.
        scaling_meta = {
            "needs_scaling": needs_scaling,
            "columns": list(X_train.columns) if needs_scaling else [],
        }
        return X_tr, X_va, scaler, scaling_meta

    def _log_run_setup(self, season, model_name, best_params, best_value, search_uses_groups):
        mlflow.set_tags({
            "season": season,
            "model": model_name,
            "cv_strategy": self.config["modeling"].get("cv_strategy", "both"),
            "search_cv_strategy": self.cv_strategy.name,
            "search_metric": "spatial" if search_uses_groups else "standard",
            "search_objective_metric": "pr_auc_macro",
            "use_smote": self.config["modeling"].get("use_smote", False),
            "imbalance_strategy": self._imbalance.resolve(model_name),
        })
        mlflow.log_params(best_params)

        side = "spatial" if search_uses_groups else "standard"
        # best_value is the PRODUCTION search's own directly-optimized
        # metric (PR-AUC-macro, see search.py's _make_objective) — purely
        # informational context on what that search converged to. It is
        # NOT the reported CV performance estimate: that number (logged
        # separately via _log_cv_metrics, from genuinely nested CV) can
        # legitimately differ, since best_value comes from folds that were
        # also used to pick best_params. Kept under its existing
        # cv_prauc_{side} key (not cv_auc_*/cv_pr_auc_macro_*) so it isn't
        # mistaken for the nested estimate by anything reading mlflow.
        mlflow.log_metric(f"cv_prauc_{side}", best_value)

    def _run_nested_cv(
        self, strategy, model_cls, X_tr, y_train, groups_train,
        season, model_name, progress_callback=None,
    ) -> Tuple[dict, list]:
        """Thin wrapper around nested_cv.run_nested_cv supplying this
        trainer's search_resampler/config. See that module for the actual
        outer/inner nesting mechanics. Returns (mean metrics dict,
        per-outer-fold metrics dicts) — same shape the old
        _run_full_metrics_pass returned."""
        return run_nested_cv(
            strategy, self.search_resampler, model_cls, model_name, season, self.config,
            X_tr, y_train, groups_train, progress_callback,
        )

    @staticmethod
    def _log_cv_metrics(
        cv_auc_standard, cv_auc_spatial,
        cv_f1_macro_standard, cv_f1_macro_spatial,
        cv_pr_auc_macro_standard, cv_pr_auc_macro_spatial,
        cv_qwk_standard, cv_qwk_spatial,
    ) -> None:
        """Single source of truth for logging the reported (genuinely
        nested) cv_auc_*/cv_f1_macro_*/cv_pr_auc_macro_*/cv_qwk_* mlflow
        metrics, for whichever side(s) were actually computed this run —
        a side whose auc is None (not run) is skipped entirely rather than
        logging partial/None metrics."""
        for side, (auc, f1, prauc, qwk) in (
            ("standard", (cv_auc_standard, cv_f1_macro_standard, cv_pr_auc_macro_standard, cv_qwk_standard)),
            ("spatial", (cv_auc_spatial, cv_f1_macro_spatial, cv_pr_auc_macro_spatial, cv_qwk_spatial)),
        ):
            if auc is None:
                continue
            mlflow.log_metric(f"cv_auc_{side}", auc)
            mlflow.log_metric(f"cv_f1_macro_{side}", f1)
            mlflow.log_metric(f"cv_pr_auc_macro_{side}", prauc)
            mlflow.log_metric(f"cv_qwk_{side}", qwk)

    @staticmethod
    def _log_fold_step_metrics(role: str, fold_list: list) -> None:
        """Per-outer-fold mlflow step metrics for one side's nested CV run
        — lets the dissertation figures show fold-level spread, not just
        the mean."""
        for i, fold_metrics in enumerate(fold_list):
            for metric in ("auc", "f1_macro", "pr_auc_macro", "qwk"):
                mlflow.log_metric(f"cv_{metric}_{role}_fold", fold_metrics[metric], step=i)

    @staticmethod
    def _log_optimism_gap(standard_metrics: dict, spatial_metrics: dict) -> dict:
        """Logs the standard-vs-spatial optimism gap (standard - spatial)
        for AUC, F1-macro, PR-AUC-macro, and QWK. Both `standard_metrics`
        and `spatial_metrics` are genuinely nested per-side mean metrics
        (nested_cv.run_nested_cv) — per-side aggregate values and
        per-fold breakdowns are logged separately via _log_cv_metrics /
        _log_fold_step_metrics, so this method only computes and logs the
        gap itself.

        Per-fold standard[i] and spatial[i] are NOT paired into a per-fold
        gap: fold i means something different under each strategy
        (row-stratified vs. block-stratified), so subtracting them
        index-wise would be statistically meaningless — this mirrors
        stage_selection.py's Kruskal-Wallis test, which treats each
        strategy's fold scores as independent samples, not matched pairs.
        Only the aggregate (mean-across-folds) gap is reported.
        """
        gap = compute_optimism_gap(standard_metrics, spatial_metrics)

        for metric, value in gap.items():
            mlflow.log_metric(f"cv_{metric}_optimism_gap", value)

        logger.info(
            f"[optimism gap] standard-spatial: AUC={gap['auc']:.4f} F1_macro={gap['f1_macro']:.4f} "
            f"PR_AUC_macro={gap['pr_auc_macro']:.4f} QWK={gap['qwk']:.4f}"
        )
        return gap

    def _fit_final_and_validate(self, model_cls, best_params, X_tr, y_train, X_va, y_val, season, model_name):
        cat_positions = cat_features_of(model_cls)
        X_fit, y_fit = self.final_resampler.resample(
            to_model_array(X_tr, cat_positions), y_train.values, context=f"[{season}][{model_name}] final refit",
            model_name=model_name,
        )

        sample_weight = self._imbalance.sample_weight_for(model_name, y_fit)
        final_model = model_cls(**best_params)
        final_model.fit(X_fit, y_fit, sample_weight=sample_weight)

        val_proba = final_model.predict_proba(to_model_array(X_va, cat_positions))
        val_pred = np.argmax(val_proba, axis=1)
        val_auc = roc_auc_score(y_val, val_proba, multi_class="ovr")
        val_f1 = f1_score(y_val, val_pred, average="macro")
        val_qwk = cohen_kappa_score(y_val, val_pred, weights="quadratic")
        return final_model, val_auc, val_f1, val_qwk

    @staticmethod
    def _log_model_artifact(model_name: str, fitted_model) -> None:
        try:
            if model_name in ("random_forest", "svm"):
                mlflow.sklearn.log_model(fitted_model.model, artifact_path="model")
            elif model_name == "xgboost":
                mlflow.xgboost.log_model(fitted_model.model, artifact_path="model")
            elif model_name == "neural_net":
                mlflow.pytorch.log_model(fitted_model.model, artifact_path="model")
            elif model_name == "catboost":
                mlflow.catboost.log_model(fitted_model.model, artifact_path="model")
            # "ordinal_lr" (mord.LogisticAT) has no branch here: it's not a
            # plain sklearn estimator (no mlflow.sklearn compatibility) and
            # mlflow has no native mord flavor, so it falls through with no
            # mlflow model artifact logged — same silent-no-op behavior this
            # if/elif already has for any other unmatched model_name. The
            # joblib artifact under models/<season>/<model_name>/<cfg_sig>/
            # (stage_train.py's _write_artifact) is unaffected and remains
            # the source of truth for reloading this model.
        except Exception as exc:
            logger.warning(f"Could not log model artifact for '{model_name}': {exc}")