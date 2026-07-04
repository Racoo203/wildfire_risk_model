# modeling/train.py

from typing import Callable, Dict, Optional
import logging

import numpy as np
import optuna
import mlflow
from tqdm import tqdm
from sklearn.model_selection import (
    StratifiedKFold,
    GroupKFold,
    train_test_split,
)
from sklearn.metrics import roc_auc_score, f1_score

from ..core.registry import MODELS
from . import models  # noqa: F401
from .dataset_prep import DatasetPrep

logger = logging.getLogger(__name__)

OPTUNA_STORAGE = "sqlite:///data/silver/dbs/optuna_studies.db"
FINISHED_STATES = {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}


class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        mlflow.set_experiment(config["modeling"]["mlflow_experiment"])

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def train_all(
        self,
        season: str,
        X_train, y_train,
        X_val, y_val,
        groups_train=None,
        progress_callback: Optional[Callable[[str, int, float], None]] = None,
    ) -> Dict[str, dict]:
        """Train every model listed in config.modeling.models for one season."""
        return {
            model_name: self.train_one(
                season, model_name, X_train, y_train, X_val, y_val,
                groups_train, progress_callback=progress_callback,
            )
            for model_name in self.config["modeling"]["models"]
        }

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
        X_tr, X_va, needs_scaling = self._prepare_features(model_cls, X_train, X_val)
        X_search, y_search = self._subsample_for_search(X_tr, y_train, season, model_name)

        logger.info(
            f"[{season}][{model_name}] train shape={X_tr.shape} | "
            f"search shape={X_search.shape} | class counts={y_train.value_counts().to_dict()}"
        )

        study = self._get_or_create_study(season, model_name)
        make_folds = self._fold_factory(X_search, y_search, groups_train)

        self._run_search(
            study, model_cls, X_search, y_search, make_folds,
            season, model_name, progress_callback,
        )

        best_params, best_value = study.best_params, study.best_value

        with mlflow.start_run(run_name=f"{season}_{model_name}"):
            mlflow.set_tags({
                "season": season,
                "model": model_name,
                "cv_strategy": self.config["modeling"].get("cv_strategy", "both"),
            })
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_auc_standard", best_value)

            cv_auc_spatial = self._spatial_cv_check(
                model_cls, best_params, X_tr, y_train, make_folds, season, model_name, best_value,
            )

            final_model, val_auc, val_f1 = self._fit_final_and_validate(
                model_cls, best_params, X_tr, y_train, X_va, y_val,
            )
            mlflow.log_metric("val_auc", val_auc)
            mlflow.log_metric("val_f1", val_f1)
            self._log_model_artifact(model_name, final_model)

        return {
            "model": final_model,
            "best_params": best_params,
            "cv_auc_standard": best_value,
            "cv_auc_spatial": cv_auc_spatial,
            "val_auc": val_auc,
            "val_f1": val_f1,
            "study": study,
        }

    # -----------------------------------------------------------------
    # Setup helpers
    # -----------------------------------------------------------------

    def _prepare_features(self, model_cls, X_train, X_val):
        """Scale (if the model family needs it) and return train/val + scaling flag."""
        prep = DatasetPrep(self.config)
        needs_scaling = model_cls().needs_scaling()
        X_tr, X_va, _ = prep.scale_for_model_family(X_train, X_val, needs_scaling)
        return X_tr, X_va, needs_scaling

    def _subsample_for_search(self, X_tr, y_train, season, model_name):
        """
        Optionally shrink the dataset used during hyperparameter search only.
        The final model is always refit on the full X_tr/y_train — this only
        speeds up the Optuna trials, it does not affect reported metrics.
        """
        n = self.config["modeling"].get("optuna_search_subsample")
        if not n or len(X_tr) <= n:
            return X_tr, y_train

        X_search, _, y_search, _ = train_test_split(
            X_tr, y_train, train_size=n, stratify=y_train, random_state=42
        )
        logger.info(f"[{season}][{model_name}] search subsample: {len(X_search):,}/{len(X_tr):,} rows")
        return X_search, y_search

    def _get_or_create_study(self, season: str, model_name: str) -> optuna.Study:
        """
        Persistent (sqlite-backed) study so optuna-dashboard can attach live,
        and so a killed/restarted run resumes instead of starting over.
        Any RUNNING/FAIL trials left over from a previous killed process are
        marked FAIL so they don't silently count toward n_remaining.
        """
        storage = optuna.storages.RDBStorage(
            url=OPTUNA_STORAGE,
            engine_kwargs={"connect_args": {"timeout": 30}},
        )
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{season}_{model_name}",
            storage=storage,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        )

        incomplete = [t for t in study.trials if t.state not in FINISHED_STATES]
        if incomplete:
            logger.info(
                f"[{season}][{model_name}] clearing {len(incomplete)} incomplete trial(s) "
                f"from a previous run: {[t.number for t in incomplete]}"
            )
            for t in incomplete:
                storage.set_trial_state_values(t._trial_id, state=optuna.trial.TrialState.FAIL)

        return study

    def _fold_factory(self, X, y, groups_train):
        """Returns a function(strategy) -> list of (train_idx, test_idx) fold splits."""
        cv_folds = self.config["modeling"]["cv_folds"]

        def make_folds(strategy: str):
            if strategy == "spatial":
                if groups_train is None:
                    raise ValueError("Spatial CV requested but no spatial groups provided.")
                return list(GroupKFold(n_splits=cv_folds).split(X, y, groups=groups_train))
            return list(StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42).split(X, y))

        return make_folds

    # -----------------------------------------------------------------
    # Search phase
    # -----------------------------------------------------------------

    def _run_search(self, study, model_cls, X_search, y_search, make_folds, season, model_name, progress_callback):
        n_trials = self.config["modeling"]["optuna_n_trials"]
        n_finished = len([t for t in study.trials if t.state in FINISHED_STATES])
        n_remaining = max(0, n_trials - n_finished)

        logger.info(f"[{season}][{model_name}] finished={n_finished} target={n_trials} remaining={n_remaining}")

        if n_remaining == 0:
            logger.info(f"[{season}][{model_name}] target trial count already reached; skipping search")
            return

        objective = self._make_objective(model_cls, X_search, y_search, make_folds, season, model_name)

        with tqdm(total=n_remaining, desc=f"[{season}] {model_name}") as pbar:
            def _callback(study, trial):
                pbar.update(1)
                if progress_callback is not None:
                    progress_callback(model_name, trial.number, trial.value or 0.0)

            study.optimize(objective, n_trials=n_remaining, callbacks=[_callback])

    def _make_objective(self, model_cls, X_search, y_search, make_folds, season, model_name):
        cv_folds = self.config["modeling"]["cv_folds"]

        def objective(trial):
            params = model_cls().param_space(trial)
            fold_aucs = []

            for fold_idx, (train_idx, test_idx) in enumerate(make_folds("standard")):
                auc = self._fit_and_score(model_cls, params, X_search, y_search, train_idx, test_idx)
                fold_aucs.append(auc)

                logger.info(
                    f"[{season}][{model_name}] trial {trial.number} fold {fold_idx + 1}/{cv_folds} "
                    f"AUC={auc:.4f} | params={params}"
                )

                trial.report(float(np.mean(fold_aucs)), step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(fold_aucs))

        return objective

    @staticmethod
    def _fit_and_score(model_cls, params, X, y, train_idx, test_idx) -> float:
        m = model_cls(**params)
        m.fit(X.iloc[train_idx].values, y.iloc[train_idx].values)
        proba = m.predict_proba(X.iloc[test_idx].values)
        try:
            return roc_auc_score(y.iloc[test_idx].values, proba, multi_class="ovr")
        except ValueError:
            return 0.0

    # -----------------------------------------------------------------
    # Post-search: spatial CV diagnostic, final refit, logging
    # -----------------------------------------------------------------

    def _spatial_cv_check(self, model_cls, best_params, X_tr, y_train, make_folds, season, model_name, standard_auc):
        cv_strategy = self.config["modeling"].get("cv_strategy", "both")
        if cv_strategy not in ("spatial", "both"):
            return None

        spatial_aucs = [
            self._fit_and_score(model_cls, best_params, X_tr, y_train, train_idx, test_idx)
            for train_idx, test_idx in make_folds("spatial")
        ]
        cv_auc_spatial = float(np.mean(spatial_aucs))
        gap = standard_auc - cv_auc_spatial

        mlflow.log_metric("cv_auc_spatial", cv_auc_spatial)
        mlflow.log_metric("cv_auc_optimism_gap", gap)
        logger.info(
            f"[{season}][{model_name}] standard CV AUC={standard_auc:.4f} vs "
            f"spatial CV AUC={cv_auc_spatial:.4f} (optimism gap={gap:.4f})"
        )
        return cv_auc_spatial

    @staticmethod
    def _fit_final_and_validate(model_cls, best_params, X_tr, y_train, X_va, y_val):
        final_model = model_cls(**best_params)
        final_model.fit(X_tr.values, y_train.values)

        val_proba = final_model.predict_proba(X_va.values)
        val_auc = roc_auc_score(y_val, val_proba, multi_class="ovr")
        val_f1 = f1_score(y_val, np.argmax(val_proba, axis=1), average="macro")
        return final_model, val_auc, val_f1

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