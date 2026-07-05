# wildfire_susceptibility/pipeline/train.py — full updated file

from typing import Callable, Dict, Optional, Tuple
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import geopandas as gpd
import optuna
import mlflow
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, GroupKFold, train_test_split
from sklearn.metrics import roc_auc_score, f1_score

from ..core.registry import MODELS
from ..modeling import models  # noqa: F401 — registers wrappers
from ..modeling.dataset_prep import DatasetPrep
from ..modeling.evaluate import evaluate_on_test, generate_susceptibility_raster

logger = logging.getLogger(__name__)

OPTUNA_STORAGE = "sqlite:///data/silver/dbs/optuna_studies.db"
FINISHED_STATES = {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}


class ModelTrainer:
    def __init__(self, config: dict):
        self.config = config
        mlflow.set_experiment(config["modeling"]["mlflow_experiment"])

    # -----------------------------------------------------------------
    # Public API — single model / single season
    # -----------------------------------------------------------------

    def train_all(
        self,
        season: str,
        X_train, y_train,
        X_val, y_val,
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
        X_train, y_train,
        X_val, y_val,
        groups_train=None,
        progress_callback: Optional[Callable[[str, int, float], None]] = None,
        ref_path: Optional[Path] = None,
        fire_test_gdf: Optional[gpd.GeoDataFrame] = None,
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
    ) -> dict:
        """
        ref_path / fire_test_gdf / x_coords / y_coords are optional — when
        given, train_one runs the full post-training evaluation (SHAP,
        susceptibility raster, time-forward validation) automatically.
        When omitted (e.g. quick smoke tests), it just skips those steps.
        """
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
                "use_smote": self.config["modeling"].get("use_smote", False),
            })
            mlflow.log_params(best_params)
            mlflow.log_metric("cv_auc_standard", best_value)

            cv_auc_spatial = self._spatial_cv_check(
                model_cls, best_params, X_tr, y_train, make_folds, season, model_name, best_value,
            )

            final_model, val_auc, val_f1 = self._fit_final_and_validate(
                model_cls, best_params, X_tr, y_train, X_va, y_val, season, model_name,
            )
            mlflow.log_metric("val_auc", val_auc)
            mlflow.log_metric("val_f1", val_f1)
            self._log_model_artifact(model_name, final_model)

            eval_results = self._maybe_evaluate(
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

    # -----------------------------------------------------------------
    # Setup helpers
    # -----------------------------------------------------------------

    def _prepare_features(self, model_cls, X_train, X_val):
        prep = DatasetPrep(self.config)
        needs_scaling = model_cls().needs_scaling()
        X_tr, X_va, _ = prep.scale_for_model_family(X_train, X_val, needs_scaling)
        return X_tr, X_va, needs_scaling

    def _subsample_for_search(self, X_tr, y_train, season, model_name):
        n = self.config["modeling"].get("optuna_search_subsample")
        if not n or len(X_tr) <= n:
            return X_tr, y_train

        X_search, _, y_search, _ = train_test_split(
            X_tr, y_train, train_size=n, stratify=y_train, random_state=42
        )
        logger.info(f"[{season}][{model_name}] search subsample: {len(X_search):,}/{len(X_tr):,} rows")
        return X_search, y_search

    def _get_or_create_study(self, season: str, model_name: str) -> optuna.Study:
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
                f"[{season}][{model_name}] removing {len(incomplete)} incomplete trial(s) "
                f"from a previous run: {[t.number for t in incomplete]}"
            )
            for t in incomplete:
                try:
                    study.tell(t.number, state=optuna.trial.TrialState.FAIL)
                except Exception as e:
                    logger.warning(f"Could not mark trial {t.number} as FAILED: {e}")

        return study

    def _fold_factory(self, X, y, groups_train):
        cv_folds = self.config["modeling"]["cv_folds"]

        def make_folds(strategy: str):
            if strategy == "spatial":
                if groups_train is None:
                    raise ValueError("Spatial CV requested but no spatial groups provided.")
                return list(GroupKFold(n_splits=cv_folds).split(X, y, groups=groups_train))
            return list(StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42).split(X, y))

        return make_folds

    # -----------------------------------------------------------------
    # SMOTE
    # -----------------------------------------------------------------

    def _maybe_resample(self, X: np.ndarray, y: np.ndarray, season: str, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        if not self.config["modeling"].get("use_smote", False):
            return X, y

        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            logger.warning(
                f"[{season}][{model_name}] use_smote=True but imbalanced-learn is not "
                f"installed (pip install -e '.[modeling]'); skipping SMOTE for this call."
            )
            return X, y

        classes, counts = np.unique(y, return_counts=True)
        min_class_count = int(counts.min())

        if min_class_count <= 1:
            logger.warning(
                f"[{season}][{model_name}] smallest class has {min_class_count} sample(s) "
                f"in this fold; skipping SMOTE (cannot form neighbors)."
            )
            return X, y

        k_neighbors = min(self.config["modeling"].get("smote_k_neighbors", 5), min_class_count - 1)
        smote = SMOTE(k_neighbors=k_neighbors, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        logger.debug(
            f"[{season}][{model_name}] SMOTE: {dict(zip(classes, counts))} -> "
            f"{dict(zip(*np.unique(y_res, return_counts=True)))} (k={k_neighbors})"
        )
        return X_res, y_res

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
                auc = self._fit_and_score(
                    model_cls, params, X_search, y_search, train_idx, test_idx, season, model_name,
                )
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

    def _fit_and_score(self, model_cls, params, X, y, train_idx, test_idx, season, model_name) -> float:
        X_tr = X.iloc[train_idx].values
        y_tr = y.iloc[train_idx].values
        X_tr, y_tr = self._maybe_resample(X_tr, y_tr, season, model_name)

        m = model_cls(**params)
        m.fit(X_tr, y_tr)
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
            self._fit_and_score(model_cls, best_params, X_tr, y_train, train_idx, test_idx, season, model_name)
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

    def _fit_final_and_validate(self, model_cls, best_params, X_tr, y_train, X_va, y_val, season, model_name):
        X_fit, y_fit = self._maybe_resample(X_tr.values, y_train.values, season, model_name)

        final_model = model_cls(**best_params)
        final_model.fit(X_fit, y_fit)

        val_proba = final_model.predict_proba(X_va.values)
        val_auc = roc_auc_score(y_val, val_proba, multi_class="ovr")
        val_f1 = f1_score(y_val, np.argmax(val_proba, axis=1), average="macro")
        return final_model, val_auc, val_f1

    # -----------------------------------------------------------------
    # Post-training evaluation (SHAP, susceptibility raster, time-forward)
    # -----------------------------------------------------------------

    def _maybe_evaluate(
        self, final_model, X_va, y_val, season, model_name,
        ref_path, fire_test_gdf, x_coords, y_coords,
    ) -> dict:
        defaults = {"shap_path": None, "time_forward_validation": None, "susceptibility_map_path": None}
        if ref_path is None:
            logger.info(f"[{season}][{model_name}] No ref_path supplied; skipping post-training evaluation.")
            return defaults

        eval_results = evaluate_on_test(
            final_model, X_va.values, y_val.values, list(X_va.columns),
            season, model_name, self.config,
            fire_test_gdf=fire_test_gdf, x_coords=x_coords, y_coords=y_coords,
        )

        map_path = None
        if x_coords is not None and y_coords is not None:
            map_path = generate_susceptibility_raster(
                final_model, X_va.values, x_coords, y_coords, ref_path, season, model_name, self.config,
            )
        eval_results["susceptibility_map_path"] = map_path
        return eval_results

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


# ----------------------------------------------------------------------
# Orchestration entry point
# ----------------------------------------------------------------------

def run_training_pipeline(
    config: dict,
    dataset_paths: Dict[str, Dict[str, object]],
    ref_path: Path,
    progress_callback: Optional[Callable[[str, str, int, float], None]] = None,
) -> Dict[str, Dict[str, dict]]:
    """
    dataset_paths: {season: {"train": path, "test": path,
                              "fire_train": gdf, "fire_test": gdf}},
    as returned by WildfirePreprocessor.run_full_pipeline().

    Runs feature prep, hyperparameter search, final refit, and the full
    post-training evaluation (SHAP, susceptibility raster, time-forward
    validation) for every active season and every configured model —
    designed to be started once and left running unattended.
    """
    prep = DatasetPrep(config)
    climate_vars = tuple(config["data_sources"]["haduk"]["sources"])
    trainer = ModelTrainer(config)
    block_size = config["modeling"].get("spatial_block_size_m", 5000.0)

    all_results: Dict[str, Dict[str, dict]] = {}

    for season, paths in dataset_paths.items():
        raw = prep.load_train_test({"train": paths["train"], "test": paths["test"]})
        df_train = prep.prepare_train(raw["train"], season, ref_path, climate_vars)
        df_test = prep.prepare_test(raw["test"], season, climate_vars)

        x_coords = df_test["_x"].to_numpy() if "_x" in df_test.columns else None
        y_coords = df_test["_y"].to_numpy() if "_y" in df_test.columns else None

        X_train = df_train.drop(columns=["label"])
        y_train = df_train["label"]
        X_test = df_test.drop(columns=["label"])
        y_test = df_test["label"]

        feature_cols = [c for c in X_train.columns if not c.startswith("_")]
        groups_train = (
            prep.assign_spatial_blocks(X_train, block_size_m=block_size)
            if config["modeling"].get("cv_strategy") in ("spatial", "both")
            else None
        )

        fire_test_gdf = paths.get("fire_test")

        season_results = {}
        for model_name in config["modeling"]["models"]:
            cb = (lambda t, v, _s=season, _m=model_name: progress_callback(_s, _m, t, v)) if progress_callback else None
            season_results[model_name] = trainer.train_one(
                season, model_name,
                X_train[feature_cols], y_train,
                X_test[feature_cols], y_test,
                groups_train=groups_train,
                progress_callback=cb,
                ref_path=ref_path,
                fire_test_gdf=fire_test_gdf,
                x_coords=x_coords,
                y_coords=y_coords,
            )

        all_results[season] = season_results
        logger.info(f"[{season}] all models trained and evaluated.")

    return all_results


def results_to_dataframe(all_results: Dict[str, Dict[str, dict]]) -> pd.DataFrame:
    """Flatten run_training_pipeline()'s nested result dict into one comparison table."""
    rows = []
    for season, models in all_results.items():
        for model_name, r in models.items():
            tf = r.get("time_forward_validation") or {}
            rows.append({
                "season": season,
                "model": model_name,
                "cv_auc_standard": r["cv_auc_standard"],
                "cv_auc_spatial": r["cv_auc_spatial"],
                "val_auc": r["val_auc"],
                "val_f1": r["val_f1"],
                "tf_pct_medium_plus": tf.get("pct_medium_plus"),
                "susceptibility_map": r.get("susceptibility_map_path"),
            })
    return pd.DataFrame(rows).sort_values(["season", "cv_auc_standard"], ascending=[True, False])