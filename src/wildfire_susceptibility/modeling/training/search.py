"""Optuna hyperparameter search: HyperparamSearch builds/reuses a study per
(season, model, param-space signature, OBJECTIVE_VERSION) — the signature and
version together mean a genuine change to the search space or the objective's
scoring logic gets a fresh study automatically, rather than silently mixing
old and new trials. Runs on cv_strategy's folds, optionally subsampled
(optuna_search_subsample(_by_model)) for speed.
"""
from pathlib import Path
from typing import Callable, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import optuna
import hashlib
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from ..cv.base import CVStrategy
from ..cv.factory import requires_spatial_groups

from ..balance import log_class_balance

logger = logging.getLogger(__name__)

OPTUNA_STORAGE = "sqlite:///data/silver/dbs/optuna_studies.db"
FINISHED_STATES = {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED}

# Bump this whenever `_make_objective`'s scoring logic changes (which metric
# is returned/pruned-on, or how the underlying per-fold scorer computes it)
# — the study name's param-space hash only captures param_space() shape,
# not the objective function body, so an objective-only change would
# otherwise silently resume/append trials from an incompatible old study
# (see get_or_create_study). v2: objective switched from mean CV AUC to
# mean CV PR-AUC-macro (AUC/F1-macro/QWK logged as trial user_attrs
# instead). v3: forces fresh studies after the landuse_class
# categorical-encoding fix (fit/transform split in dataset_prep.py) — old
# v2 trials were scored under the broken encoding and are not comparable
# to post-fix trials. v4: forces fresh studies after the
# fix-spatial-cv-auc-missing-classes fix — fit_and_score_full's AUC/PR-AUC
# no longer silently return 0.0 for folds missing a class (now a real
# score restricted to observed classes, or NaN if genuinely degenerate),
# so old v3 trials for spatial/stratified_spatial_block searches (which
# hit this path whenever a fold's training split was missing a class) are
# not comparable to post-fix trials. v5: forces fresh studies for models
# switched to imbalance_strategy="native_balanced" (random_forest,
# catboost) — the imbalance mechanism is now bound onto model_cls and
# active DURING search itself (trainer.py), not just at final refit, so
# what counts as a good hyperparameter genuinely changed (e.g.
# BalancedRandomForestClassifier's much smaller effective per-tree
# sample); old v4 trials were scored under cost_weighted and are not
# comparable. Old v4 studies are orphaned, not deleted.
OBJECTIVE_VERSION = "objv5-nativebalanced"


class HyperparamSearch:
    """Owns everything needed to run an efficient Optuna search: subsampling
    the training data down to a searchable size, building CV folds on that
    subsample via the configured CVStrategy, and running/resuming the study.
    `trainer.py` should never need to touch subsampling or fold-building
    directly — call prepare_search_data() then run_search_and_report()."""

    def __init__(self, config: dict, cv_strategy: CVStrategy):
        self.config = config
        self.n_trials = config["modeling"]["optuna_n_trials"]
        self.cv_strategy = cv_strategy
        self._ensure_storage_dir()

    # -----------------------------------------------------------------
    # Search-data preparation (subsampling + fold construction)
    # -----------------------------------------------------------------

    def prepare_search_data(self, X_tr, y_train, groups_train, season, model_name):
        n = self._resolve_subsample_size(model_name, population_size=len(X_tr))

        if requires_spatial_groups(self.cv_strategy.name):
            X_search, y_search, groups_search = self._subsample_blocks(
                X_tr, y_train, groups_train, n, season, model_name
            )
            folds = self.cv_strategy.make_folds(X_search, y_search, groups_search)
        else:
            X_search, y_search = self._subsample_rows(X_tr, y_train, n, season, model_name)
            folds = self.cv_strategy.make_folds(X_search, y_search)

        return X_search, y_search, folds

    def _resolve_subsample_size(self, model_name: str, population_size: int) -> Optional[int]:
        """Resolves optuna_search_subsample(_by_model) to an absolute row
        count. A configured value <= 1.0 is interpreted as a fraction of
        `population_size`; > 1.0 is an absolute row count (unchanged
        legacy behavior)."""
        raw = (
            self.config["modeling"].get("optuna_search_subsample_by_model", {}).get(model_name)
            or self.config["modeling"].get("optuna_search_subsample")
        )
        if not raw:
            return None
        if raw <= 1.0:
            n = int(round(raw * population_size))
            logger.info(
                f"[{model_name}] resolved subsample fraction {raw} of {population_size:,} rows -> {n:,}"
            )
            return n
        return int(raw)

    def _subsample_rows(self, X_tr, y_train, n, season, model_name):
        if not n or len(X_tr) <= n:
            return X_tr, y_train
        X_search, _, y_search, _ = train_test_split(
            X_tr, y_train, train_size=n, stratify=y_train, random_state=42
        )
        log_class_balance(logger, f"[{season}][{model_name}] search subsample", y_search)
        logger.info(f"[{season}][{model_name}] search subsample: {len(X_search):,}/{len(X_tr):,} rows")
        return X_search, y_search

    def _subsample_blocks(self, X_tr, y_train, groups_train, n, season, model_name):
        if not n or len(X_tr) <= n:
            return X_tr, y_train, groups_train

        groups_series = (
            groups_train if isinstance(groups_train, pd.Series)
            else pd.Series(groups_train, index=X_tr.index)
        )
        y_series = pd.Series(y_train).reset_index(drop=True)
        groups_series = groups_series.reset_index(drop=True)

        # Always keep every block that contains the rarest class(es), so a
        # random block-fill can never accidentally drop the only evidence
        # of the minority label from the search subsample entirely.
        class_totals = y_series.value_counts()
        rarest_class = class_totals.idxmin()
        must_keep_blocks = set(groups_series[y_series == rarest_class].unique())

        block_sizes = groups_series.value_counts()
        must_keep_rows = int(block_sizes[list(must_keep_blocks)].sum()) if must_keep_blocks else 0

        remaining_budget = max(n - must_keep_rows, 0)
        candidate_blocks = block_sizes.drop(index=list(must_keep_blocks), errors="ignore").sample(
            frac=1.0, random_state=42
        )
        cumulative = candidate_blocks.cumsum()
        fill_blocks = cumulative[cumulative <= remaining_budget].index

        selected_blocks = set(must_keep_blocks) | set(fill_blocks)
        if not selected_blocks:
            selected_blocks = set(block_sizes.index[:1])

        mask = groups_series.isin(selected_blocks)
        logger.info(
            f"[{season}][{model_name}] block search subsample: "
            f"{int(mask.sum()):,}/{len(X_tr):,} rows across {len(selected_blocks)} block(s) "
            f"({len(must_keep_blocks)} force-kept for rarest class {rarest_class})"
        )
        log_class_balance(logger, f"[{season}][{model_name}] block search subsample", y_series[mask.values])
        return X_tr.iloc[np.where(mask.values)[0]], y_series[mask.values], groups_series[mask.values]

    # -----------------------------------------------------------------
    # Study lifecycle
    # -----------------------------------------------------------------

    @staticmethod
    def _param_space_signature(model_cls) -> str:
        """
        A short hash capturing the *shape* of param_space — distribution
        types and, for categoricals, their choices — so that changing a
        model's search space produces a new study name instead of colliding
        with an old study's incompatible persisted distributions.
        """
        probe_study = optuna.create_study()
        probe_trial = probe_study.ask()
        space = model_cls().param_space(probe_trial)

        parts = []
        for name in sorted(space.keys()):
            dist = probe_trial.distributions[name]
            parts.append(f"{name}:{dist!r}")
        signature = "|".join(parts)
        return hashlib.sha1(signature.encode()).hexdigest()[:8]

    @staticmethod
    def _ensure_storage_dir() -> None:
        db_path = Path(OPTUNA_STORAGE.replace("sqlite:///", ""))
        db_path.parent.mkdir(parents=True, exist_ok=True)

    def get_or_create_study(self, season: str, model_name: str) -> optuna.Study:
        from ...core.registry import MODELS

        model_cls = MODELS[model_name]
        sig = self._param_space_signature(model_cls)
        study_name = f"{season}_{model_name}_{sig}_{OBJECTIVE_VERSION}"

        storage = optuna.storages.RDBStorage(
            url=OPTUNA_STORAGE,
            engine_kwargs={"connect_args": {"timeout": 30}},
        )
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
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

    def run(
        self,
        study: optuna.Study,
        model_cls,
        X_search, y_search,
        folds,
        season: str,
        model_name: str,
        progress_callback: Optional[Callable[[str, int, float], None]] = None,
    ) -> None:
        n_finished = len([t for t in study.trials if t.state in FINISHED_STATES])
        n_remaining = max(0, self.n_trials - n_finished)

        logger.info(f"[{season}][{model_name}] finished={n_finished} target={self.n_trials} remaining={n_remaining}")
        if n_remaining == 0:
            logger.info(f"[{season}][{model_name}] target trial count already reached; skipping search")
            return

        objective = self._make_objective(model_cls, X_search, y_search, folds, season, model_name)

        with tqdm(total=n_remaining, desc=f"[{season}] {model_name}") as pbar:
            def _callback(study, trial):
                pbar.update(1)
                if progress_callback is not None:
                    progress_callback(model_name, trial.number, trial.value or 0.0)

            study.optimize(objective, n_trials=n_remaining, callbacks=[_callback])

    def run_search_and_report(
        self, model_cls, model_name: str, season: str,
        X_tr: pd.DataFrame, y_train: pd.Series, groups_train: Optional[pd.Series],
        progress_callback: Optional[Callable[[str, int, float], None]] = None,
    ) -> dict:
        """Single entry point trainer.py calls: subsample -> build folds ->
        get/resume the study -> run remaining trials -> return everything
        the caller needs (best params/value, the study, and the search
        subsample actually used, for shape logging)."""
        X_search, y_search, folds = self.prepare_search_data(X_tr, y_train, groups_train, season, model_name)

        study = self.get_or_create_study(season, model_name)
        self.run(study, model_cls, X_search, y_search, folds, season, model_name, progress_callback)

        return {
            "study": study,
            "best_params": study.best_params,
            "best_value": study.best_value,
            "X_search": X_search,
        }

    def _make_objective(self, model_cls, X_search, y_search, folds, season, model_name):
        """PR-AUC-macro (one-vs-rest-style, computed on one-hot y) is the
        primary objective — see fit_and_score_full. It's a better tuning
        target than plain ROC-AUC for this 4-class ordinal problem: ROC-AUC
        stays high under majority-class collapse in a way PR-AUC and
        F1-macro don't, which is exactly the AUC-fine/F1-poor pattern that
        motivated this change. F1-macro and QWK are computed every trial
        too (fit_and_score_full returns all four every fold, at no extra
        fit cost) and stashed as trial user_attrs for reporting/model
        comparison — they never influence pruning or the tuned value."""
        def _nanmean_metric(fold_metrics: list, metric: str, context: str) -> float:
            # fit_and_score_full's underlying scorer (score_multiclass_fold)
            # returns NaN, not 0.0, for a fold whose validation split shares
            # fewer than 2 classes with what the model can score — a real
            # scoring failure, not a real (if terrible) score. np.mean would
            # silently poison the whole trial's value to NaN the moment any
            # one fold hits this; np.nanmean excludes those folds instead,
            # matching trainer.py's _run_full_metrics_pass aggregation.
            values = [f[metric] for f in fold_metrics]
            if all(np.isnan(v) for v in values):
                logger.warning(
                    f"{context}: every fold's {metric} was NaN (genuinely degenerate — "
                    f"see per-fold warnings above); this trial's {metric} is NaN, not a "
                    f"silently-averaged real number."
                )
            return float(np.nanmean(values))

        def objective(trial):
            params = model_cls().param_space(trial)
            fold_metrics = []
            trial_context = f"[{season}][{model_name}] trial {trial.number}"

            for fold_idx, (train_idx, test_idx) in enumerate(folds):
                context = f"{trial_context} fold {fold_idx + 1}"
                scores = self.cv_strategy.fit_and_score_full(
                    model_cls, params, X_search, y_search, train_idx, test_idx, context=context,
                    model_name=model_name,
                )
                fold_metrics.append(scores)
                mean_pr_auc = _nanmean_metric(fold_metrics, "pr_auc_macro", context)
                logger.info(
                    f"{context} PR_AUC_macro={scores['pr_auc_macro']:.4f} AUC={scores['auc']:.4f} "
                    f"F1_macro={scores['f1_macro']:.4f} QWK={scores['qwk']:.4f} | params={params}"
                )

                trial.report(mean_pr_auc, step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            trial.set_user_attr("auc", _nanmean_metric(fold_metrics, "auc", trial_context))
            trial.set_user_attr("f1_macro", _nanmean_metric(fold_metrics, "f1_macro", trial_context))
            trial.set_user_attr("qwk", _nanmean_metric(fold_metrics, "qwk", trial_context))

            final_pr_auc = _nanmean_metric(fold_metrics, "pr_auc_macro", trial_context)
            if np.isnan(final_pr_auc):
                # Optuna rejects NaN as a completed trial's return value
                # outright ("The value nan is not acceptable") — every fold
                # in this trial was genuinely degenerate (see the warning
                # _nanmean_metric already logged above), so there is no
                # real PR-AUC-macro to report. Pruning is Optuna's own
                # idiom for "this trial can't be scored," and keeps one
                # unscorable trial from crashing the whole study (which a
                # raised ValueError from returning NaN would otherwise do
                # once every trial in the study hit this and
                # study.best_value had no completed trial left to return).
                logger.warning(f"{trial_context}: PR-AUC-macro is NaN for every fold; pruning this trial.")
                raise optuna.TrialPruned()
            return final_pr_auc

        return objective