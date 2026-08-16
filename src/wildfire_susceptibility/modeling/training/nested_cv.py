# wildfire_susceptibility/modeling/training/nested_cv.py
"""Genuine nested cross-validation: removes the Varma & Simon (2006)
optimism bias from this project's reported CV metrics by never letting a
row/fold used to report performance also influence hyperparameter
selection.

    - OUTER loop: whatever CVStrategy is configured for reporting
      (standard / spatial / stratified_spatial_block). Produces the
      train/test splits used ONLY for performance estimation. The outer
      test fold is never seen by the inner tuning process at all, at the
      row level or the block level.
    - INNER loop: always StandardKFoldCV, built fresh from each outer
      fold's TRAINING data only (never the full dataset, never the outer
      test fold), regardless of the outer strategy. Used only to score
      candidate hyperparameter settings during Optuna's search.

The winning hyperparameter setting from the inner loop is refit on the
FULL outer-training-fold data and scored once on the untouched outer test
fold — that score, and only that score, contributes to the reported
performance estimate.

The inner loop is deliberately non-spatial regardless of the outer
strategy, per Schratz et al. (2019, Ecological Modelling 406, Section
5.1.4): no major differences in model performance were found between
spatial and non-spatial hyperparameter tuning once the outer/reporting
loop is already spatially blocked. The outer/inner nesting structure
itself follows Schratz et al. (2019) Section 3.2.1 / Fig. 3.

No CVStrategy subclass (including StandardKFoldCV, used here as the
inner loop) has any awareness of being "inner" or "outer" — that
knowledge belongs entirely to this module. CVStrategy.make_folds and
fit_and_score_full are reused completely unchanged.
"""

from typing import Callable, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

from ..cv.base import CVStrategy
from ..cv.factory import get_cv_strategy, requires_spatial_groups
from ..resampling import SMOTEResampler
from .search import HyperparamSearch

logger = logging.getLogger(__name__)

METRICS = ("auc", "f1_macro", "pr_auc_macro", "qwk")


def _inner_config(config: dict) -> dict:
    """Inner loop gets its own (typically smaller) fold count via
    modeling.optuna_inner_cv_folds — CVStrategy.__init__ always reads
    config["modeling"]["cv_folds"], so this overrides that key rather than
    adding any "am I inner" branch to CVStrategy itself."""
    inner_folds = config["modeling"].get("optuna_inner_cv_folds", 3)
    return {**config, "modeling": {**config["modeling"], "cv_folds": inner_folds}}


def run_nested_cv(
    outer_strategy: CVStrategy,
    search_resampler: SMOTEResampler,
    model_cls,
    model_name: str,
    season: str,
    config: dict,
    X_tr: pd.DataFrame,
    y_train: pd.Series,
    groups_train: Optional[pd.Series],
    progress_callback: Optional[Callable[[str, int, float], None]] = None,
) -> Tuple[dict, List[dict]]:
    """Runs genuinely nested CV under `outer_strategy` and returns
    (mean_metrics, per_outer_fold_metrics) — the same shape the old
    single-level `_run_full_metrics_pass` returned, so trainer.py's call
    sites swap in with no shape changes downstream (stage_selection.py's
    Kruskal-Wallis test, manifest fields, etc. all keep receiving
    outer-fold-level scores).

    For each outer fold: an independent Optuna study is created, scoped
    to ONLY that outer fold's training rows, tuned via a fresh
    StandardKFoldCV instance built on those rows (never the outer test
    fold, never another outer fold's rows). The winning params are then
    refit on the full outer-training fold and scored once on the
    untouched outer test fold via outer_strategy.fit_and_score_full —
    reused as-is, so any strategy-specific override (e.g.
    StratifiedSpatialBlockCV's own resampling/logging) is honored
    automatically.
    """
    if groups_train is not None and not isinstance(groups_train, pd.Series):
        groups_train = pd.Series(groups_train, index=X_tr.index)

    outer_groups = groups_train if requires_spatial_groups(outer_strategy.name) else None
    outer_folds = outer_strategy.make_folds(X_tr, y_train, outer_groups)
    n_outer = len(outer_folds)

    inner_cfg = _inner_config(config)

    per_outer_fold_metrics = []
    for i, (outer_train_idx, outer_test_idx) in enumerate(outer_folds):
        X_outer_train = X_tr.iloc[outer_train_idx].reset_index(drop=True)
        y_outer_train = y_train.iloc[outer_train_idx].reset_index(drop=True)
        groups_outer_train = (
            groups_train.iloc[outer_train_idx].reset_index(drop=True)
            if groups_train is not None else None
        )

        # Inner strategy/search are built fresh per outer fold: always
        # "standard" regardless of outer_strategy.name, and scoped only to
        # this outer fold's training rows — this is what guarantees
        # _subsample_blocks (block-aware search subsampling) never fires
        # for the inner loop (requires_spatial_groups("standard") is
        # always False), and that row-level subsampling only ever sees
        # this outer fold's own training population, not the full dataset.
        inner_strategy = get_cv_strategy("standard", inner_cfg, search_resampler)
        inner_search = HyperparamSearch(inner_cfg, inner_strategy)

        # Reusing `season` as the Optuna study-name vehicle (get_or_create_study
        # builds the study name from it) keeps HyperparamSearch/search.py
        # completely untouched: the outer-fold identity and "inner-standard"
        # marker just ride along as part of the season string, guaranteeing
        # these studies can't collide with the unrelated production study
        # (plain `season`, no suffix) or with each other across outer folds
        # or across "both" mode's two outer strategies.
        inner_season_label = (
            f"{season}__outerfold{i}of{n_outer}__outer-{outer_strategy.name}__inner-standard"
        )
        context = f"[{season}][{model_name}] nested outer fold {i + 1}/{n_outer} ({outer_strategy.name})"
        logger.info(f"{context}: tuning inner loop on {len(outer_train_idx):,} outer-train rows")

        inner_result = inner_search.run_search_and_report(
            model_cls, model_name, inner_season_label,
            X_outer_train, y_outer_train, groups_outer_train,
            progress_callback,
        )
        best_inner_params = inner_result["best_params"]

        # Refit on the FULL outer-training fold (not the inner search's own
        # subsample), score once on the untouched outer test fold. This is
        # the only score that contributes to the reported estimate.
        fold_metrics = outer_strategy.fit_and_score_full(
            model_cls, best_inner_params, X_tr, y_train, outer_train_idx, outer_test_idx,
            context=f"{context} refit+score", model_name=model_name,
        )
        logger.info(
            f"{context}: AUC={fold_metrics['auc']:.4f} F1_macro={fold_metrics['f1_macro']:.4f} "
            f"PR_AUC_macro={fold_metrics['pr_auc_macro']:.4f} QWK={fold_metrics['qwk']:.4f} "
            f"| best_inner_params={best_inner_params}"
        )
        per_outer_fold_metrics.append(fold_metrics)

    mean_metrics = {
        metric: float(np.mean([f[metric] for f in per_outer_fold_metrics]))
        for metric in METRICS
    }
    return mean_metrics, per_outer_fold_metrics
