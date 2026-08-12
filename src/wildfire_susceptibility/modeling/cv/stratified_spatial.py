"""Stratified Spatial Block Cross-Validation with In-Fold Resampling.

This is the dissertation's headline CV method (the other two strategies
are now baselines reported for comparison). Steps, mapped to methods:

    1. Grid Partitioning       -> DatasetPrep.assign_spatial_blocks (block_size_m)
    2. Class Stratification    -> _stratify_blocks_into_folds
    3. Boundary Buffering      -> CVStrategy._apply_spatial_buffer (spatial_buffer_m,
                                   optional, 0 by default) drops train blocks within
                                   the buffer ring of each fold's test blocks
    4. Strict Validation       -> make_folds returns held-out fold UNTOUCHED
       Isolation                 (no resampling ever applied to test_idx)
    5. In-Fold Training         -> fit_and_score resamples ONLY the training
       Resampling                 side, using self.resampler (never disabled
                                   for this strategy, unlike the plain spatial
                                   baseline's search-time SMOTE-off convention)
    6. Model Training/Eval      -> fit_and_score_full reports PR-AUC-macro
                                    (the Optuna HPO objective) + AUC/F1-macro/
                                    QWK as documented in the dissertation
                                    methodology
"""
from collections import defaultdict
from typing import List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, cohen_kappa_score

from .base import CVStrategy
from ..balance import log_class_balance, warn_if_class_missing

logger = logging.getLogger(__name__)


class StratifiedSpatialBlockCV(CVStrategy):
    """Blocks (not rows) are the CV unit; blocks are greedily assigned to
    K folds to balance every class's representation — including the
    rarest ("very high") class — across folds."""

    name = "stratified_spatial_block"

    def make_folds(self, X, y, groups: Optional[pd.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError(...)
        groups = pd.Series(groups).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        fold_of_block = self._stratify_blocks_into_folds(groups, y, self.cv_folds)
        row_fold = groups.map(fold_of_block).to_numpy()

        folds = []
        for k in range(self.cv_folds):
            test_idx = np.where(row_fold == k)[0]
            train_idx = np.where(row_fold != k)[0]
            train_idx = self._apply_spatial_buffer(train_idx, test_idx, groups)
            if len(test_idx) == 0 or len(train_idx) == 0:
                logger.warning(f"[stratified_spatial_block] fold {k} is degenerate (empty split); skipping.")
                continue

            # Verifies Step 2 (Class Stratification): confirm every fold —
            # including the smallest, "very high" class — is represented,
            # and log each fold's raw (pre-resample) proportions so a
            # reader can visually confirm stratification worked rather
            # than just trusting the assignment algorithm ran.
            log_class_balance(
                logger, f"[stratified_spatial_block] fold {k} (validation, untouched)", y.iloc[test_idx],
            )
            warn_if_class_missing(
                logger, f"[stratified_spatial_block] fold {k} (validation)", y.iloc[test_idx], expected_classes=sorted(y.unique()),
            )
            folds.append((train_idx, test_idx))

        self._log_cross_fold_balance_summary(y, folds)
        return folds

    @staticmethod
    def _log_cross_fold_balance_summary(y: pd.Series, folds) -> None:
        """One-line-per-class summary of how evenly each class's rows were
        spread across folds — the actual thing 'balanced stratification'
        means, distinct from in-fold SMOTE balance."""
        classes = sorted(y.unique())
        per_fold_props = []
        for _, test_idx in folds:
            fold_y = y.iloc[test_idx]
            fold_counts = fold_y.value_counts()
            per_fold_props.append({c: fold_counts.get(c, 0) / max(len(fold_y), 1) for c in classes})

        for c in classes:
            props = [p[c] for p in per_fold_props]
            spread = (max(props) - min(props)) if props else 0.0
            logger.info(
                f"[stratified_spatial_block] class {c} proportion across folds: "
                f"{[round(p, 3) for p in props]} (spread={spread:.3f})"
            )
            if spread > 0.15:
                logger.warning(
                    f"[stratified_spatial_block] class {c} spread across folds ({spread:.3f}) "
                    f"exceeds 0.15 — block-level stratification may be too coarse for this class "
                    f"at the current spatial_block_size_m."
                )

    @staticmethod
    def _stratify_blocks_into_folds(groups: pd.Series, y: pd.Series, n_folds: int) -> dict:
        classes = sorted(y.unique())
        block_class_counts: dict = defaultdict(lambda: {c: 0 for c in classes})
        for block, cls in zip(groups, y):
            block_class_counts[block][cls] += 1

        class_totals = y.value_counts()
        rarity_order = class_totals.sort_values().index.tolist()

        fold_class_counts = [{c: 0 for c in classes} for _ in range(n_folds)]
        fold_of_block: dict = {}

        # --- Pass 1 (coverage guarantee): for each class, rarest first,
        # round-robin its blocks across every fold BEFORE the greedy fill
        # below runs. This is what actually enforces "every fold has the
        # rarest class represented" — the old single greedy pass could
        # concentrate a scarce class's few blocks into fewer folds than
        # n_folds if their "dominant class" tie-breaking favored a
        # different fold each time.
        for c in rarity_order:
            blocks_with_c = [b for b, counts in block_class_counts.items() if counts[c] > 0 and b not in fold_of_block]
            # Largest-count-first so the biggest contributors to this
            # class get placed while every fold still needs coverage.
            blocks_with_c.sort(key=lambda b: -block_class_counts[b][c])
            for i, block in enumerate(blocks_with_c[:n_folds]):
                target_fold = i % n_folds
                fold_of_block[block] = target_fold
                for cc in classes:
                    fold_class_counts[target_fold][cc] += block_class_counts[block][cc]

        # --- Pass 2 (general greedy fill): remaining unassigned blocks,
        # same logic as before, minimizing per-fold imbalance.
        remaining_blocks = [b for b in block_class_counts if b not in fold_of_block]

        def _priority(block):
            counts = block_class_counts[block]
            return tuple(-counts[c] for c in rarity_order)

        remaining_blocks.sort(key=_priority)

        for block in remaining_blocks:
            counts = block_class_counts[block]
            dominant_class = min(rarity_order, key=lambda c: -counts[c] if counts[c] else 0)
            target_fold = min(
                range(n_folds),
                key=lambda k: (fold_class_counts[k][dominant_class], sum(fold_class_counts[k].values())),
            )
            fold_of_block[block] = target_fold
            for c in classes:
                fold_class_counts[target_fold][c] += counts[c]

        return fold_of_block

    def fit_and_score_full(
        self, model_cls, params, X, y, train_idx, test_idx, context: str = "",
        model_name: Optional[str] = None,
    ) -> dict:
        """Fit once, return the full metric set the dissertation reports
        (AUC, F1-macro, PR-AUC-macro, QWK). `fit_and_score` delegates here
        so a fold's model is only ever fit once even when both the scalar
        and the full metric set are needed (see trainer.py's optimism-gap
        logging). `model_name` resolves modeling.imbalance_strategy the
        same way the base class's fit_and_score_full does — see that
        docstring."""
        y_tr_raw = y.iloc[train_idx]
        log_class_balance(logger, context, y_tr_raw, note="train, pre-resample", level=logging.DEBUG)

        X_tr = X.iloc[train_idx].values
        y_tr = y_tr_raw.values
        X_tr, y_tr = self.resampler.resample(X_tr, y_tr, context=context, model_name=model_name)
        log_class_balance(logger, context, y_tr, note="train, post-resample (used to fit)")

        sample_weight = self._imbalance.sample_weight_for(model_name, y_tr) if model_name else None
        model = model_cls(**params)
        model.fit(X_tr, y_tr, sample_weight=sample_weight)

        X_va, y_va = X.iloc[test_idx].values, y.iloc[test_idx].values
        log_class_balance(logger, context, y_va, note="validation, untouched — isolation check")
        proba = model.predict_proba(X_va)
        pred = np.argmax(proba, axis=1)

        try:
            all_classes = sorted(y.unique())
            auc = float(roc_auc_score(y_va, proba, multi_class="ovr", labels=all_classes))
        except Exception as exc:
            logger.warning(f"{context}: AUC scoring failed ({exc}); scoring as 0.0")
            auc = 0.0

        f1_macro = float(f1_score(y_va, pred, average="macro"))
        y_va_bin = np.eye(proba.shape[1])[y_va.astype(int)]
        pr_auc_macro = float(average_precision_score(y_va_bin, proba, average="macro"))
        qwk = float(cohen_kappa_score(y_va, pred, weights="quadratic"))

        logger.info(
            f"{context} AUC={auc:.4f} F1_macro={f1_macro:.4f} PR_AUC_macro={pr_auc_macro:.4f} "
            f"QWK={qwk:.4f} (validation fold untouched/imbalanced)"
        )
        return {"auc": auc, "f1_macro": f1_macro, "pr_auc_macro": pr_auc_macro, "qwk": qwk}

    def fit_and_score(
        self, model_cls, params, X, y, train_idx, test_idx, context: str = "",
        model_name: Optional[str] = None,
    ) -> float:
        return self.fit_and_score_full(
            model_cls, params, X, y, train_idx, test_idx, context=context, model_name=model_name,
        )["auc"]