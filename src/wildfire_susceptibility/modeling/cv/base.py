"""Shared CV mechanics every strategy inherits: fold scoring and the
mean-AUC convenience wrapper. Subclasses implement only make_folds()."""

from abc import ABC, abstractmethod
from math import ceil
from typing import List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score, cohen_kappa_score

from ..resampling import SMOTEResampler

logger = logging.getLogger(__name__)


class CVStrategy(ABC):
    """One CV strategy = one way to build folds + one way to fit/score a fold.

    Subclasses embody a *specific* strategy (standard / spatial / stratified
    spatial block) rather than being selected via a string switch at call
    time — callers get the right strategy from cv.factory.get_cv_strategy().
    """

    name: str = "base"

    def __init__(self, config: dict, resampler: SMOTEResampler):
        self.config = config
        self.cv_folds = config["modeling"]["cv_folds"]
        self.resampler = resampler
        self.spatial_buffer_m = config["modeling"].get("spatial_buffer_m", 0.0) or 0.0
        self.spatial_block_size_m = config["modeling"].get("spatial_block_size_m", 5000.0)

    @abstractmethod
    def make_folds(
        self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return a list of (train_idx, test_idx) positional-index pairs."""
        ...

    def _buffer_radius_blocks(self) -> int:
        """How many grid cells out from a test block to exclude, so no kept
        train point can be closer than spatial_buffer_m to a test block's
        edge. Blocks are spatial_block_size_m squares, so a ring of radius r
        guarantees a minimum gap of (r-1)*spatial_block_size_m; ceil() rounds
        up so the configured buffer is always met, never just approached."""
        if self.spatial_buffer_m <= 0:
            return 0
        return ceil(self.spatial_buffer_m / self.spatial_block_size_m)

    def _apply_spatial_buffer(
        self, train_idx: np.ndarray, test_idx: np.ndarray, groups: pd.Series,
    ) -> np.ndarray:
        """Drop from train_idx any row whose (block_x, block_y) falls within
        the buffer ring around any test block for this fold. test_idx itself
        is never touched — buffering only removes near-boundary train rows
        that would otherwise sit right next to a test block with zero
        enforced separation (see cv/spatial.py, cv/stratified_spatial.py).
        No-op when spatial_buffer_m is 0 (the default), so existing configs
        and results are unaffected unless a buffer is explicitly set."""
        radius = self._buffer_radius_blocks()
        if radius <= 0:
            return train_idx

        test_blocks = {groups.iloc[i] for i in test_idx}
        excluded_blocks = {
            (bx + dx, by + dy)
            for bx, by in test_blocks
            for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
        }

        train_blocks = groups.iloc[train_idx].to_numpy()
        keep_mask = np.fromiter(
            (block not in excluded_blocks for block in train_blocks),
            dtype=bool, count=len(train_blocks),
        )
        dropped = int((~keep_mask).sum())
        if dropped:
            logger.info(
                f"[{self.name}] spatial buffer ({self.spatial_buffer_m:.0f}m, "
                f"radius={radius} block(s)): dropped {dropped:,}/{len(train_idx):,} "
                f"train rows within the buffer ring of this fold's test blocks"
            )
        return train_idx[keep_mask]

    def fit_and_score_full(
        self, model_cls, params: dict, X: pd.DataFrame, y: pd.Series,
        train_idx: np.ndarray, test_idx: np.ndarray, context: str = "",
    ) -> dict:
        """Fit once on train_idx (resampled per self.resampler), score AUC,
        F1-macro, PR-AUC-macro, and QWK on test_idx untouched — all
        multiclass-safe (roc_auc_score/f1_score use ovr/macro averaging,
        average_precision_score is computed on one-hot-encoded y, QWK uses
        quadratic-weighted Cohen's kappa since the 4 classes are an ordinal
        risk discretization — distant misclassifications are penalized more
        than adjacent ones). Shared by every strategy; strategies that need
        different resampling semantics (e.g. stratified-spatial-block's
        extra class-balance logging) override this. `fit_and_score`
        delegates here so a fold's model is only ever fit once even when
        both the scalar and the full metric set are needed (see
        trainer.py's optimism-gap logging)."""
        logger.info(f"{context}: fitting on {len(train_idx):,} rows...")
        X_tr = X.iloc[train_idx].values
        y_tr = y.iloc[train_idx].values
        X_tr, y_tr = self.resampler.resample(X_tr, y_tr, context=context)

        model = model_cls(**params)
        model.fit(X_tr, y_tr)

        y_va = y.iloc[test_idx].values
        proba = model.predict_proba(X.iloc[test_idx].values)
        pred = np.argmax(proba, axis=1)

        try:
            all_classes = sorted(y.unique())
            auc_score = float(roc_auc_score(y_va, proba, multi_class="ovr", labels=all_classes))
        except Exception as exc:
            logger.warning(f"{context}: AUC scoring failed on this fold ({exc}); scoring as 0.0")
            auc_score = 0.0

        f1_macro = float(f1_score(y_va, pred, average="macro"))

        try:
            y_va_bin = np.eye(proba.shape[1])[y_va.astype(int)]
            pr_auc_macro = float(average_precision_score(y_va_bin, proba, average="macro"))
        except Exception as exc:
            logger.warning(f"{context}: PR-AUC scoring failed on this fold ({exc}); scoring as 0.0")
            pr_auc_macro = 0.0

        qwk = float(cohen_kappa_score(y_va, pred, weights="quadratic"))

        return {"auc": auc_score, "f1_macro": f1_macro, "pr_auc_macro": pr_auc_macro, "qwk": qwk}

    def fit_and_score(
        self, model_cls, params: dict, X: pd.DataFrame, y: pd.Series,
        train_idx: np.ndarray, test_idx: np.ndarray, context: str = "",
    ) -> float:
        """Scalar-AUC convenience wrapper around fit_and_score_full, kept
        for callers (HyperparamSearch's Optuna objective) that only need
        the one number."""
        return self.fit_and_score_full(model_cls, params, X, y, train_idx, test_idx, context=context)["auc"]

    def mean_auc_across_folds(
        self, model_cls, params, X, y, folds, context: str = "",
    ) -> float:
        scores = []
        for i, (train_idx, test_idx) in enumerate(folds):
            fold_context = f"{context} fold {i + 1}/{len(folds)}"
            auc = self.fit_and_score(model_cls, params, X, y, train_idx, test_idx, context=fold_context)
            logger.info(f"{fold_context} AUC={auc:.4f}")
            scores.append(auc)
        return float(np.mean(scores))