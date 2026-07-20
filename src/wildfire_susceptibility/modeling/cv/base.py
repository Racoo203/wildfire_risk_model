"""Shared CV mechanics every strategy inherits: fold scoring and the
mean-AUC convenience wrapper. Subclasses implement only make_folds()."""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from ..training.resampling import SMOTEResampler

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

    @abstractmethod
    def make_folds(
        self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return a list of (train_idx, test_idx) positional-index pairs."""
        ...

    def fit_and_score(
        self, model_cls, params: dict, X: pd.DataFrame, y: pd.Series,
        train_idx: np.ndarray, test_idx: np.ndarray, context: str = "",
    ) -> float:
        """Fit on train_idx (resampled per self.resampler), score AUC on
        test_idx untouched. Shared by every strategy; strategies that need
        different resampling semantics (e.g. stratified-spatial-block's
        strict train-only resampling) override this."""
        logger.info(f"{context}: fitting on {len(train_idx):,} rows...")
        X_tr = X.iloc[train_idx].values
        y_tr = y.iloc[train_idx].values
        X_tr, y_tr = self.resampler.resample(X_tr, y_tr, context=context)

        model = model_cls(**params)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X.iloc[test_idx].values)

        try:
            # all_classes = sorted(y.unique())
            precision, recall, _ = precision_recall_curve(y.iloc[test_idx].values, proba)
            score = auc(recall, precision)
            return float(
                # roc_auc_score(
                #     y.iloc[test_idx].values, proba, multi_class="ovr", labels=all_classes,
                # )
                score   
            )
        except Exception as exc:
            logger.warning(f"{context}: AUC scoring failed on this fold ({exc}); scoring as 0.0")
            return 0.0

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