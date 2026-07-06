# wildfire_susceptibility/modeling/training/cv.py
from typing import List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

from .resampling import SMOTEResampler

logger = logging.getLogger(__name__)


class FoldStrategy:
    """
    Builds standard (StratifiedKFold) and/or spatial (GroupKFold) fold
    splits for a given (X, y, groups), and knows how to fit+score a model
    on one fold — including the SMOTE resample step.
    """

    def __init__(self, config: dict, resampler: SMOTEResampler):
        self.cv_folds = config["modeling"]["cv_folds"]
        self.resampler = resampler

    def make_folds(self, X, y, groups, strategy: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        if strategy == "spatial":
            if groups is None:
                raise ValueError("Spatial CV requested but no spatial groups provided.")
            return list(GroupKFold(n_splits=self.cv_folds).split(X, y, groups=groups))
        return list(StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42).split(X, y))

    def fit_and_score(
        self,
        model_cls,
        params: dict,
        X: pd.DataFrame,
        y: pd.Series,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        context: str = "",
    ) -> float:
        X_tr = X.iloc[train_idx].values
        y_tr = y.iloc[train_idx].values
        X_tr, y_tr = self.resampler.resample(X_tr, y_tr, context=context)

        model = model_cls(**params)
        model.fit(X_tr, y_tr)
        proba = model.predict_proba(X.iloc[test_idx].values)

        try:
            return roc_auc_score(y.iloc[test_idx].values, proba, multi_class="ovr")
        except Exception as exc:
            logger.warning(f"{context}: AUC scoring failed on this fold ({exc}); scoring as 0.0")
            return 0.0

    def mean_auc_across_folds(
        self, model_cls, params, X, y, folds, context: str = "",
    ) -> float:
        scores = [self.fit_and_score(model_cls, params, X, y, tr, te, context) for tr, te in folds]
        return float(np.mean(scores))