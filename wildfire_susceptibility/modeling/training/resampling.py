# wildfire_susceptibility/modeling/training/resampling.py
"""SMOTE oversampling, isolated from the training loop so ModelTrainer
doesn't need to know anything about imblearn or degenerate-fold edge cases."""

from typing import Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)


class SMOTEResampler:
    """
    Applies SMOTE to a single training fold/refit, never to a held-out
    fold or the test set — call this only on data you're about to fit on.
    """

    def __init__(self, config: dict):
        modeling_cfg = config["modeling"]
        self.enabled = modeling_cfg.get("use_smote", False)
        self.k_neighbors = modeling_cfg.get("smote_k_neighbors", 5)

    def resample(self, X: np.ndarray, y: np.ndarray, context: str = "") -> Tuple[np.ndarray, np.ndarray]:
        if not self.enabled:
            return X, y

        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            logger.warning(
                f"{context}: use_smote=True but imbalanced-learn is not installed "
                f"(pip install -e '.[modeling]'); skipping SMOTE for this call."
            )
            return X, y

        classes, counts = np.unique(y, return_counts=True)
        min_class_count = int(counts.min())

        if min_class_count <= 1:
            logger.warning(
                f"{context}: smallest class has {min_class_count} sample(s) in this fold; "
                f"skipping SMOTE (cannot form neighbors)."
            )
            return X, y

        k = min(self.k_neighbors, min_class_count - 1)
        smote = SMOTE(k_neighbors=k, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        logger.debug(
            f"{context}: SMOTE {dict(zip(classes, counts))} -> "
            f"{dict(zip(*np.unique(y_res, return_counts=True)))} (k={k})"
        )
        return X_res, y_res