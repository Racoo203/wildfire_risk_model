from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .base import CVStrategy


class StandardKFoldCV(CVStrategy):
    """Baseline: class-stratified K-fold, no spatial awareness. Used to
    quantify the optimism gap against spatially-aware strategies."""

    name = "standard"

    def make_folds(self, X, y, groups: Optional[pd.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        return list(
            StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42).split(X, y)
        )