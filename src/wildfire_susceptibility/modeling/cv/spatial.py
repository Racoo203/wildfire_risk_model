from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .base import CVStrategy

class SpatialGroupKFoldCV(CVStrategy):
    """Baseline spatial CV: GroupKFold on pre-assigned spatial blocks.
    Removes spatial leakage but does NOT class-stratify blocks and does
    NOT isolate resampling to the training fold only — that refinement
    is StratifiedSpatialBlockCV."""

    name = "spatial"

    def make_folds(self, X, y, groups: Optional[pd.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("Spatial CV requested but no spatial groups (blocks) were provided.")
        return list(GroupKFold(n_splits=self.cv_folds).split(X, y, groups=groups))