"""Deliberately naive, non-spatial CV baseline.

StandardKFoldCV is the intentionally spatially-blind counterpart to
SpatialGroupKFoldCV / StratifiedSpatialBlockCV: it exists so this project
can reproduce the overoptimism-gap demonstration in Schratz et al. (2019,
Ecological Modelling 406, Section 4.2.3 / Fig. 6), which shows non-spatial
CV systematically overestimates predictive performance on spatially
autocorrelated data relative to spatially-blocked CV. It plays the same
role Schratz et al.'s "non-spatial" setting plays in their comparison.

This is NOT a bug to fix or an implementation to improve — the class
imbalance/data leakage it doesn't guard against (no spatial blocking, no
buffering) is the point. Do not swap in a spatially-aware split here or
add spatial `groups` handling; that would defeat its purpose as the
overoptimistic baseline in the standard-vs-spatial comparison."""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .base import CVStrategy


class StandardKFoldCV(CVStrategy):
    """Baseline: class-stratified K-fold, no spatial awareness. Used to
    quantify the optimism gap against spatially-aware strategies (see the
    module docstring above — this is the deliberately naive baseline from
    Schratz et al. (2019), not intended to be improved)."""

    name = "standard"

    def make_folds(self, X, y, groups: Optional[pd.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        return list(
            StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42).split(X, y)
        )