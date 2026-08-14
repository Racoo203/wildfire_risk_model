# wildfire_susceptibility/modeling/categorical.py
"""Categorical-feature handling shared by DatasetPrep and the training/CV
machinery.

landuse_class (features/proximity.py::_write_landuse_class) is rasterized
as the 1-indexed position of each fclass in the configured human_fclasses
list (0 = no human-activity land use). That position is arbitrary config
list order, not a real ordinal relationship between land-use types, so it
must never reach a model as a plain numeric column — see
DatasetPrep.encode_categoricals_for_model_family for the per-model-family
handling.
"""
from typing import List, Sequence

import numpy as np
import pandas as pd

CATEGORICAL_FEATURES = ("landuse_class",)


def cat_features_of(model_cls_or_instance) -> List[int]:
    """Recover the cat_features positional-index list carried by a model
    class/instance, regardless of whether it's:
      - a functools.partial-bound model class (pre-fit, used while
        threading cat_features through CV/search, which instantiates
        fresh model objects per fold/trial via model_cls(**params)) — the
        binding lives in `.keywords`, or
      - an already-fitted model instance (post-fit, e.g. at evaluation
        time) — every wrapper stores its constructor kwargs as
        `self.params`, so a bound cat_features kwarg lands there too.
    Returns [] for anything that never had cat_features bound (every
    non-CatBoost model, always)."""
    keywords = getattr(model_cls_or_instance, "keywords", None)
    if keywords is not None:
        return list(keywords.get("cat_features", []) or [])
    params = getattr(model_cls_or_instance, "params", None)
    if params is not None:
        return list(params.get("cat_features", []) or [])
    return []


def to_model_array(X: pd.DataFrame, cat_feature_positions: Sequence[int]) -> np.ndarray:
    """Convert a features DataFrame to the array a model wrapper's fit()/
    predict_proba() expects.

    With no categorical columns, this is exactly the prior `.values`
    behavior (a plain float64 ndarray) — unchanged for every model
    family except CatBoost.

    With categorical columns, a homogeneous float64 ndarray can't carry
    them at all: CatBoost validates cat_features against the whole
    array's dtype, not per-column, and rejects a float array outright
    ("no categorical features" error) even when cat_features is set and
    the column itself holds integer-valued codes. The column has to
    reach CatBoost as an actual int/str value inside an object-dtype
    array (or a DataFrame) — confirmed directly against CatBoostClassifier
    before implementing this.
    """
    if not cat_feature_positions:
        return X.values
    arr = X.to_numpy(dtype=object)
    for pos in cat_feature_positions:
        arr[:, pos] = X.iloc[:, pos].astype(int).to_numpy()
    return arr
