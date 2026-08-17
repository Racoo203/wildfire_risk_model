"""Shared one-hot expansion for categorical columns in EDA/diagnostic
correlation views (VIF/Spearman heatmap, full pairwise Spearman export).
Not used anywhere near model training -- see modeling/categorical.py and
DatasetPrep.fit_categorical_encoding/apply_categorical_encoding for the
train/apply-split encoding models actually see."""

from typing import Dict, List, Tuple

import pandas as pd


def one_hot_encode_categoricals(
    df: pd.DataFrame, categorical_cols: Tuple[str, ...]
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Replace each categorical column with its one-hot dummy columns.

    Returns the expanded dataframe and a {original_col: [dummy_cols]} map
    so callers can flag which one-hot columns share a parent categorical --
    pairs within one group are mechanically anti-correlated (mutual
    exclusivity: at most one "1" per row across the group), not a
    substantive association, and should be reported as such rather than
    silently mixed in with real associations.

    Dummies are cast to float64 (pandas' get_dummies default is bool) so
    every column in the returned dataframe is a uniform numeric dtype. The
    source column is cast to nullable Int64 first (raster-derived columns
    like landuse_class arrive as float64) so dummy names read
    "landuse_class_3" not "landuse_class_3.0" -- the same cosmetic fix
    DatasetPrep.fit_categorical_encoding already applies for the same
    reason (dataset_prep.py).
    """
    out = df.copy()
    parent_map: Dict[str, List[str]] = {}
    for col in categorical_cols:
        if col not in out.columns:
            continue
        dummies = pd.get_dummies(out[col].astype("Int64"), prefix=col, dtype="float64")
        parent_map[col] = list(dummies.columns)
        out = out.drop(columns=[col]).join(dummies)
    return out, parent_map
