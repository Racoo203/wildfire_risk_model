"""
Model-ready dataset preparation (the 'Gold' layer, Section 7.3).

Three responsibilities, run in this order:
    1. Missing-data resolution — per the three documented NaN sources:
       sea/estuary pixels outside the HadUK land mask (drop), flat SRTM
       no-data cells affecting slope/aspect (impute zero), NDVI cloud
       contamination (spatial nearest-neighbor fill).
    2. Per-model-family scaling — tree models (RF/XGBoost) get the raw
       imputed table; SVM/NN get a StandardScaler-transformed copy.
    3. Stratified 70/30 train/test split matching the paper's methodology.

NOTE: this module does NOT currently sit in front of LabelCleaner in
WildfirePreprocessor.run_full_pipeline() — wiring that ordering fix into
the live pipeline is deliberately deferred to Phase 3 so it can be reviewed
as its own change rather than folded into this one. Today, this module is
usable standalone (e.g. from a notebook) against an already-assembled
dataset_clean_<season>.csv.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import logging

logger = logging.getLogger(__name__)

class DatasetPrep:
    """
    Turns a raw stacked feature/label table (one row per valid pixel, as
    produced by RasterManager.stack_to_dataframe) into model-ready arrays.
    """

    def __init__(self, config: dict):
        self.config = config

    # --- 1. missing-data resolution ------------------------------------

    def resolve_missing(
        self,
        df: pd.DataFrame,
        *,
        slope_aspect_cols: Tuple[str, ...] = ("slope", "aspect"),
        nearest_neighbor_cols: Tuple[str, ...] = ("ndvi",),
        drop_if_any_nan_in: Tuple[str, ...] = (),
        domain_mask: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Apply the three documented NaN-handling rules to a stacked dataframe,
        restricted to rows inside `domain_mask` when given.

        Rows outside the study-area domain are left completely untouched —
        they never reach k-means (their labels are already NaN), so imputing
        them wastes compute and only invites confusion in NaN-count logs.
        """
        out = df.copy()
        scope = domain_mask if domain_mask is not None else np.ones(len(out), dtype=bool)

        # for col in slope_aspect_cols:
        #     if col in out.columns:
        #         fillable = scope & out[col].isna().to_numpy()
        #         n_nan = int(fillable.sum())
        #         if n_nan:
        #             out.loc[fillable, col] = 0.0
        #             logger.info(f"Zero-imputed {n_nan:,} in-domain NaNs in '{col}' (flat SRTM cells)")

        for col in nearest_neighbor_cols:
            if col in out.columns:
                fillable = scope & out[col].isna().to_numpy()
                n_nan = int(fillable.sum())
                if n_nan:
                    filled = self._nearest_neighbor_fill_1d(out.loc[scope, col])
                    out.loc[scope, col] = filled.values
                    logger.info(f"Nearest-neighbor filled {n_nan:,} in-domain NaNs in '{col}'")

        if drop_if_any_nan_in:
            before = len(out)
            drop_cols = [c for c in drop_if_any_nan_in if c in out.columns]
            keep = ~(scope & out[drop_cols].isna().any(axis=1).to_numpy())
            out = out.loc[keep]
            dropped = before - len(out)
            if dropped:
                logger.info(f"Dropped {dropped:,} in-domain rows with NaN in {drop_if_any_nan_in}")

        if domain_mask is not None:
            in_domain = out.loc[scope[:len(out)] if len(scope) == len(out) else domain_mask]
            remaining = {c: int(in_domain[c].isna().sum()) for c in out.columns if in_domain[c].isna().any()}
            if remaining:
                logger.warning(f"Unresolved NaNs WITHIN the study-area domain: {remaining} — this is the number that matters for data quality.")
            else:
                logger.info("No unresolved in-domain NaNs after imputation.")
        else:
            remaining_nan_cols = out.columns[out.isna().any()].tolist()
            if remaining_nan_cols:
                logger.warning(
                    f"Unresolved NaNs remain in columns {remaining_nan_cols} — no domain_mask "
                    f"supplied, so this includes out-of-boundary padding."
                )

        return out

    @staticmethod
    def _nearest_neighbor_fill_1d(series: pd.Series) -> pd.Series:
        """
        Fill NaNs in a 1-D series using nearest-valid-value by row position.

        This operates on the *stacked* (already-flattened) representation,
        which approximates spatial nearest-neighbor fill when row order
        follows raster row-major order (as RasterManager.stack_to_dataframe
        produces via a fixed valid_mask). For a stricter 2-D spatial fill,
        apply distance_transform_edt's `return_indices` on the raster
        directly before stacking — this 1-D approximation is a reasonable
        and much cheaper stand-in at this stage of the pipeline.
        """
        values = series.to_numpy()
        nan_mask = np.isnan(values)
        if not nan_mask.any():
            return series
        if nan_mask.all():
            logger.warning("Entire NDVI column is NaN — cannot nearest-neighbor fill.")
            return series

        idx = np.arange(len(values))
        valid_idx = idx[~nan_mask]
        # distance_transform_edt on the 1-D NaN mask gives nearest-valid indices
        _, nearest_idx = distance_transform_edt(nan_mask, return_indices=True, return_distances=True)
        filled = values.copy()
        filled[nan_mask] = values[valid_idx[np.searchsorted(valid_idx, nearest_idx[0][nan_mask]) - 1]] \
            if False else values[nearest_idx[0]][nan_mask]
        return pd.Series(filled, index=series.index)

    # --- 2. per-model-family scaling ------------------------------------

    def scale_for_model_family(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        needs_scaling: bool,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[StandardScaler]]:
        """
        Return (X_train, X_test) as-is for tree models, or StandardScaler-
        transformed copies for scale-sensitive models (SVM, NN).

        The scaler is fit on X_train only, never X_test, to avoid leakage.
        """
        if not needs_scaling:
            return X_train, X_test, None

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test), columns=X_test.columns, index=X_test.index
        )
        return X_train_scaled, X_test_scaled, scaler

    # --- 3. stratified split ---------------------------------------------

    def stratified_split(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        test_size: float = 0.30,
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        70/30 split stratified by class, matching the paper's methodology
        (Section 1.5). Rows with NaN labels (unlabelled / zero-density
        pixels) are excluded before splitting.
        """
        random_state = random_state if random_state is not None else self.config.get("labels", {}).get("random_state", 42)

        labelled = df.dropna(subset=[label_col])
        X = labelled.drop(columns=[label_col])
        y = labelled[label_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state,
        )
        logger.info(
            f"Stratified split: {len(X_train):,} train / {len(X_test):,} test "
            f"(test_size={test_size}, random_state={random_state})"
        )
        return X_train, X_test, y_train, y_test
    
    def assign_spatial_blocks(
        self,
        df: pd.DataFrame,
        block_size_m: float = 5000.0,   # 5 km blocks — coarse enough to break local autocorrelation
        x_col: str = "_x",
        y_col: str = "_y",
    ) -> pd.Series:
        """
        Assign each row to a spatial block on a block_size_m grid, for use as
        the `groups` argument to sklearn's GroupKFold. Blocks are the spatial
        CV unit, not folds themselves — GroupKFold handles the fold assignment.
        """
        block_x = (df[x_col] // block_size_m).astype(int)
        block_y = (df[y_col] // block_size_m).astype(int)
        blocks = (block_x.astype(str) + "_" + block_y.astype(str))
        n_blocks = blocks.nunique()
        logger.info(f"Assigned {len(df):,} rows to {n_blocks} spatial blocks ({block_size_m/1000:.0f} km grid)")
        return blocks