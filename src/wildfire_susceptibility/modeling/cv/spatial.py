"""Baseline spatial CV: delegates block/fold construction to
verde.BlockKFold (Fatiando a Terra project, https://www.fatiando.org/verde/)
rather than sklearn.model_selection.GroupKFold over our own pre-assigned
blocks. verde.BlockKFold is the standard, actively-maintained, citable
spatial-block K-fold splitter used across the Python geospatial ecosystem;
it explicitly cites Roberts et al. (2017) — already part of this project's
spatial-CV reference set — as its methodological justification, and its
fold-size balancing (balance=True) avoids the uneven-fold-size failure mode
a hand-rolled block/fold assignment can hit when block density is uneven
across space.

This is the deliberately NON-class-aware spatial CV baseline: it removes
spatial leakage between train/test (points cannot leak across blocks that
verde keeps together) but does not stratify blocks by label the way
StratifiedSpatialBlockCV does — that refinement, and the choice of which of
the two is the dissertation's headline method, live entirely in
stratified_spatial.py. SpatialGroupKFoldCV's role here is to be the boring,
generic "standard thing people reach for when they do spatial CV in
Python" reference point stratified_spatial_block is compared against.

`make_folds`'s `groups` argument is unchanged in shape/meaning at the
interface level (still DatasetPrep.assign_spatial_blocks's output: a
pd.Series of (block_x, block_y) integer grid-index tuples, one per row, on
a spatial_block_size_m grid — see dataset_prep.py). verde.BlockKFold wants
a raw (n_samples, 2) coordinate matrix instead of a group-label Series, so
internally each row's block tuple is expanded back into a representative
(easting, northing) coordinate at that block's center before being handed
to verde. Passing spacing=spatial_block_size_m (the same config key
assign_spatial_blocks already used to build these blocks, reused here
rather than introducing a second block-size knob) means verde's own
binning reproduces our existing grid: every row sharing a block tuple
shares the same synthetic coordinate, so it can never be split into a
different verde block than the rest of its block-mates. Buffering
(spatial_buffer_m, CVStrategy._apply_spatial_buffer) still operates on the
original block-tuple `groups` afterward, unchanged, since it stays
strategy-agnostic.
"""
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import verde as vd

from .base import CVStrategy


class SpatialGroupKFoldCV(CVStrategy):
    """Baseline spatial CV: verde.BlockKFold over pre-assigned spatial
    blocks, with an optional buffer ring (spatial_buffer_m) dropping train
    rows adjacent to each fold's test blocks. Removes spatial leakage but
    does NOT class-stratify blocks and does NOT isolate resampling to the
    training fold only — that refinement is StratifiedSpatialBlockCV."""

    name = "spatial"

    def make_folds(self, X, y, groups: Optional[pd.Series] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        if groups is None:
            raise ValueError("Spatial CV requested but no spatial groups (blocks) were provided.")
        groups = pd.Series(groups).reset_index(drop=True)

        block_size = self.spatial_block_size_m
        block_indices = np.array(groups.tolist(), dtype=float)  # (n, 2): (block_x, block_y)
        coords = (block_indices + 0.5) * block_size  # block-center (easting, northing)

        kfold = vd.BlockKFold(
            spacing=block_size, n_splits=self.cv_folds,
            shuffle=True, random_state=42, balance=True,
        )
        folds = []
        for train_idx, test_idx in kfold.split(coords):
            train_idx = self._apply_spatial_buffer(train_idx, test_idx, groups)
            folds.append((train_idx, test_idx))
        return folds
