# tests/unit/modeling/cv/test_spatial_group_kfold_buffer_gap.py
"""Documents the DEFAULT behavior of spatial CV: with spatial_buffer_m left
at its default (0.0), block assignment (DatasetPrep.assign_spatial_blocks)
is a plain grid floor-division with no distance buffer/exclusion zone
between adjacent blocks. A point can sit a metre from a block boundary and
still land in the opposite CV fold from a near-neighbour just across that
boundary, so unbuffered "spatial CV" only guarantees fold separation at the
block-ID level, not at any minimum physical distance.

Buffered spatial CV (CVStrategy._apply_spatial_buffer, opt-in via
spatial_buffer_m > 0) now exists and fixes exactly this - see
test_spatial_group_kfold_buffer_enforced.py for the corresponding passing
test with a buffer configured. This test intentionally keeps
spatial_buffer_m unset to prove the *default* stays backward-compatible
(no buffer, prior behavior/results unaffected) - it should keep xfailing
forever, not just until a fix lands, since the fix is opt-in by design.

xfail(strict=True): if this ever unexpectedly passes, that means unbuffered
spatial CV started enforcing a distance guarantee it didn't ask for, which
would itself be worth investigating as a silent behavior change."""

import numpy as np
import pandas as pd
import pytest

from wildfire_susceptibility.modeling.dataset_prep import DatasetPrep
from wildfire_susceptibility.modeling.cv.spatial import SpatialGroupKFoldCV
from wildfire_susceptibility.modeling.resampling import SMOTEResampler

BLOCK_SIZE_M = 1000.0
N_BLOCKS_PER_AXIS = 6
MIN_ACCEPTABLE_BUFFER_M = 100.0  # any real buffer implementation should clear this easily


def _build_dataset():
    """A grid of blocks each populated with a few interior points, plus one
    boundary-straddling pair of points (2m apart) at every internal x
    boundary along a fixed row - maximizing the chance at least one such
    pair ends up split across folds, which is exactly the failure mode
    being documented."""
    rng = np.random.default_rng(0)
    rows = []
    for bx in range(N_BLOCKS_PER_AXIS):
        for by in range(N_BLOCKS_PER_AXIS):
            for _ in range(4):
                x = bx * BLOCK_SIZE_M + rng.uniform(150, BLOCK_SIZE_M - 150)
                y = by * BLOCK_SIZE_M + rng.uniform(150, BLOCK_SIZE_M - 150)
                rows.append((x, y))

    boundary_pairs = []
    mid_y = (N_BLOCKS_PER_AXIS // 2) * BLOCK_SIZE_M + BLOCK_SIZE_M / 2
    for bx in range(1, N_BLOCKS_PER_AXIS):
        boundary_x = bx * BLOCK_SIZE_M
        idx_a = len(rows)
        rows.append((boundary_x - 1.0, mid_y))
        idx_b = len(rows)
        rows.append((boundary_x + 1.0, mid_y))
        boundary_pairs.append((idx_a, idx_b))

    df = pd.DataFrame(rows, columns=["_x", "_y"])
    y = pd.Series(rng.integers(0, 2, size=len(df)))
    return df, y, boundary_pairs


@pytest.mark.xfail(
    strict=True,
    reason=(
        "spatial_buffer_m defaults to 0.0 (no buffer) for backward "
        "compatibility - see CVStrategy._apply_spatial_buffer in cv/base.py "
        "and test_spatial_group_kfold_buffer_enforced.py for the opt-in fix."
    ),
)
def test_spatial_group_kfold_enforces_minimum_train_test_distance_by_default():
    df, y, boundary_pairs = _build_dataset()
    X = df[["_x", "_y"]]
    coords = df[["_x", "_y"]].to_numpy()

    prep = DatasetPrep({"modeling": {}})
    groups = prep.assign_spatial_blocks(df, block_size_m=BLOCK_SIZE_M)

    resampler = SMOTEResampler({"modeling": {"use_smote": False}})
    cv = SpatialGroupKFoldCV({"modeling": {"cv_folds": 4}}, resampler)
    folds = cv.make_folds(X, y, groups=groups)

    min_cross_fold_distance = np.inf
    for train_idx, test_idx in folds:
        train_set, test_set = set(train_idx), set(test_idx)
        for idx_a, idx_b in boundary_pairs:
            split_across_fold = (
                (idx_a in train_set and idx_b in test_set)
                or (idx_a in test_set and idx_b in train_set)
            )
            if split_across_fold:
                d = float(np.linalg.norm(coords[idx_a] - coords[idx_b]))
                min_cross_fold_distance = min(min_cross_fold_distance, d)

    assert min_cross_fold_distance != np.inf, (
        "test setup issue: no boundary-adjacent pair was split across train/test "
        "by any fold - widen N_BLOCKS_PER_AXIS/cv_folds so at least one pair crosses."
    )
    assert min_cross_fold_distance > MIN_ACCEPTABLE_BUFFER_M, (
        f"Spatial CV placed boundary-adjacent points only "
        f"{min_cross_fold_distance:.1f}m apart across train/test - no buffer enforced."
    )
