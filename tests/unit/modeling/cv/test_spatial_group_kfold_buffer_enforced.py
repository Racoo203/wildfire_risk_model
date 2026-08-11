# tests/unit/modeling/cv/test_spatial_group_kfold_buffer_enforced.py
"""Regression coverage for buffered spatial CV (spatial_buffer_m, opt-in,
see CVStrategy._apply_spatial_buffer in cv/base.py). Companion to
test_spatial_group_kfold_buffer_gap.py, which documents that the *default*
(spatial_buffer_m=0.0) still has no buffer by design - these tests prove
that explicitly configuring a buffer actually enforces one, for both
SpatialGroupKFoldCV and StratifiedSpatialBlockCV."""

import numpy as np
import pandas as pd
import pytest

from wildfire_susceptibility.modeling.dataset_prep import DatasetPrep
from wildfire_susceptibility.modeling.cv.base import CVStrategy
from wildfire_susceptibility.modeling.cv.spatial import SpatialGroupKFoldCV
from wildfire_susceptibility.modeling.cv.stratified_spatial import StratifiedSpatialBlockCV
from wildfire_susceptibility.modeling.training.resampling import SMOTEResampler

BLOCK_SIZE_M = 1000.0
N_BLOCKS_PER_AXIS = 6


def _grid_dataset(rng, rows_per_block=4):
    rows = []
    for bx in range(N_BLOCKS_PER_AXIS):
        for by in range(N_BLOCKS_PER_AXIS):
            for _ in range(rows_per_block):
                x = bx * BLOCK_SIZE_M + rng.uniform(50, BLOCK_SIZE_M - 50)
                y = by * BLOCK_SIZE_M + rng.uniform(50, BLOCK_SIZE_M - 50)
                rows.append((x, y))
    df = pd.DataFrame(rows, columns=["_x", "_y"])
    y = pd.Series(rng.integers(0, 2, size=len(df)))
    return df, y


def _assert_no_train_block_within_radius_of_test(groups, folds, radius):
    for train_idx, test_idx in folds:
        test_blocks = {groups.iloc[i] for i in test_idx}
        train_blocks = {groups.iloc[i] for i in train_idx}
        for tb in train_blocks:
            for sb in test_blocks:
                chebyshev = max(abs(tb[0] - sb[0]), abs(tb[1] - sb[1]))
                assert chebyshev > radius, (
                    f"train block {tb} is within {chebyshev} grid cell(s) of "
                    f"test block {sb} (radius={radius}) - buffer not enforced"
                )


@pytest.mark.parametrize("buffer_m,expected_radius", [(0.0, 0), (999.0, 1), (1000.0, 1), (1500.0, 2), (2000.0, 2)])
def test_buffer_radius_blocks_math(buffer_m, expected_radius):
    resampler = SMOTEResampler({"modeling": {"use_smote": False}})
    strategy = SpatialGroupKFoldCV(
        {"modeling": {"cv_folds": 4, "spatial_buffer_m": buffer_m, "spatial_block_size_m": BLOCK_SIZE_M}},
        resampler,
    )
    assert strategy._buffer_radius_blocks() == expected_radius


def test_spatial_buffer_defaults_to_zero_and_changes_nothing():
    """spatial_buffer_m omitted from config entirely must behave exactly
    like spatial_buffer_m=0.0 - the default has to be a true no-op."""
    resampler = SMOTEResampler({"modeling": {"use_smote": False}})
    strategy = SpatialGroupKFoldCV({"modeling": {"cv_folds": 4}}, resampler)
    assert strategy.spatial_buffer_m == 0.0
    assert strategy._buffer_radius_blocks() == 0


def test_spatial_group_kfold_buffer_excludes_adjacent_blocks():
    rng = np.random.default_rng(21)
    df, y = _grid_dataset(rng)
    X = df[["_x", "_y"]]

    prep = DatasetPrep({"modeling": {}})
    groups = prep.assign_spatial_blocks(df, block_size_m=BLOCK_SIZE_M)

    resampler = SMOTEResampler({"modeling": {"use_smote": False}})
    unbuffered = SpatialGroupKFoldCV(
        {"modeling": {"cv_folds": 4, "spatial_block_size_m": BLOCK_SIZE_M}}, resampler,
    )
    buffered = SpatialGroupKFoldCV(
        {"modeling": {"cv_folds": 4, "spatial_buffer_m": BLOCK_SIZE_M, "spatial_block_size_m": BLOCK_SIZE_M}},
        resampler,
    )

    unbuffered_folds = unbuffered.make_folds(X, y, groups=groups)
    buffered_folds = buffered.make_folds(X, y, groups=groups)

    total_train_unbuffered = sum(len(train_idx) for train_idx, _ in unbuffered_folds)
    total_train_buffered = sum(len(train_idx) for train_idx, _ in buffered_folds)
    assert total_train_buffered < total_train_unbuffered, (
        "buffering should drop some train rows near test blocks; totals are identical"
    )

    _assert_no_train_block_within_radius_of_test(groups, buffered_folds, radius=1)


def test_stratified_spatial_block_buffer_excludes_adjacent_blocks():
    rng = np.random.default_rng(23)
    df, y = _grid_dataset(rng)
    X = df[["_x", "_y"]]

    prep = DatasetPrep({"modeling": {}})
    groups = prep.assign_spatial_blocks(df, block_size_m=BLOCK_SIZE_M)

    resampler = SMOTEResampler({"modeling": {"use_smote": False}})
    buffered = StratifiedSpatialBlockCV(
        {"modeling": {"cv_folds": 4, "spatial_buffer_m": BLOCK_SIZE_M, "spatial_block_size_m": BLOCK_SIZE_M}},
        resampler,
    )
    folds = buffered.make_folds(X, y, groups=groups)

    assert len(folds) == 4
    _assert_no_train_block_within_radius_of_test(groups, folds, radius=1)


def test_apply_spatial_buffer_is_noop_when_unset():
    """Direct unit test of the shared base-class method: with the default
    config (no spatial_buffer_m), train_idx must come back byte-for-byte
    unchanged - no accidental filtering, no copy-with-different-dtype."""
    resampler = SMOTEResampler({"modeling": {"use_smote": False}})
    strategy = SpatialGroupKFoldCV({"modeling": {"cv_folds": 4}}, resampler)

    groups = pd.Series([(0, 0), (0, 0), (1, 0), (1, 0), (5, 5)])
    train_idx = np.array([2, 3, 4])
    test_idx = np.array([0, 1])

    result = strategy._apply_spatial_buffer(train_idx, test_idx, groups)
    assert np.array_equal(result, train_idx)
