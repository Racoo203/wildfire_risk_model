# tests/unit/modeling/cv/test_spatial_verde_blockkfold.py
"""Direct coverage for SpatialGroupKFoldCV's verde.BlockKFold-backed
make_folds() (cv/spatial.py) — the generic, non-class-aware spatial CV
baseline, now delegating block/fold construction to verde.BlockKFold
instead of sklearn.model_selection.GroupKFold over our own pre-assigned
blocks. Confirms the handoff from DatasetPrep.assign_spatial_blocks's
(block_x, block_y) grouping to verde's own coordinate-based block/fold
logic preserves the properties that mattered before: blocks (not
individual rows) are the CV unit, fold count matches config, and the split
is deterministic given the hardcoded random_state=42 (same convention as
StandardKFoldCV — see test_standard.py). Buffer-enforcement coverage for
this strategy lives in test_spatial_group_kfold_buffer_enforced.py /
test_spatial_group_kfold_buffer_gap.py and is unchanged by this swap."""

import numpy as np
import pandas as pd

from wildfire_susceptibility.modeling.dataset_prep import DatasetPrep
from wildfire_susceptibility.modeling.cv.spatial import SpatialGroupKFoldCV
from wildfire_susceptibility.modeling.resampling import SMOTEResampler

BLOCK_SIZE_M = 1000.0
N_BLOCKS_PER_AXIS = 8
ROWS_PER_BLOCK = 5
CV_FOLDS = 4


def _grid_dataset(rng, rows_per_block=ROWS_PER_BLOCK):
    rows = []
    for bx in range(N_BLOCKS_PER_AXIS):
        for by in range(N_BLOCKS_PER_AXIS):
            for _ in range(rows_per_block):
                x = bx * BLOCK_SIZE_M + rng.uniform(50, BLOCK_SIZE_M - 50)
                y = by * BLOCK_SIZE_M + rng.uniform(50, BLOCK_SIZE_M - 50)
                rows.append((x, y))
    df = pd.DataFrame(rows, columns=["_x", "_y"])
    y = pd.Series(rng.integers(0, 4, size=len(df)))
    return df, y


def _make_strategy(cv_folds=CV_FOLDS, block_size_m=BLOCK_SIZE_M):
    resampler = SMOTEResampler({"modeling": {"use_smote": False}})
    return SpatialGroupKFoldCV(
        {"modeling": {"cv_folds": cv_folds, "spatial_block_size_m": block_size_m}}, resampler,
    )


def test_missing_groups_raises():
    strategy = _make_strategy()
    X = pd.DataFrame({"f1": [1.0, 2.0]})
    y = pd.Series([0, 1])
    try:
        strategy.make_folds(X, y, groups=None)
        assert False, "expected ValueError when groups is None"
    except ValueError as exc:
        assert "spatial groups" in str(exc).lower()


def test_blocks_never_split_across_train_and_test_within_a_fold():
    """The actual spatial-leakage guarantee: every row sharing an
    assign_spatial_blocks block tuple must land entirely in train or
    entirely in test for a given fold — verde's own re-binning (at
    spacing=spatial_block_size_m) must not fragment an original block."""
    rng = np.random.default_rng(11)
    df, y = _grid_dataset(rng)
    X = df[["_x", "_y"]]

    prep = DatasetPrep({"modeling": {}})
    groups = prep.assign_spatial_blocks(df, block_size_m=BLOCK_SIZE_M)

    strategy = _make_strategy()
    folds = strategy.make_folds(X, y, groups=groups)

    for train_idx, test_idx in folds:
        train_blocks = set(groups.iloc[train_idx])
        test_blocks = set(groups.iloc[test_idx])
        assert train_blocks.isdisjoint(test_blocks), (
            "a spatial block was split across train and test within one fold"
        )


def test_fold_count_matches_cv_folds_config():
    rng = np.random.default_rng(13)
    df, y = _grid_dataset(rng)
    X = df[["_x", "_y"]]
    prep = DatasetPrep({"modeling": {}})
    groups = prep.assign_spatial_blocks(df, block_size_m=BLOCK_SIZE_M)

    for cv_folds in (3, 4, 5):
        strategy = _make_strategy(cv_folds=cv_folds)
        folds = strategy.make_folds(X, y, groups=groups)
        assert len(folds) == cv_folds


def test_no_train_test_overlap_and_every_row_covered_once():
    rng = np.random.default_rng(17)
    df, y = _grid_dataset(rng)
    X = df[["_x", "_y"]]
    prep = DatasetPrep({"modeling": {}})
    groups = prep.assign_spatial_blocks(df, block_size_m=BLOCK_SIZE_M)

    strategy = _make_strategy()
    folds = strategy.make_folds(X, y, groups=groups)

    row_test_count = np.zeros(len(y), dtype=int)
    for train_idx, test_idx in folds:
        assert set(train_idx).isdisjoint(set(test_idx))
        row_test_count[test_idx] += 1

    assert (row_test_count == 1).all()


def test_make_folds_is_deterministic_across_calls():
    """random_state is fixed (hardcoded 42, same convention as
    StandardKFoldCV), so repeated calls on the same inputs must yield
    byte-identical fold assignments."""
    rng = np.random.default_rng(19)
    df, y = _grid_dataset(rng)
    X = df[["_x", "_y"]]
    prep = DatasetPrep({"modeling": {}})
    groups = prep.assign_spatial_blocks(df, block_size_m=BLOCK_SIZE_M)

    strategy = _make_strategy()
    folds_a = strategy.make_folds(X, y, groups=groups)
    folds_b = strategy.make_folds(X, y, groups=groups)

    assert len(folds_a) == len(folds_b)
    for (train_a, test_a), (train_b, test_b) in zip(folds_a, folds_b):
        np.testing.assert_array_equal(sorted(train_a), sorted(train_b))
        np.testing.assert_array_equal(sorted(test_a), sorted(test_b))


def test_fold_sizes_are_reasonably_balanced():
    """verde.BlockKFold(balance=True) targets approximately equal
    per-fold row counts (not just equal block counts) - confirms this
    default is actually taking effect through our call, not silently
    overridden."""
    rng = np.random.default_rng(23)
    df, y = _grid_dataset(rng)
    X = df[["_x", "_y"]]
    prep = DatasetPrep({"modeling": {}})
    groups = prep.assign_spatial_blocks(df, block_size_m=BLOCK_SIZE_M)

    strategy = _make_strategy()
    folds = strategy.make_folds(X, y, groups=groups)

    test_sizes = [len(test_idx) for _, test_idx in folds]
    expected = len(y) / CV_FOLDS
    for size in test_sizes:
        assert abs(size - expected) / expected < 0.35, (
            f"fold test size {size} deviates too much from the balanced "
            f"target {expected:.0f} across sizes {test_sizes}"
        )
