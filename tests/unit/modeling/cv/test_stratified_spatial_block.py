# tests/unit/modeling/cv/test_stratified_spatial_block.py
"""Regression coverage for StratifiedSpatialBlockCV - the dissertation's
headline CV strategy. Had zero direct test coverage before this file;
every integration test forces cv_strategy='standard' to sidestep it."""

import numpy as np
import pandas as pd
import pytest

from wildfire_susceptibility.modeling.cv.stratified_spatial import StratifiedSpatialBlockCV
from wildfire_susceptibility.modeling.resampling import SMOTEResampler


def _make_strategy(cv_folds=4):
    resampler = SMOTEResampler({"modeling": {"use_smote": False}})
    return StratifiedSpatialBlockCV({"modeling": {"cv_folds": cv_folds}}, resampler)


def _synthetic_blocks_and_labels(rng, n_blocks=40, rows_per_block=10, n_rare_blocks=4):
    """Most blocks are class 0/1 only; a handful of blocks carry the rare
    class 2, mimicking a scarce 'very high' susceptibility class confined
    to few spatial blocks. The first two rows of each rare block are
    deterministically class 2, so the rare class's presence never depends
    on random draw (avoiding a flaky "rare class missing from this block"
    failure unrelated to what's under test)."""
    groups, y = [], []
    for b in range(n_blocks):
        block_id = f"blk_{b}"
        is_rare_block = b < n_rare_blocks
        for i in range(rows_per_block):
            groups.append(block_id)
            if is_rare_block and i < 2:
                y.append(2)
            else:
                y.append(int(rng.integers(0, 2)))
    return pd.Series(groups), pd.Series(y)


def test_every_fold_gets_the_rare_class_represented():
    rng = np.random.default_rng(3)
    groups, y = _synthetic_blocks_and_labels(rng)
    X = pd.DataFrame({"f": np.zeros(len(y))})

    strategy = _make_strategy(cv_folds=4)
    folds = strategy.make_folds(X, y, groups=groups)

    assert len(folds) == 4
    for _, test_idx in folds:
        assert 2 in set(y.iloc[test_idx]), "a fold's validation split is missing the rare class"


def test_blocks_are_never_split_across_folds():
    """The CV unit is the block, not the row - every row belonging to a
    given block must land in exactly one fold's test split."""
    rng = np.random.default_rng(5)
    groups, y = _synthetic_blocks_and_labels(rng)
    X = pd.DataFrame({"f": np.zeros(len(y))})

    strategy = _make_strategy(cv_folds=4)
    folds = strategy.make_folds(X, y, groups=groups)

    block_to_folds = {}
    for fold_k, (_, test_idx) in enumerate(folds):
        for row in test_idx:
            block = groups.iloc[row]
            block_to_folds.setdefault(block, set()).add(fold_k)

    split_blocks = {b: folds_seen for b, folds_seen in block_to_folds.items() if len(folds_seen) > 1}
    assert not split_blocks, f"blocks assigned to more than one fold: {split_blocks}"


def test_train_and_test_never_overlap_within_a_fold():
    rng = np.random.default_rng(9)
    groups, y = _synthetic_blocks_and_labels(rng)
    X = pd.DataFrame({"f": np.zeros(len(y))})

    strategy = _make_strategy(cv_folds=4)
    folds = strategy.make_folds(X, y, groups=groups)

    for train_idx, test_idx in folds:
        assert set(train_idx).isdisjoint(set(test_idx))


def test_make_folds_raises_without_groups():
    strategy = _make_strategy()
    X = pd.DataFrame({"f": [0, 1, 2]})
    y = pd.Series([0, 1, 0])
    with pytest.raises(ValueError):
        strategy.make_folds(X, y, groups=None)


def test_stratify_blocks_into_folds_gives_every_fold_the_rarest_class():
    """Direct unit test of the block-assignment algorithm itself (Pass 1 /
    Pass 2 in _stratify_blocks_into_folds), independent of make_folds()'s
    row bookkeeping."""
    rng = np.random.default_rng(11)
    groups, y = _synthetic_blocks_and_labels(rng, n_blocks=60, n_rare_blocks=6)

    fold_of_block = StratifiedSpatialBlockCV._stratify_blocks_into_folds(groups, y, n_folds=5)

    # every block gets assigned to exactly one fold in [0, n_folds)
    assert set(fold_of_block.keys()) == set(groups.unique())
    assert all(0 <= f < 5 for f in fold_of_block.values())

    rare_blocks = [b for b, cls in zip(groups, y) if cls == 2]
    rare_folds = {fold_of_block[b] for b in rare_blocks}
    assert len(rare_folds) > 1, "the rare class's blocks were all dumped into a single fold"
