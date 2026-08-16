# tests/unit/modeling/cv/test_stratified_spatial_block.py
"""Regression coverage for StratifiedSpatialBlockCV - the dissertation's
headline CV strategy. Had zero direct test coverage before this file;
every integration test forces cv_strategy='standard' to sidestep it."""

import numpy as np
import pandas as pd
import pytest

from wildfire_susceptibility.core.registry import MODELS
from wildfire_susceptibility.modeling import models as _models  # noqa: F401 — registers wrappers
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


def test_fit_and_score_full_survives_a_fold_that_lost_a_class_in_training():
    """Regression test: with a small/adjacent-block rare class (e.g. class
    4 confined to two neighboring blocks in real data), spatial buffering
    can strip a class out of a fold's training split entirely while the
    fold's own validation split - built from the untouched held-out
    blocks - still contains it. predict_proba then returns fewer columns
    than the full label space, and building the one-hot y_va_bin as
    np.eye(proba.shape[1])[y_va] raised IndexError uncaught (unlike
    cv/base.py's fit_and_score_full, which already guards this same
    average_precision_score call with try/except). This must degrade to a
    logged warning + pr_auc_macro=0.0, not crash the whole Optuna trial."""
    rng = np.random.default_rng(7)
    n = 60
    X = pd.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n)})
    y = pd.Series([0, 1] * (n // 2))

    # Training split never sees class 2 at all; validation split does -
    # mirrors a fold whose training blocks lost the rare class to buffering
    # while its own test blocks (never buffered) retained it.
    train_idx = np.arange(0, 40)
    test_idx = np.arange(40, 60)
    y.iloc[test_idx[:3]] = 2

    resampler = SMOTEResampler({"modeling": {"use_smote": False}})
    strategy = StratifiedSpatialBlockCV({"modeling": {"cv_folds": 4}}, resampler)

    result = strategy.fit_and_score_full(
        MODELS["random_forest"], {}, X, y, train_idx, test_idx, context="test",
    )

    assert result["pr_auc_macro"] == 0.0
    assert set(result.keys()) == {"auc", "f1_macro", "pr_auc_macro", "qwk"}


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


def _worked_example_blocks_and_labels():
    """The exact six-block, one-rare-block example walked through in
    stratified_spatial.py's module docstring (blocks A-F, class 2 confined
    to block C) - kept here so that example is executed as a real
    assertion, not just prose a reader has to trust."""
    per_block_class_counts = {
        "A": {0: 1, 1: 2, 2: 0},
        "B": {0: 2, 1: 1, 2: 0},
        "C": {0: 0, 1: 0, 2: 3},
        "D": {0: 1, 1: 2, 2: 0},
        "E": {0: 3, 1: 0, 2: 0},
        "F": {0: 0, 1: 3, 2: 0},
    }
    blocks, labels = [], []
    for block, counts in per_block_class_counts.items():
        for cls, n in counts.items():
            blocks.extend([block] * n)
            labels.extend([cls] * n)
    return pd.Series(blocks), pd.Series(labels)


def test_count_rows_per_block_and_class_matches_worked_example():
    groups, y = _worked_example_blocks_and_labels()
    counts = StratifiedSpatialBlockCV._count_rows_per_block_and_class(groups, y)

    assert counts["C"] == {0: 0, 1: 0, 2: 3}
    assert counts["E"] == {0: 3, 1: 0, 2: 0}
    assert set(counts.keys()) == {"A", "B", "C", "D", "E", "F"}


def test_seed_every_fold_with_rarest_class_coverage_places_the_only_rare_block():
    """Direct unit test of Pass 1 in isolation: block C is the sole carrier
    of the rarest class (2), so it must be the very first block seeded,
    landing in fold 0 (the first slot in round-robin order)."""
    groups, y = _worked_example_blocks_and_labels()
    block_class_counts = StratifiedSpatialBlockCV._count_rows_per_block_and_class(groups, y)
    rarity_order = y.value_counts().sort_values().index.tolist()
    assert rarity_order == [2, 0, 1]

    fold_of_block, fold_class_counts = StratifiedSpatialBlockCV._seed_every_fold_with_rarest_class_coverage(
        block_class_counts, rarity_order, n_folds=3,
    )

    assert fold_of_block["C"] == 0
    assert fold_class_counts[0][2] == 3
    # fold_class_counts only reflects what's actually been placed so far
    assert sum(sum(f.values()) for f in fold_class_counts) == sum(
        sum(block_class_counts[b].values()) for b in fold_of_block
    )


def test_greedily_fill_remaining_blocks_places_every_leftover_block():
    """Direct unit test of Pass 2 in isolation, continuing from Pass 1's
    output: every block Pass 1 left unassigned must end up placed exactly
    once, without Pass 2 touching what Pass 1 already decided.

    Uses the larger random block fixture (not the six-block worked
    example) because the worked example is small enough that Pass 1 places
    every block by itself, leaving Pass 2 nothing to do - a test built on
    it would pass even if Pass 2's fill loop were completely broken."""
    rng = np.random.default_rng(5)
    groups, y = _synthetic_blocks_and_labels(rng, n_blocks=40, n_rare_blocks=4)
    block_class_counts = StratifiedSpatialBlockCV._count_rows_per_block_and_class(groups, y)
    rarity_order = y.value_counts().sort_values().index.tolist()

    fold_of_block, fold_class_counts = StratifiedSpatialBlockCV._seed_every_fold_with_rarest_class_coverage(
        block_class_counts, rarity_order, n_folds=4,
    )
    seeded_placements = dict(fold_of_block)  # snapshot before Pass 2 extends it
    # precondition: Pass 1 must actually leave blocks unplaced, or this
    # test isn't exercising Pass 2's fill loop at all.
    assert len(seeded_placements) < len(block_class_counts)

    StratifiedSpatialBlockCV._greedily_fill_remaining_blocks(
        block_class_counts, rarity_order, n_folds=4,
        fold_of_block=fold_of_block, fold_class_counts=fold_class_counts,
    )

    assert set(fold_of_block.keys()) == set(block_class_counts.keys())
    for block, fold in seeded_placements.items():
        assert fold_of_block[block] == fold, "Pass 2 reassigned a block Pass 1 already placed"
    for c in rarity_order:
        assert sum(f[c] for f in fold_class_counts) == int((y == c).sum())
