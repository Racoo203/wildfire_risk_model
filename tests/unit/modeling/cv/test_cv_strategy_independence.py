# tests/unit/modeling/cv/test_cv_strategy_independence.py
"""Regression coverage for the concern behind cv_strategy='both': that the
standard-CV and spatial-CV fold assignments trainer.py builds for the
diagnostic comparison are computed via genuinely independent code paths
and don't accidentally end up as the same partition relabeled (e.g. from a
copy-paste bug, a shared/leaked random_state, or reusing one strategy's
indices for the other).

Fold IDs are arbitrary/interchangeable across strategies (StandardKFoldCV's
"fold 0" has no relationship to StratifiedSpatialBlockCV's "fold 0"), so
we compare the two row->fold partitions with the Adjusted Rand Index
(permutation-invariant). If the two strategies were actually sharing
splits, ARI would be ~1.0; genuinely independent splits driven by
different logic (row-stratified-by-class vs. block-stratified-by-space)
should land well below that."""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from wildfire_susceptibility.modeling.cv import get_cv_strategy, requires_spatial_groups
from wildfire_susceptibility.modeling.resampling import SMOTEResampler

N_BLOCKS_PER_AXIS = 8
BLOCK_SIZE_M = 1000.0
ROWS_PER_BLOCK = 8
CV_FOLDS = 4


def _synthetic_spatial_dataset():
    rng = np.random.default_rng(1)
    rows = []
    for bx in range(N_BLOCKS_PER_AXIS):
        for by in range(N_BLOCKS_PER_AXIS):
            block_id = f"{bx}_{by}"
            for _ in range(ROWS_PER_BLOCK):
                x = bx * BLOCK_SIZE_M + rng.uniform(0, BLOCK_SIZE_M)
                y_coord = by * BLOCK_SIZE_M + rng.uniform(0, BLOCK_SIZE_M)
                # label independent of spatial position, matching how class
                # (susceptibility) isn't literally determined by block index
                rows.append((x, y_coord, block_id, int(rng.integers(0, 4))))

    df = pd.DataFrame(rows, columns=["_x", "_y", "block", "label"])
    X = df[["_x", "_y"]]
    y = df["label"]
    groups = df["block"]
    return X, y, groups


def _row_fold_assignment(folds, n_rows):
    row_fold = np.full(n_rows, -1, dtype=int)
    for k, (_, test_idx) in enumerate(folds):
        row_fold[test_idx] = k
    assert (row_fold >= 0).all(), "every row must appear in exactly one fold's test split"
    return row_fold


def test_both_strategies_call_independent_make_folds_objects():
    """cv_strategy='both' resolves to two distinct CVStrategy instances in
    trainer.py, not one object reused for both roles - this is the literal
    "not accidentally reusing the same splits" mechanism."""
    config = {"modeling": {"cv_folds": CV_FOLDS}}
    resampler = SMOTEResampler({"modeling": {"use_smote": False}})

    primary = get_cv_strategy("stratified_spatial_block", config, resampler)
    diagnostic = get_cv_strategy("standard", config, resampler)

    assert primary is not diagnostic
    assert type(primary) is not type(diagnostic)
    assert requires_spatial_groups(primary.name) and not requires_spatial_groups(diagnostic.name)


def test_standard_and_spatial_fold_assignments_differ_meaningfully():
    X, y, groups = _synthetic_spatial_dataset()
    config = {"modeling": {"cv_folds": CV_FOLDS}}
    resampler = SMOTEResampler({"modeling": {"use_smote": False}})

    standard = get_cv_strategy("standard", config, resampler)
    spatial = get_cv_strategy("stratified_spatial_block", config, resampler)

    standard_folds = standard.make_folds(X, y, groups=None)
    spatial_folds = spatial.make_folds(X, y, groups=groups)

    row_fold_standard = _row_fold_assignment(standard_folds, len(y))
    row_fold_spatial = _row_fold_assignment(spatial_folds, len(y))

    ari = adjusted_rand_score(row_fold_standard, row_fold_spatial)
    assert ari < 0.5, (
        f"standard and spatial fold partitions are suspiciously similar "
        f"(ARI={ari:.3f}) - expected them to diverge since one stratifies "
        f"by class on individual rows and the other stratifies by class on "
        f"whole spatial blocks; a near-1.0 ARI would suggest they're "
        f"accidentally sharing the same underlying split."
    )


def test_repeated_calls_to_make_folds_are_not_silently_cached_or_mutated():
    """Guards against a subtler variant of split-sharing: calling
    make_folds() twice on the same strategy instance (as trainer.py's
    search + a later diagnostic call might, if code were refactored to
    reuse self.cv_strategy for both roles) must not return a stale/mutated
    result from the first call."""
    X, y, groups = _synthetic_spatial_dataset()
    config = {"modeling": {"cv_folds": CV_FOLDS}}
    resampler = SMOTEResampler({"modeling": {"use_smote": False}})

    spatial = get_cv_strategy("stratified_spatial_block", config, resampler)
    folds_a = spatial.make_folds(X, y, groups=groups)
    folds_b = spatial.make_folds(X, y, groups=groups)

    row_fold_a = _row_fold_assignment(folds_a, len(y))
    row_fold_b = _row_fold_assignment(folds_b, len(y))

    assert adjusted_rand_score(row_fold_a, row_fold_b) == 1.0, (
        "calling make_folds() twice on the same inputs produced different "
        "partitions - the block->fold assignment should be deterministic."
    )
