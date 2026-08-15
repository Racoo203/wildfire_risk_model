# tests/unit/modeling/cv/test_standard.py
"""Direct coverage for StandardKFoldCV (cv/standard.py) — the deliberately
naive, non-spatial baseline kept to reproduce Schratz et al. (2019)'s
overoptimism-gap demonstration (see the module docstring in standard.py).
This strategy previously only had indirect coverage (via
test_cv_strategy_independence.py, test_imbalance_wiring.py, and
test_fit_and_score_multiclass.py) — none of which pin down make_folds()'s
own basic contract: deterministic given a fixed random_state, no
train/test overlap within a fold, and a fold count matching cv_folds."""

import numpy as np
import pandas as pd

from wildfire_susceptibility.modeling.cv.standard import StandardKFoldCV
from wildfire_susceptibility.modeling.resampling import SMOTEResampler


def _synthetic_dataset(n=80, n_classes=4):
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["f1", "f2", "f3"])
    y = pd.Series(np.tile(np.arange(n_classes), n // n_classes))
    return X, y


def _make_strategy(cv_folds=5):
    resampler = SMOTEResampler({"modeling": {"use_smote": False}})
    return StandardKFoldCV({"modeling": {"cv_folds": cv_folds}}, resampler)


def test_make_folds_is_deterministic_across_calls():
    """random_state is fixed (hardcoded 42 in standard.py), so repeated
    calls on the same inputs must yield byte-identical fold assignments —
    this is what makes the standard-vs-spatial optimism-gap comparison in
    trainer.py reproducible across runs, not just within one process."""
    X, y = _synthetic_dataset()
    strategy = _make_strategy()

    folds_a = strategy.make_folds(X, y)
    folds_b = strategy.make_folds(X, y)

    assert len(folds_a) == len(folds_b)
    for (train_a, test_a), (train_b, test_b) in zip(folds_a, folds_b):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)


def test_no_train_test_overlap_within_a_fold():
    X, y = _synthetic_dataset()
    strategy = _make_strategy()

    for train_idx, test_idx in strategy.make_folds(X, y):
        assert set(train_idx).isdisjoint(set(test_idx)), (
            "a row appeared in both train and test for the same fold"
        )
        # every row accounted for between the two splits
        assert len(train_idx) + len(test_idx) == len(y)


def test_fold_count_matches_cv_folds_config():
    X, y = _synthetic_dataset()
    for cv_folds in (3, 4, 5):
        strategy = _make_strategy(cv_folds=cv_folds)
        folds = strategy.make_folds(X, y)
        assert len(folds) == cv_folds


def test_every_row_appears_in_exactly_one_test_split():
    """Across all folds combined (not just within one), the test splits
    should partition the full dataset - each row tested exactly once,
    consistent with StratifiedKFold's contract."""
    X, y = _synthetic_dataset()
    strategy = _make_strategy()

    row_test_count = np.zeros(len(y), dtype=int)
    for _, test_idx in strategy.make_folds(X, y):
        row_test_count[test_idx] += 1

    assert (row_test_count == 1).all()
