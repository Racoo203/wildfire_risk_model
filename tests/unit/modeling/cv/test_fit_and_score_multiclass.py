# tests/unit/modeling/cv/test_fit_and_score_multiclass.py
"""Regression coverage for a bug in CVStrategy.fit_and_score (cv/base.py):
it called sklearn.precision_recall_curve directly on multiclass predict_proba
output. precision_recall_curve is binary-only, so on this project's 4-class
ordinal susceptibility target every call raised ValueError, was swallowed by
a bare `except Exception`, and every fold silently scored 0.0 — while the
docstring claimed it scored AUC (the real roc_auc_score call was dead,
commented out). This made StandardKFoldCV/SpatialGroupKFoldCV (the two
strategies that inherit fit_and_score from the base class rather than
overriding it) unusable for this multiclass problem: whenever cv_strategy
drove HPO through one of them, or trainer.py used one as the diagnostic
strategy under cv_strategy="both", the resulting "AUC" was always exactly
0.0, and the standard-vs-spatial optimism gap collapsed to
gap == the other side's value (not a real gap at all).

Fixed by making fit_and_score delegate to fit_and_score_full, which scores
AUC via roc_auc_score(multi_class="ovr") — the same multiclass-safe pattern
StratifiedSpatialBlockCV already used correctly."""

import numpy as np
import pandas as pd

from wildfire_susceptibility.modeling.cv.standard import StandardKFoldCV
from wildfire_susceptibility.modeling.cv.spatial import SpatialGroupKFoldCV
from wildfire_susceptibility.modeling.resampling import SMOTEResampler
from wildfire_susceptibility.modeling.models.random_forest import RandomForestModel

N_PER_CLASS = 60
N_CLASSES = 4


def _well_separated_4class_dataset():
    """Classes are cleanly separable by feature value, so a fit model should
    score well above 0.0 on every metric - if a strategy's fit_and_score
    still collapses to 0.0, that's the bug, not an artifact of a hard/noisy
    synthetic dataset."""
    rng = np.random.default_rng(0)
    rows, labels = [], []
    for cls in range(N_CLASSES):
        center = cls * 10.0
        for _ in range(N_PER_CLASS):
            rows.append([center + rng.normal(0, 0.5), rng.normal(0, 0.5)])
            labels.append(cls)
    X = pd.DataFrame(rows, columns=["f1", "f2"])
    y = pd.Series(labels)
    return X, y


def _make_strategy(cls, cv_folds=3):
    resampler = SMOTEResampler({"modeling": {"use_smote": False}})
    return cls({"modeling": {"cv_folds": cv_folds}}, resampler)


def _fold_indices(n_rows, n_folds=3):
    rng = np.random.default_rng(1)
    perm = rng.permutation(n_rows)
    folds = np.array_split(perm, n_folds)
    for i in range(n_folds):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        yield train_idx, test_idx


def test_standard_kfold_fit_and_score_does_not_collapse_to_zero_on_multiclass():
    X, y = _well_separated_4class_dataset()
    strategy = _make_strategy(StandardKFoldCV)

    train_idx, test_idx = next(_fold_indices(len(y)))
    score = strategy.fit_and_score(
        RandomForestModel, {"n_estimators": 50}, X, y, train_idx, test_idx, context="test",
    )

    assert score > 0.5, (
        f"fit_and_score returned {score} on a well-separated 4-class dataset - "
        f"expected a real AUC well above 0, not the silent 0.0 fallback from the "
        f"multiclass precision_recall_curve bug."
    )


def test_spatial_group_kfold_fit_and_score_does_not_collapse_to_zero_on_multiclass():
    X, y = _well_separated_4class_dataset()
    strategy = _make_strategy(SpatialGroupKFoldCV)

    train_idx, test_idx = next(_fold_indices(len(y)))
    score = strategy.fit_and_score(
        RandomForestModel, {"n_estimators": 50}, X, y, train_idx, test_idx, context="test",
    )

    assert score > 0.5


def test_fit_and_score_full_returns_matching_auc_and_all_three_metrics():
    X, y = _well_separated_4class_dataset()
    strategy = _make_strategy(StandardKFoldCV)
    train_idx, test_idx = next(_fold_indices(len(y)))

    full = strategy.fit_and_score_full(
        RandomForestModel, {"n_estimators": 50}, X, y, train_idx, test_idx, context="test",
    )

    assert set(full) == {"auc", "f1_macro", "pr_auc_macro"}
    assert all(v > 0.5 for v in full.values()), full


def test_fit_and_score_scalar_matches_fit_and_score_full_auc():
    """fit_and_score is documented as delegating to fit_and_score_full and
    returning just its "auc" key - lock that contract down so a future edit
    can't silently reintroduce two divergent scoring paths (which is exactly
    how the original bug happened: fit_and_score's docstring claimed AUC
    while its body computed something else)."""
    X, y = _well_separated_4class_dataset()
    strategy = _make_strategy(StandardKFoldCV)
    train_idx, test_idx = next(_fold_indices(len(y)))

    # Same params/model_cls, same fold split, same random_state inside
    # RandomForestModel.fit -> deterministic, so the two independent calls
    # (each fitting its own model) should score identically.
    scalar = strategy.fit_and_score(
        RandomForestModel, {"n_estimators": 50}, X, y, train_idx, test_idx, context="test",
    )
    full = strategy.fit_and_score_full(
        RandomForestModel, {"n_estimators": 50}, X, y, train_idx, test_idx, context="test",
    )

    assert scalar == full["auc"]
