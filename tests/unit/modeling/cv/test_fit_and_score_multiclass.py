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
import pytest

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


def test_fit_and_score_full_returns_matching_auc_and_all_four_metrics():
    X, y = _well_separated_4class_dataset()
    strategy = _make_strategy(StandardKFoldCV)
    train_idx, test_idx = next(_fold_indices(len(y)))

    full = strategy.fit_and_score_full(
        RandomForestModel, {"n_estimators": 50}, X, y, train_idx, test_idx, context="test",
    )

    assert set(full) == {"auc", "f1_macro", "pr_auc_macro", "qwk"}
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


class _FixedProbaModel:
    """Ignores training data entirely and always predicts a caller-supplied
    probability matrix - lets a test pin fit_and_score_full's metrics
    against known, hand-crafted predictions instead of whatever a real
    fitted model happens to produce."""

    _proba = None  # set per-test via _FixedProbaModel.set_proba(...)

    def __init__(self, **_params):
        pass

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        assert len(X) == len(self._proba), "test fixture size mismatch"
        return self._proba

    @classmethod
    def set_proba(cls, proba: np.ndarray):
        cls._proba = proba


# 8 known test-fold rows: true label, predicted-argmax label, and the miss
# distance (0 = correct, 1 = adjacent-class miss, 3 = maximally-far miss) -
# a mix so PR-AUC/F1-macro/QWK are all exercised on non-trivial predictions.
_KNOWN_Y_VA = np.array([0, 1, 2, 3, 0, 1, 2, 3])
_KNOWN_PROBA = np.array([
    [0.90, 0.05, 0.03, 0.02],  # true 0, pred 0 (correct)
    [0.05, 0.85, 0.05, 0.05],  # true 1, pred 1 (correct)
    [0.10, 0.60, 0.20, 0.10],  # true 2, pred 1 (adjacent miss)
    [0.05, 0.05, 0.10, 0.80],  # true 3, pred 3 (correct)
    [0.10, 0.10, 0.10, 0.70],  # true 0, pred 3 (far miss)
    [0.10, 0.70, 0.10, 0.10],  # true 1, pred 1 (correct)
    [0.05, 0.10, 0.75, 0.10],  # true 2, pred 2 (correct)
    [0.10, 0.10, 0.60, 0.20],  # true 3, pred 2 (adjacent miss)
])


def test_fit_and_score_full_matches_independently_computed_metrics_on_known_predictions():
    """Pins fit_and_score_full's PR-AUC-macro/F1-macro/QWK against the same
    sklearn calls applied directly to a known, hand-crafted prediction
    fixture - catches wiring bugs (wrong axis for the one-hot encoding,
    wrong `pred`/`y_va` array passed to the wrong metric, forgetting
    weights="quadratic", etc.) that a property-only test (">0.5") can't."""
    from sklearn.metrics import f1_score, average_precision_score, cohen_kappa_score

    X = pd.DataFrame({"f1": np.zeros(len(_KNOWN_Y_VA)), "f2": np.zeros(len(_KNOWN_Y_VA))})
    y = pd.Series(_KNOWN_Y_VA)
    strategy = _make_strategy(StandardKFoldCV)
    _FixedProbaModel.set_proba(_KNOWN_PROBA)

    train_idx = np.array([0, 1])  # unused by _FixedProbaModel.fit, just needs to be valid indices
    test_idx = np.arange(len(_KNOWN_Y_VA))

    full = strategy.fit_and_score_full(
        _FixedProbaModel, {}, X, y, train_idx, test_idx, context="test",
    )

    pred = np.argmax(_KNOWN_PROBA, axis=1)
    y_va_bin = np.eye(_KNOWN_PROBA.shape[1])[_KNOWN_Y_VA]
    expected_f1_macro = float(f1_score(_KNOWN_Y_VA, pred, average="macro"))
    expected_pr_auc_macro = float(average_precision_score(y_va_bin, _KNOWN_PROBA, average="macro"))
    expected_qwk = float(cohen_kappa_score(_KNOWN_Y_VA, pred, weights="quadratic"))

    assert full["f1_macro"] == pytest.approx(expected_f1_macro)
    assert full["pr_auc_macro"] == pytest.approx(expected_pr_auc_macro)
    assert full["qwk"] == pytest.approx(expected_qwk)


def test_qwk_penalizes_far_misclassification_more_than_adjacent():
    """Ordinal-awareness check motivating QWK's addition: for the same
    number of wrong predictions, a far miss (class 0 predicted as class 3)
    must score worse (lower QWK) than an adjacent miss (class 0 predicted
    as class 1) - unlike unweighted Cohen's kappa or F1-macro, which don't
    distinguish "how wrong" a wrong prediction is."""
    from sklearn.metrics import cohen_kappa_score, f1_score

    y_true = np.array([0, 1, 2, 3])
    y_pred_adjacent_miss = np.array([0, 2, 2, 3])  # true=1 predicted as 2 (distance 1)
    y_pred_far_miss = np.array([0, 3, 2, 3])       # true=1 predicted as 3 (distance 2)

    qwk_adjacent = cohen_kappa_score(y_true, y_pred_adjacent_miss, weights="quadratic")
    qwk_far = cohen_kappa_score(y_true, y_pred_far_miss, weights="quadratic")

    assert qwk_adjacent > qwk_far, (
        f"QWK should penalize the far miss more than the adjacent miss "
        f"(adjacent={qwk_adjacent}, far={qwk_far}), otherwise it isn't adding "
        f"anything over unweighted accuracy/F1 for this ordinal target."
    )

    # Unweighted F1-macro, by contrast, is legitimately blind to *how far*
    # a misclassification is - both fixtures have exactly one wrong
    # prediction on class 1, so their F1-macro should match. This isn't a
    # flaw to fix; it's precisely the gap QWK is added to cover.
    f1_adjacent = f1_score(y_true, y_pred_adjacent_miss, average="macro")
    f1_far = f1_score(y_true, y_pred_far_miss, average="macro")
    assert f1_adjacent == pytest.approx(f1_far)
