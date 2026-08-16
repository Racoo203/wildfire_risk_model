# tests/unit/modeling/cv/test_score_multiclass_fold.py
"""Direct unit coverage for cv/base.py::score_multiclass_fold — the shared
scorer both CVStrategy.fit_and_score_full (base.py) and
StratifiedSpatialBlockCV.fit_and_score_full (stratified_spatial.py) now
delegate to.

Branch fix-spatial-cv-auc-missing-classes: this scorer replaces a pattern
that (1) built `labels=`/one-hot targets from the FULL configured class
list rather than the classes a fold's model actually saw at fit time, and
(2) caught the resulting shape mismatches and silently reported them as
`0.0` — indistinguishable from "the model genuinely scored zero." These
tests reproduce the exact failure shapes from the diagnosed smoke-test log
(`pipeline.log`, 08/15 run of baseline.yaml) directly against the pure
scoring function, without needing a full CV/model-fitting harness."""

import logging
import math

import numpy as np
import pytest

from wildfire_susceptibility.modeling.cv.base import score_multiclass_fold


def test_training_split_missing_three_of_five_classes_scores_a_real_value():
    """Mirrors the smoke-test's actual shape: 5 configured classes, but
    this fold's training split (and therefore predict_proba's column
    space) only ever saw 2 of them — the log's
    'Number of given labels, 5, not equal to the number of columns in
    y_score, 2' case. The validation split still contains all 5 classes
    (rows for the 3 missing ones are simply unscorable for AUC/PR-AUC, not
    the whole fold)."""
    y_tr = np.array([0] * 10 + [1] * 10)  # classes 2, 3, 4 never seen in training
    y_va = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    proba = np.zeros((10, 2))
    proba[:, 0] = [0.9, 0.8, 0.2, 0.1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    proba[:, 1] = 1 - proba[:, 0]

    result = score_multiclass_fold(y_va, proba, y_tr, context="test")

    assert not math.isnan(result["auc"])
    assert not math.isnan(result["pr_auc_macro"])
    assert 0.0 < result["auc"] <= 1.0
    assert 0.0 < result["pr_auc_macro"] <= 1.0
    assert result["auc"] != 0.0 and result["pr_auc_macro"] != 0.0


def test_predict_proba_fewer_columns_than_label_space_with_noncontiguous_trained_classes():
    """The second logged failure shape ('y should be a 1d array, got an
    array of shape (1290, 2) instead') plus a non-contiguous trained-class
    set (columns for classes {1, 3}, not {0, 1}) — the old
    np.eye(proba.shape[1])[y_va] one-hot construction would IndexError the
    moment a row's true label (e.g. 3) exceeded proba.shape[1]-1. Confirms
    this is handled by the same restrict-to-observed-classes path as the
    5-class case above, not a separate patch."""
    y_tr = np.array([1, 1, 1, 3, 3, 3])  # classes 0, 2, 4 never seen in training
    y_va = np.array([1, 1, 3, 3, 0, 2, 4])
    proba = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.1, 0.9],
        [0.5, 0.5],
        [0.5, 0.5],
        [0.5, 0.5],
    ])

    result = score_multiclass_fold(y_va, proba, y_tr, context="test")

    assert not math.isnan(result["auc"])
    assert not math.isnan(result["pr_auc_macro"])
    assert result["auc"] > 0.5, "the two scorable rows are unambiguous; restricted AUC should reflect that"


def test_genuinely_degenerate_fold_scores_nan_not_zero_and_logs_a_warning(caplog):
    """Zero classes in common between validation ground truth and what the
    model can score: AUC/PR-AUC are mathematically undefined, not just
    hard to compute. Must produce NaN (a clear 'unscorable', not a
    misleading '0.0' that reads as a real, terrible score) plus a logged
    warning naming the fold. F1-macro/QWK don't need this restriction and
    must still compute a real, finite value from the raw labels."""
    y_tr = np.array([0, 0, 1, 1])
    y_va = np.array([2, 2, 3, 3])  # shares nothing with trained_classes {0, 1}
    proba = np.array([[0.6, 0.4], [0.5, 0.5], [0.4, 0.6], [0.3, 0.7]])

    with caplog.at_level(logging.WARNING):
        result = score_multiclass_fold(y_va, proba, y_tr, context="[test] degenerate fold")

    assert math.isnan(result["auc"])
    assert math.isnan(result["pr_auc_macro"])
    assert not math.isnan(result["f1_macro"])
    assert not math.isnan(result["qwk"])
    assert any(
        "fewer than 2 classes in common" in r.message and "[test] degenerate fold" in r.message
        for r in caplog.records
    ), f"expected a clear degenerate-fold warning naming the context, got: {[r.message for r in caplog.records]}"


def test_exactly_two_observed_classes_does_not_hit_sklearns_binary_routing_pitfall():
    """When a fold happens to end up with exactly 2 classes in common
    between trained and validation, sklearn's roc_auc_score silently
    reroutes to its binary code path regardless of multi_class='ovr',
    which expects a 1D positive-class score — the exact 'y should be a 1d
    array, got shape (1290, 2)' error from the log. This must not raise
    and must not fall back through the try/except's NaN path."""
    y_tr = np.array([0, 0, 1, 1, 1])
    y_va = np.array([0, 0, 1, 1, 1])
    proba = np.array([
        [0.9, 0.1],
        [0.8, 0.2],
        [0.2, 0.8],
        [0.3, 0.7],
        [0.1, 0.9],
    ])

    result = score_multiclass_fold(y_va, proba, y_tr, context="test")

    assert not math.isnan(result["auc"])
    assert result["auc"] == pytest.approx(1.0)  # perfectly separated by construction


def test_neural_net_gapless_column_convention_is_inferred_from_shape():
    """The neural_net wrapper (ann.py: n_classes = max(y) + 1) always
    builds a gapless 0..max(y_tr) output layer, so a missing MIDDLE class
    leaves a real (untrained) column in place rather than shrinking the
    column count — the opposite convention from RF/CatBoost/mord, where a
    missing class shrinks proba.shape[1]. score_multiclass_fold must infer
    this purely from shape (proba.shape[1] != len(unique(y_tr))) and use
    range(proba.shape[1]) as the column->label mapping, not
    sorted(unique(y_tr))."""
    y_tr = np.array([0, 0, 1, 1, 4, 4])  # classes present: {0, 1, 4}; NN still builds 5 columns (max=4 -> n_classes=5)
    proba = np.zeros((3, 5))
    proba[0] = [0.9, 0.02, 0.02, 0.02, 0.04]  # confident class 0
    proba[1] = [0.02, 0.9, 0.02, 0.02, 0.04]  # confident class 1
    proba[2] = [0.02, 0.02, 0.02, 0.02, 0.92]  # confident class 4 (column index 4)
    y_va = np.array([0, 1, 4])

    result = score_multiclass_fold(y_va, proba, y_tr, context="test")

    # If the scorer wrongly assumed sorted(unique(y_tr)) = [0, 1, 4] mapped
    # onto proba's 5 columns positionally, column index 2 would be
    # mislabeled as class 4 and this would misclassify/misalign scoring.
    assert not math.isnan(result["auc"])
    assert result["f1_macro"] == pytest.approx(1.0)
    assert result["qwk"] == pytest.approx(1.0)


def test_pred_label_mapping_fix_corrects_f1_and_qwk_for_a_missing_middle_class():
    """Root-cause companion to the AUC/PR-AUC fix: pred = argmax(proba) is
    a COLUMN POSITION, not a class label. For RF/CatBoost/mord, whenever a
    fold's training split is missing a non-maximal class (trained_classes
    = [0, 1, 4], not gapless), the old code compared this raw position
    directly against y_va — e.g. a row the model correctly scored highest
    for class 4 (column index 2) would have been reported as predicted
    class '2', silently corrupting F1-macro/QWK with no exception (so it
    never appeared in the original 'scoring as 0.0' log lines)."""
    y_tr = np.array([0, 0, 1, 1, 4, 4])  # missing classes 2, 3 (middle of the label space)
    proba = np.array([
        [0.9, 0.05, 0.05],   # column 0 -> class 0
        [0.05, 0.9, 0.05],   # column 1 -> class 1
        [0.05, 0.05, 0.9],   # column 2 -> class 4 (NOT class 2 — this is the bug this test locks down)
    ])
    y_va = np.array([0, 1, 4])

    result = score_multiclass_fold(y_va, proba, y_tr, context="test")

    assert result["f1_macro"] == pytest.approx(1.0), (
        "pred must be mapped through trained_classes (column 2 -> label 4), "
        "not used as a raw column position (which would silently score this as a total miss)"
    )
    assert result["qwk"] == pytest.approx(1.0)
    assert not math.isnan(result["auc"])
    assert result["auc"] == pytest.approx(1.0)
