"""Regression coverage for compute_full_metrics under n_classes != 4:
before class_labels.py existed, its default CLASS_NAMES was a hardcoded
4-name list, so `class_names[:n_classes]` silently truncated to 4 entries
for a 5-class run -- the highest class's precision/recall/f1/support/auc
metrics were never computed, with no exception and no warning."""

import numpy as np
import pytest

from wildfire_susceptibility.modeling.metrics import compute_full_metrics


def _synthetic_proba(n_classes: int, n: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, n_classes, size=n)
    logits = rng.normal(size=(n, n_classes))
    logits[np.arange(n), y_true] += 3.0  # bias toward the true class so AUC is well-defined
    proba = np.exp(logits)
    proba /= proba.sum(axis=1, keepdims=True)
    return y_true, proba


@pytest.mark.parametrize("n_classes,expected_last_key", [
    (3, "precision_high"),
    (5, "precision_very_high"),
])
def test_per_class_metrics_cover_every_class(n_classes, expected_last_key):
    y_true, y_proba = _synthetic_proba(n_classes)

    result = compute_full_metrics(y_true, y_proba)
    scalars = result["scalars"]

    assert expected_last_key in scalars
    assert f"support_{expected_last_key.split('_', 1)[1]}" in scalars

    total_support = sum(
        v for k, v in scalars.items() if k.startswith("support_")
    )
    assert total_support == len(y_true)


def test_confusion_matrix_is_full_size_regardless_of_name_list():
    y_true, y_proba = _synthetic_proba(5)
    result = compute_full_metrics(y_true, y_proba)
    cm = result["confusion_matrix"]
    assert len(cm) == 5
    assert all(len(row) == 5 for row in cm)
