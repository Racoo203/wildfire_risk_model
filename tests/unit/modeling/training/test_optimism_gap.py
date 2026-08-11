# tests/unit/modeling/training/test_optimism_gap.py
"""compute_optimism_gap is the arithmetic core of the standard-vs-spatial
optimism gap logged by trainer.py's _log_optimism_gap: gap = standard -
spatial, per metric, computed from fixed synthetic values so the sign
convention and per-metric independence are locked down without needing to
run any actual CV fold or fit a model."""

import pytest

from wildfire_susceptibility.modeling.training.optimism_gap import compute_optimism_gap


def test_gap_is_standard_minus_spatial_for_every_metric():
    standard = {"auc": 0.90, "f1_macro": 0.80, "pr_auc_macro": 0.70}
    spatial = {"auc": 0.75, "f1_macro": 0.65, "pr_auc_macro": 0.60}

    gap = compute_optimism_gap(standard, spatial)

    assert gap["auc"] == pytest.approx(0.15)
    assert gap["f1_macro"] == pytest.approx(0.15)
    assert gap["pr_auc_macro"] == pytest.approx(0.10)


def test_gap_is_zero_when_standard_equals_spatial():
    values = {"auc": 0.82, "f1_macro": 0.77, "pr_auc_macro": 0.71}
    gap = compute_optimism_gap(values, dict(values))
    assert all(v == pytest.approx(0.0) for v in gap.values())


def test_gap_can_be_negative_when_spatial_scores_higher():
    """Not the expected direction (spatial CV is usually the more
    conservative estimate), but the arithmetic itself must not clip or
    assume a sign - a negative gap is a legitimate, reportable result."""
    standard = {"auc": 0.70, "f1_macro": 0.60, "pr_auc_macro": 0.55}
    spatial = {"auc": 0.80, "f1_macro": 0.60, "pr_auc_macro": 0.50}

    gap = compute_optimism_gap(standard, spatial)

    assert gap["auc"] == pytest.approx(-0.10)
    assert gap["f1_macro"] == pytest.approx(0.0)
    assert gap["pr_auc_macro"] == pytest.approx(0.05)


def test_gap_respects_custom_metric_subset():
    standard = {"auc": 0.9, "f1_macro": 0.8, "pr_auc_macro": 0.7, "extra": 1.0}
    spatial = {"auc": 0.6, "f1_macro": 0.5, "pr_auc_macro": 0.4, "extra": 0.0}

    gap = compute_optimism_gap(standard, spatial, metrics=("auc",))

    assert gap == {"auc": pytest.approx(0.3)}


def test_gap_raises_on_missing_metric_key():
    standard = {"auc": 0.9, "f1_macro": 0.8}  # missing pr_auc_macro
    spatial = {"auc": 0.6, "f1_macro": 0.5, "pr_auc_macro": 0.4}

    with pytest.raises(KeyError):
        compute_optimism_gap(standard, spatial)
