"""Unit tests for stage_selection's pure aggregation logic
(_select_best_per_season, _kruskal_wallis_per_season). These touch no
I/O beyond what the caller already provides as in-memory manifest
dicts, so they're tested directly against fabricated manifests rather
than requiring real trained artifacts on disk."""

import pytest

from wildfire_susceptibility.pipeline.stage_selection import (
    _select_best_per_season,
    _kruskal_wallis_per_season,
)


def _manifest(val_auc, cv_auc_standard, cv_auc_spatial, folds=None):
    return {
        "val_auc": val_auc,
        "val_f1": 0.5,
        "cv_auc_standard": cv_auc_standard,
        "cv_auc_spatial": cv_auc_spatial,
        "cv_auc_spatial_folds": folds,
    }


def test_select_best_auc_picks_highest_val_auc():
    manifests = {
        "summer": {
            "random_forest": _manifest(val_auc=0.85, cv_auc_standard=0.80, cv_auc_spatial=0.75),
            "svm": _manifest(val_auc=0.90, cv_auc_standard=0.82, cv_auc_spatial=0.78),
        }
    }
    result = _select_best_per_season(manifests, "best_auc")
    assert result["summer"]["selected_model"] == "svm"
    assert result["summer"]["val_auc"] == 0.90


def test_select_most_conservative_picks_smallest_optimism_gap_within_threshold():
    manifests = {
        "summer": {
            # standard-auc leader, but big overfit gap (0.90 - 0.60 = 0.30)
            "xgboost": _manifest(val_auc=0.88, cv_auc_standard=0.90, cv_auc_spatial=0.60),
            # within 1% of leader (0.895 >= 0.90 - 0.01), much smaller gap (0.895-0.85=0.045)
            "random_forest": _manifest(val_auc=0.86, cv_auc_standard=0.895, cv_auc_spatial=0.85),
            # not within 1% of leader (0.85 < 0.89), excluded from candidates
            "svm": _manifest(val_auc=0.80, cv_auc_standard=0.85, cv_auc_spatial=0.84),
        }
    }
    result = _select_best_per_season(manifests, "most_conservative")
    assert result["summer"]["selected_model"] == "random_forest"


def test_select_unknown_rule_raises():
    manifests = {"summer": {"rf": _manifest(0.8, 0.8, 0.7)}}
    with pytest.raises(ValueError, match="Unknown selection_rule"):
        _select_best_per_season(manifests, "coin_flip")


def test_select_per_season_independent():
    manifests = {
        "summer": {
            "rf": _manifest(val_auc=0.9, cv_auc_standard=0.9, cv_auc_spatial=0.85),
            "svm": _manifest(val_auc=0.7, cv_auc_standard=0.7, cv_auc_spatial=0.65),
        },
        "spring": {
            "rf": _manifest(val_auc=0.6, cv_auc_standard=0.6, cv_auc_spatial=0.55),
            "svm": _manifest(val_auc=0.95, cv_auc_standard=0.95, cv_auc_spatial=0.90),
        },
    }
    result = _select_best_per_season(manifests, "best_auc")
    assert result["summer"]["selected_model"] == "rf"
    assert result["spring"]["selected_model"] == "svm"


def test_kruskal_wallis_detects_no_difference_between_identical_groups():
    manifests = {
        "summer": {
            "rf": _manifest(0.8, 0.8, 0.75, folds=[0.75, 0.76, 0.74, 0.75, 0.75]),
            "svm": _manifest(0.8, 0.8, 0.75, folds=[0.75, 0.76, 0.74, 0.75, 0.75]),
        }
    }
    result = _kruskal_wallis_per_season(manifests)
    assert result["summer"]["p_value"] > 0.99
    assert result["summer"]["significant_at_0.05"] is False


def test_kruskal_wallis_detects_clear_difference_between_separated_groups():
    manifests = {
        "summer": {
            "rf": _manifest(0.9, 0.9, 0.9, folds=[0.90, 0.91, 0.89, 0.90, 0.92]),
            "svm": _manifest(0.5, 0.5, 0.5, folds=[0.50, 0.51, 0.49, 0.50, 0.48]),
        }
    }
    result = _kruskal_wallis_per_season(manifests)
    assert result["summer"]["significant_at_0.05"] is True
    assert result["summer"]["p_value"] < 0.05


def test_kruskal_wallis_skips_season_with_fewer_than_two_scored_models():
    manifests = {
        "summer": {
            "rf": _manifest(0.8, 0.8, 0.75, folds=[0.75, 0.76, 0.74]),
            "svm": _manifest(0.8, 0.8, 0.75, folds=None),  # no spatial CV run
        }
    }
    result = _kruskal_wallis_per_season(manifests)
    assert "skipped" in result["summer"]


def test_kruskal_wallis_handles_unequal_fold_counts_across_models():
    """Confirms the documented assumption in stage_selection.py:
    kruskal() does NOT require equal-length or index-aligned fold lists
    across groups — unlike a paired/repeated-measures test."""
    manifests = {
        "summer": {
            "rf": _manifest(0.8, 0.8, 0.75, folds=[0.70, 0.72, 0.71]),               # 3 folds
            "svm": _manifest(0.8, 0.8, 0.75, folds=[0.60, 0.61, 0.59, 0.62, 0.58]),  # 5 folds
        }
    }
    result = _kruskal_wallis_per_season(manifests)
    assert "p_value" in result["summer"]
    assert "skipped" not in result["summer"]