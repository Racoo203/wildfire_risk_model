"""Unit tests for stage_selection's pure aggregation logic
(_select_best_per_season, _kruskal_wallis_per_season). These touch no
I/O beyond what the caller already provides as in-memory manifest
dicts, so they're tested directly against fabricated manifests rather
than requiring real trained artifacts on disk."""

import math

import pytest

from wildfire_susceptibility.pipeline.stage_selection import (
    _select_best_per_season,
    _kruskal_wallis_per_season,
    _kruskal_results_to_csv,
)


def _manifest(
    val_auc, cv_auc_standard, cv_auc_spatial, folds=None,
    val_pr_auc_macro=None, cv_pr_auc_macro_standard=None, cv_pr_auc_macro_spatial=None,
):
    return {
        "val_auc": val_auc,
        "val_f1": 0.5,
        "val_pr_auc_macro": val_pr_auc_macro,
        "cv_auc_standard": cv_auc_standard,
        "cv_auc_spatial": cv_auc_spatial,
        "cv_pr_auc_macro_standard": cv_pr_auc_macro_standard,
        "cv_pr_auc_macro_spatial": cv_pr_auc_macro_spatial,
        "cv_pr_auc_macro_spatial_folds": folds,
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


def test_select_best_pr_auc_picks_highest_val_pr_auc_macro():
    """Demonstrates why best_pr_auc and best_auc can disagree: random_forest
    wins on plain AUC (0.90 > 0.85) but svm wins on PR-AUC-macro (0.70 >
    0.60) — the majority-class-collapse-insensitivity gap that motivated
    switching Optuna's HPO objective to PR-AUC-macro (search.py) applies
    just as much to final model selection."""
    manifests = {
        "summer": {
            "random_forest": _manifest(
                val_auc=0.90, cv_auc_standard=0.80, cv_auc_spatial=0.75, val_pr_auc_macro=0.60,
            ),
            "svm": _manifest(
                val_auc=0.85, cv_auc_standard=0.82, cv_auc_spatial=0.78, val_pr_auc_macro=0.70,
            ),
        }
    }
    result = _select_best_per_season(manifests, "best_pr_auc")
    assert result["summer"]["selected_model"] == "svm"
    assert result["summer"]["val_pr_auc_macro"] == 0.70


def test_select_most_conservative_picks_smallest_optimism_gap_within_threshold():
    manifests = {
        "summer": {
            # standard-PR-AUC-macro leader, but big overfit gap (0.90 - 0.60 = 0.30)
            "xgboost": _manifest(
                val_auc=0.88, cv_auc_standard=0.90, cv_auc_spatial=0.60,
                cv_pr_auc_macro_standard=0.90, cv_pr_auc_macro_spatial=0.60,
            ),
            # within 1% of leader (0.895 >= 0.90 - 0.01), much smaller gap (0.895-0.85=0.045)
            "random_forest": _manifest(
                val_auc=0.86, cv_auc_standard=0.895, cv_auc_spatial=0.85,
                cv_pr_auc_macro_standard=0.895, cv_pr_auc_macro_spatial=0.85,
            ),
            # not within 1% of leader (0.85 < 0.89), excluded from candidates
            "svm": _manifest(
                val_auc=0.80, cv_auc_standard=0.85, cv_auc_spatial=0.84,
                cv_pr_auc_macro_standard=0.85, cv_pr_auc_macro_spatial=0.84,
            ),
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


def test_kruskal_results_to_csv_has_one_row_per_season_with_expected_columns():
    """kruskal_wallis.json was replaced with kruskal_wallis.csv to match
    every other stat table's export format (vif.csv, correlation_spearman.csv,
    stationarity_summary.csv, spatial_correlogram.csv, model_comparison.csv)."""
    manifests = {
        "summer": {
            "rf": _manifest(0.9, 0.9, 0.9, folds=[0.90, 0.91, 0.89]),
            "svm": _manifest(0.5, 0.5, 0.5, folds=[0.50, 0.51, 0.49]),
        }
    }
    kruskal_results = _kruskal_wallis_per_season(manifests)
    df = _kruskal_results_to_csv(kruskal_results)

    assert list(df.columns) == [
        "season", "statistic", "p_value", "significant_at_0.05",
        "n_models_compared", "models_compared", "skip_reason",
    ]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["season"] == "summer"
    assert row["n_models_compared"] == 2
    assert set(row["models_compared"].split(";")) == {"rf", "svm"}
    assert row["skip_reason"] is None


def test_kruskal_wallis_drops_nan_fold_scores_before_scoring():
    """fix-spatial-cv-auc-missing-classes: cv_pr_auc_macro_spatial_folds can
    now contain NaN entries (score_multiclass_fold's degenerate-fold marker).
    scipy.stats.kruskal's default nan_policy='propagate' would otherwise
    turn the statistic/p-value NaN for the WHOLE season the moment any one
    model has any one NaN fold, even though the other folds/models are
    perfectly real and comparable — those NaNs must be filtered out
    per-model before scipy ever sees them, and the real scores still used."""
    manifests = {
        "summer": {
            "rf": _manifest(0.9, 0.9, 0.9, folds=[0.90, float("nan"), 0.91, 0.89]),
            "svm": _manifest(0.5, 0.5, 0.5, folds=[0.50, 0.51, 0.49]),
        }
    }
    result = _kruskal_wallis_per_season(manifests)
    assert "skipped" not in result["summer"]
    assert not math.isnan(result["summer"]["statistic"])
    assert not math.isnan(result["summer"]["p_value"])
    assert result["summer"]["significant_at_0.05"] is True


def test_kruskal_wallis_drops_a_model_whose_every_fold_is_nan():
    """A model with fold-level scores present (not None, so it passes the
    existing `if m.get("cv_pr_auc_macro_spatial_folds")` truthiness check) but every
    one of them NaN (a fully degenerate spatial CV run for that model)
    must be excluded from the comparison entirely — same treatment as
    today's 'no fold-level scores at all' case — rather than contributing
    an empty sample to scipy.stats.kruskal."""
    manifests = {
        "summer": {
            "rf": _manifest(0.8, 0.8, 0.75, folds=[0.75, 0.76, 0.74]),
            "svm": _manifest(0.8, 0.8, 0.75, folds=[float("nan"), float("nan")]),
        }
    }
    result = _kruskal_wallis_per_season(manifests)
    assert "skipped" in result["summer"]
    assert result["summer"]["skipped"] == "fewer than 2 models with fold-level scores"


def test_kruskal_wallis_keeps_a_model_with_some_real_folds_alongside_some_nan_ones():
    """The model in the previous test is dropped only because it has ZERO
    real folds — a model with a mix of real and NaN folds must still
    contribute its real folds, not be dropped outright."""
    manifests = {
        "summer": {
            "rf": _manifest(0.9, 0.9, 0.9, folds=[0.90, 0.91, 0.89, 0.92, 0.90]),
            "svm": _manifest(0.5, 0.5, 0.5, folds=[float("nan"), 0.50, 0.51, float("nan"), 0.49]),
        }
    }
    result = _kruskal_wallis_per_season(manifests)
    assert "skipped" not in result["summer"]
    assert set(result["summer"]["models_compared"]) == {"rf", "svm"}
    assert result["summer"]["significant_at_0.05"] is True


def test_kruskal_results_to_csv_represents_skipped_season_as_a_row_not_a_dropped_one():
    manifests = {
        "summer": {
            "rf": _manifest(0.8, 0.8, 0.75, folds=[0.75, 0.76, 0.74]),
            "svm": _manifest(0.8, 0.8, 0.75, folds=None),
        }
    }
    kruskal_results = _kruskal_wallis_per_season(manifests)
    df = _kruskal_results_to_csv(kruskal_results)

    assert len(df) == 1
    row = df.iloc[0]
    assert row["season"] == "summer"
    assert row["statistic"] is None
    assert row["skip_reason"] == "fewer than 2 models with fold-level scores"