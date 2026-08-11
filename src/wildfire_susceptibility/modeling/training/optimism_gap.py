"""Standard-vs-spatial-CV optimism gap: the credibility result showing
whether spatial autocorrelation inflates apparent model performance.

Convention: gap = standard - spatial, for every metric. Standard CV folds
routinely place spatially-autocorrelated neighbors in both train and test,
letting the model "cheat" via nearby-point leakage; spatial CV (block-
grouped, optionally buffered via spatial_buffer_m) removes that leakage. A
positive gap means standard CV was optimistic relative to the
leakage-resistant spatial estimate; a gap near zero means spatial
autocorrelation wasn't meaningfully inflating the standard-CV numbers for
that metric.
"""
from typing import Dict, Sequence

DEFAULT_GAP_METRICS = ("auc", "f1_macro", "pr_auc_macro")


def compute_optimism_gap(
    standard: Dict[str, float],
    spatial: Dict[str, float],
    metrics: Sequence[str] = DEFAULT_GAP_METRICS,
) -> Dict[str, float]:
    """standard/spatial: dicts of mean-across-folds metric values keyed by
    metric name (e.g. {"auc": ..., "f1_macro": ..., "pr_auc_macro": ...}).
    Both must carry every key in `metrics`."""
    missing_standard = [m for m in metrics if m not in standard]
    missing_spatial = [m for m in metrics if m not in spatial]
    if missing_standard or missing_spatial:
        raise KeyError(
            f"compute_optimism_gap: missing metric(s) — standard missing {missing_standard}, "
            f"spatial missing {missing_spatial}"
        )
    return {m: standard[m] - spatial[m] for m in metrics}
