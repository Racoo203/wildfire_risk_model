"""Non-map figures: ROC curves, SHAP summaries, VIF/correlation heatmaps,
class balance, NaN coverage — all logged to figures/manifest.json."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc as sklearn_auc

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .manifest import append_to_manifest

import logging

def _save_and_log(
    fig, 
    out_path: Path, 
    figures_dir: Path, 
    category: str, 
    generated_by: str,
    season: Optional[str] = None, 
    params: Optional[dict] = None
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    append_to_manifest(figures_dir, out_path, category, generated_by, season, params)
    return out_path


def plot_roc_curves(
    y_true: np.ndarray,
    proba_by_model: Dict[str, np.ndarray],
    n_classes: int,
    figures_dir: Path,
    season: Optional[str] = None,
) -> Path:
    """One-vs-rest ROC curves, one line per model, averaged across classes."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, proba in proba_by_model.items():
        y_bin = np.eye(n_classes)[y_true.astype(int)]
        fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
        roc_auc = sklearn_auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC — model comparison" + (f" ({season})" if season else ""))
    ax.legend()

    out_path = Path(figures_dir) / "models" / (season or "static") / "roc_comparison.png"
    return _save_and_log(fig, out_path, figures_dir, "roc_curve", "viz.charts.plot_roc_curves", season)


def plot_shap_summary(shap_values, feature_names: List[str], figures_dir: Path,
                       season: Optional[str] = None, model_name: Optional[str] = None) -> Path:
    """SHAP beeswarm summary plot for the selected best model."""
    import shap  # local import — heavy, optional dependency

    fig = plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, feature_names=feature_names, show=False)

    fname = f"shap_summary_{model_name or 'model'}.png"
    out_path = Path(figures_dir) / "models" / (season or "static") / fname
    return _save_and_log(fig, out_path, figures_dir, "shap_summary", "viz.charts.plot_shap_summary",
                          season, params={"model": model_name})

def plot_vif_correlation(
    df: pd.DataFrame,
    feature_cols: List[str],
    figures_dir: Path,
    season: Optional[str] = None,
    vif_threshold: float = 10.0,
    corr_threshold: float = 0.8,
) -> Dict[str, Path]:
    """
    VIF bar chart + Pearson correlation heatmap, using the paper's exact
    thresholds (Section 1.3): VIF > 10 or |corr| > 0.8 flags a feature.
    Returns paths to both figures.
    """
    X = df[feature_cols].dropna()
    X_with_const = sm.add_constant(X)

    vifs = pd.Series(
        [
        variance_inflation_factor(X_with_const.values, i)
        for i in range(X_with_const.shape[1])
        ],
        index=X_with_const.columns,
    ).drop("const")

    vifs = vifs.sort_values(ascending=False)

    fig1, ax1 = plt.subplots(figsize=(7, max(4, 0.35 * len(feature_cols))))
    colors = ["crimson" if v > vif_threshold else "steelblue" for v in vifs]
    ax1.barh(vifs.index, vifs.values, color=colors)
    ax1.axvline(vif_threshold, color="black", linestyle="--", label=f"threshold={vif_threshold}")
    ax1.set_xlabel("VIF")
    ax1.set_title(f"Variance Inflation Factor" + (f" ({season})" if season else ""))
    ax1.legend()
    vif_path = _save_and_log(
        fig1, Path(figures_dir) / "eda" / f"vif_{season or 'static'}.png", figures_dir,
        "vif", "viz.charts.plot_vif_correlation", season, params={"threshold": vif_threshold},
    )

    corr = X.corr()
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    mask_high = corr.abs() > corr_threshold
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax2,
                annot=False, square=True)
    ax2.set_title(f"Feature Correlation (|r| > {corr_threshold} flagged)" + (f" — {season}" if season else ""))
    corr_path = _save_and_log(
        fig2, Path(figures_dir) / "eda" / f"correlation_{season or 'static'}.png", figures_dir,
        "correlation", "viz.charts.plot_vif_correlation", season, params={"threshold": corr_threshold},
    )

    corr_spearman = X.corr(method="spearman")
    fig3, ax3 = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr_spearman, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax3, square=True)
    ax3.set_title(f"Spearman correlation" + (f" — {season}" if season else ""))
    spearman_path = _save_and_log(
        fig3, Path(figures_dir) / "eda" / f"correlation_spearman_{season or 'static'}.png",
        figures_dir, "correlation_spearman", "viz.charts.plot_vif_correlation", season,
    )

    n_flagged = int(((mask_high.sum().sum() - len(feature_cols))) / 2)  # exclude diagonal, undouble
    if n_flagged > 0 or (vifs > vif_threshold).any():
        logging.getLogger(__name__).warning(
            f"[{season}] VIF/correlation: {(vifs > vif_threshold).sum()} feature(s) exceed VIF>{vif_threshold}, "
            f"{n_flagged} pair(s) exceed |corr|>{corr_threshold}"
        )

    disagreement = (corr_spearman - corr).abs()
    mask_upper = np.triu(np.ones(disagreement.shape, dtype=bool), k=1)
    notable = disagreement.where(mask_upper).stack()
    notable = notable[notable > 0.15].index.tolist()
    if notable:
        logging.getLogger(__name__).info(
            f"[{season}] {len(notable)} feature pair(s) show notable Pearson/Spearman "
            f"divergence (>0.15) — possible non-linear monotonic relationship: {notable[:5]}"
        )

    return {"vif": vif_path, "correlation": corr_path, "spearman": spearman_path}


def plot_class_balance(
    labels_before: np.ndarray,
    labels_after: np.ndarray,
    figures_dir: Path,
    season: Optional[str] = None,
) -> Path:
    """Side-by-side class counts before/after k-means Low-cleaning."""
    classes = ["Low", "Medium", "High", "Very High"]

    def _counts(arr):
        valid = arr[~np.isnan(arr)]
        return [int(np.sum(valid == c)) for c in range(4)]

    before, after = _counts(labels_before), _counts(labels_after)

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(4)
    width = 0.35
    ax.bar(x - width / 2, before, width, label="Before cleaning", color="steelblue")
    ax.bar(x + width / 2, after, width, label="After cleaning", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Pixel count")
    ax.set_title("Class balance before/after label cleaning" + (f" — {season}" if season else ""))
    ax.legend()

    out_path = Path(figures_dir) / "eda" / f"class_balance_{season or 'static'}.png"
    return _save_and_log(fig, out_path, figures_dir, "class_balance",
                          "viz.charts.plot_class_balance", season)

def plot_nan_coverage(
    feature_arrays: Dict[str, np.ndarray],
    figures_dir: Path,
    season: Optional[str] = None,
) -> Path:
    """Per-feature NaN percentage bar chart — the Step-4 EDA figure."""
    pct_nan = {
        name: 100 * np.isnan(arr).sum() / arr.size
        for name, arr in feature_arrays.items()
    }
    series = pd.Series(pct_nan).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(series))))
    ax.barh(series.index, series.values, color="slategray")
    ax.set_xlabel("NaN (%)")
    ax.set_title("NaN coverage by feature" + (f" — {season}" if season else ""))

    out_path = Path(figures_dir) / "eda" / f"nan_coverage_{season or 'static'}.png"
    return _save_and_log(fig, out_path, figures_dir, "nan_coverage",
                          "viz.charts.plot_nan_coverage", season)

def plot_cv_comparison(figures_dir, season=None, mlflow_experiment=None):
    """Standard vs spatial CV AUC per model — the optimism-gap figure."""
    import mlflow

    runs = mlflow.search_runs(experiment_names=[mlflow_experiment])
    if season:
        runs = runs[runs["tags.season"] == season]

    models = runs["tags.model"].tolist()
    standard = runs["metrics.cv_auc_standard"].tolist()
    spatial = runs["metrics.cv_auc_spatial"].tolist()

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width/2, standard, width, label="Standard CV", color="steelblue")
    ax.bar(x + width/2, spatial, width, label="Spatial CV", color="darkorange")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("AUC")
    ax.set_title(f"Standard vs Spatial CV — optimism gap" + (f" ({season})" if season else ""))
    ax.legend()

    out_path = Path(figures_dir) / "models" / (season or "static") / "cv_comparison.png"
    return _save_and_log(fig, out_path, figures_dir, "cv_comparison", "viz.charts.plot_cv_comparison", season)