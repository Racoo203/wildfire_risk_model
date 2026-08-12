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

    out_path = Path(figures_dir) / (season or "static") / "models" / "roc_comparison.png"
    return _save_and_log(fig, out_path, figures_dir, "roc_curve", "viz.charts.plot_roc_curves", season)


def _split_shap_by_class(shap_values) -> List[np.ndarray]:
    """Normalize the several shapes SHAP explainers return into a list of
    per-class (n_samples, n_features) arrays: a per-class list
    (TreeExplainer, some SHAP versions), a single stacked
    (n_samples, n_features, n_classes) array (this project's installed
    SHAP version's TreeExplainer.shap_values() for multiclass models), or
    a plain 2D array (single-output case) wrapped as a one-item list."""
    if isinstance(shap_values, list):
        return shap_values
    if shap_values.ndim == 3:
        return [shap_values[:, :, k] for k in range(shap_values.shape[-1])]
    return [shap_values]


def plot_shap_summary(shap_values, feature_names, figures_dir, season=None, model_name=None):
    import shap
    import matplotlib.pyplot as plt

    by_class = _split_shap_by_class(shap_values)
    # shap.summary_plot's legacy code path only accepts a bare 2D array or
    # a list of per-class 2D arrays — not the single stacked 3D array some
    # SHAP versions' TreeExplainer.shap_values() returns for multiclass
    # models, which raises inside its internal sorting logic.
    shap.summary_plot(by_class if len(by_class) > 1 else by_class[0], feature_names=feature_names, show=False)
    fig = plt.gcf()          # capture AFTER the call, not before
    fig.set_size_inches(8, 6)

    fname = f"shap_summary_{model_name or 'model'}.png"
    out_path = Path(figures_dir) / (season or "static") / "models" / fname
    return _save_and_log(fig, out_path, figures_dir, "shap_summary", "viz.charts.plot_shap_summary",
                          season, params={"model": model_name})


def _shap_values_to_2d(shap_values) -> np.ndarray:
    """Collapse any SHAP explainer output down to a single
    (n_samples, n_features) array, averaging across classes for the
    multiclass case — gives an overview of each feature's average effect
    rather than a per-class breakdown."""
    by_class = _split_shap_by_class(shap_values)
    return np.mean(np.stack(by_class, axis=-1), axis=-1)


def plot_shap_dependence(
    shap_values,
    X: np.ndarray,
    feature_names: List[str],
    figures_dir: Path,
    season: Optional[str] = None,
    model_name: Optional[str] = None,
    top_n: int = 5,
) -> Path:
    """Dependence plots (SHAP value vs. feature value) for the top-N
    features by mean |SHAP|, as one multi-panel figure rather than N
    separate files. `X` must be the same array passed to the explainer
    that produced `shap_values` (same rows, same column order as
    `feature_names`)."""
    import shap

    shap_2d = _shap_values_to_2d(shap_values)
    mean_abs = np.abs(shap_2d).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][: min(top_n, len(feature_names))]

    n = len(top_idx)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for i, feat_idx in enumerate(top_idx):
        ax = axes[i // ncols][i % ncols]
        shap.dependence_plot(
            int(feat_idx), shap_2d, X, feature_names=feature_names,
            interaction_index=None, ax=ax, show=False,
        )
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("SHAP dependence — top features" + (f" ({model_name}, {season})" if model_name else ""))

    fname = f"shap_dependence_{model_name or 'model'}.png"
    out_path = Path(figures_dir) / (season or "static") / "models" / fname
    return _save_and_log(fig, out_path, figures_dir, "shap_dependence", "viz.charts.plot_shap_dependence",
                          season, params={"model": model_name, "top_n": top_n})

def plot_vif_correlation(
    df: pd.DataFrame,
    feature_cols: List[str],
    figures_dir: Path,
    season: Optional[str] = None,
    vif_threshold: float = 10.0,
    corr_threshold: float = 0.8,
) -> Dict[str, Path]:
    """
    VIF bar chart + Spearman correlation heatmap, using the paper's exact
    thresholds (Section 1.3): VIF > 10 flags a feature.
    Returns paths to VIF and Spearman figures.
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
    ax1.set_title("Variance Inflation Factor" + (f" ({season})" if season else ""))
    ax1.legend()
    vif_path = _save_and_log(
        fig1, Path(figures_dir) / (season or "static") / "eda" / "vif.png", figures_dir,
        "vif", "viz.charts.plot_vif_correlation", season, params={"threshold": vif_threshold},
    )

    if (vifs > vif_threshold).any():
        logging.getLogger(__name__).warning(
            f"[{season}] VIF: {(vifs > vif_threshold).sum()} feature(s) exceed VIF>{vif_threshold}"
        )

    corr_spearman = X.corr(method="spearman")
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr_spearman, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax2, square=True,
                annot=True, fmt=".2g", annot_kws={"size": 8})
    ax2.set_title("Spearman correlation" + (f" — {season}" if season else ""))
    spearman_path = _save_and_log(
        fig2, Path(figures_dir) / (season or "static") / "eda" / "correlation_spearman.png",
        figures_dir, "correlation_spearman", "viz.charts.plot_vif_correlation", season,
    )

    return {"vif": vif_path, "spearman": spearman_path}


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

    out_path = Path(figures_dir) / (season or "static") / "eda" / "class_balance.png"
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

    out_path = Path(figures_dir) / (season or "static") / "eda" / "nan_coverage.png"
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

    out_path = Path(figures_dir) / (season or "static") / "models" / "cv_comparison.png"
    return _save_and_log(fig, out_path, figures_dir, "cv_comparison", "viz.charts.plot_cv_comparison", season)


def plot_optuna_optimization_history(study, figures_dir, season=None, model_name=None) -> Path:
    """Optuna's per-trial objective value + running best, for one
    (season, model) hyperparameter search."""
    import optuna.visualization.matplotlib as optuna_mpl

    ax = optuna_mpl.plot_optimization_history(study)
    fig = ax.get_figure()

    fname = f"optuna_history_{model_name or 'model'}.png"
    out_path = Path(figures_dir) / (season or "static") / "models" / fname
    return _save_and_log(fig, out_path, figures_dir, "optuna_optimization_history",
                          "viz.charts.plot_optuna_optimization_history", season, params={"model": model_name})


def plot_optuna_param_importances(study, figures_dir, season=None, model_name=None) -> Path:
    """Optuna's fANOVA-based hyperparameter importance ranking for one
    (season, model) search."""
    import optuna.visualization.matplotlib as optuna_mpl

    ax = optuna_mpl.plot_param_importances(study)
    fig = ax.get_figure()

    fname = f"optuna_param_importance_{model_name or 'model'}.png"
    out_path = Path(figures_dir) / (season or "static") / "models" / fname
    return _save_and_log(fig, out_path, figures_dir, "optuna_param_importance",
                          "viz.charts.plot_optuna_param_importances", season, params={"model": model_name})


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    figures_dir: Path,
    season: Optional[str] = None,
    model_name: Optional[str] = None,
    normalize: bool = True,
) -> Path:
    cm = np.array(cm, dtype="float64")
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f" if normalize else ".0f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    title = "Confusion Matrix" + (" (row-normalized)" if normalize else "")
    if model_name:
        title += f" — {model_name}"
    if season:
        title += f" ({season})"
    ax.set_title(title)

    fname = f"confusion_matrix_{model_name or 'model'}.png"
    out_path = Path(figures_dir) / (season or "static") / "models" / fname
    return _save_and_log(fig, out_path, figures_dir, "confusion_matrix",
                          "viz.charts.plot_confusion_matrix", season, params={"model": model_name})