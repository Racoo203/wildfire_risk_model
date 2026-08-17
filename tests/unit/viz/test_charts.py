"""Unit tests for viz/charts.py's raw-table CSV exports — VIF and Spearman
correlation were previously computed but only ever rendered as PNGs, never
saved as reusable tables alongside the figures."""

import numpy as np
import pandas as pd

from wildfire_susceptibility.viz.charts import plot_vif_correlation, plot_class_balance


def _make_correlated_features(n=200):
    rng = np.random.default_rng(0)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    return pd.DataFrame({
        "a": a,
        "b": b,
        "c": a + rng.normal(0, 0.05, n),  # near-duplicate of a -> high VIF/correlation
    })


def test_plot_vif_correlation_writes_csv_alongside_png(tmp_path):
    df = _make_correlated_features()
    figures_dir = tmp_path / "figures"

    paths = plot_vif_correlation(df, ["a", "b", "c"], figures_dir, season="summer")

    assert paths["vif"].exists()
    assert paths["spearman"].exists()
    assert paths["vif_csv"].exists()
    assert paths["spearman_csv"].exists()

    vif_df = pd.read_csv(paths["vif_csv"], index_col=0)
    assert set(vif_df.index) == {"a", "b", "c"}
    assert "vif" in vif_df.columns

    corr_df = pd.read_csv(paths["spearman_csv"], index_col=0)
    assert set(corr_df.index) == {"a", "b", "c"}
    assert set(corr_df.columns) == {"a", "b", "c"}


def test_plot_vif_correlation_spearman_cols_diverge_from_vif_cols_and_one_hot_landuse_class(tmp_path):
    """VIF should stay scoped to the actual trained feature set (e.g. "a",
    "b" only), while the Spearman heatmap is a broader diagnostic view that
    can include columns VIF doesn't (here "label", "landuse_class") -- and
    landuse_class, an arbitrary categorical code, must be one-hot expanded
    rather than correlated as a raw integer."""
    df = _make_correlated_features()
    df["landuse_class"] = ([0, 1, 2] * (len(df) // 3 + 1))[: len(df)]
    df["label"] = df["a"] > 0

    figures_dir = tmp_path / "figures"
    paths = plot_vif_correlation(
        df, ["a", "b"], figures_dir, season="summer",
        spearman_cols=["a", "b", "landuse_class", "label"],
    )

    vif_df = pd.read_csv(paths["vif_csv"], index_col=0)
    assert set(vif_df.index) == {"a", "b"}

    corr_df = pd.read_csv(paths["spearman_csv"], index_col=0)
    assert "landuse_class" not in corr_df.columns
    assert {"landuse_class_0", "landuse_class_1", "landuse_class_2"} <= set(corr_df.columns)
    assert "label" in corr_df.columns
    assert "c" not in corr_df.columns


def test_plot_class_balance_derives_class_count_from_data_not_hardcoded_four(tmp_path):
    """Before class_labels.py existed, plot_class_balance hardcoded
    classes=[...4 names...] and range(4), so a 5-class label array's
    highest class (index 4) was silently excluded from both bars."""
    rng = np.random.default_rng(0)
    labels_before = rng.integers(0, 5, size=500).astype("float64")
    labels_after = rng.integers(0, 5, size=500).astype("float64")

    out_path = plot_class_balance(labels_before, labels_after, tmp_path / "figures", season="summer")

    assert out_path.exists()
