"""Unit tests for viz/spearman_significance.py's full-dataset Spearman
significance export -- the branch's whole point is that this covers every
column (including one-hot encoded categoricals and columns the used-feature
VIF/Spearman heatmap excludes), not just the modeling feature subset.

Deliberately does NOT call scipy.stats.spearmanr anywhere in this file --
see spearman_significance.py's module docstring: on this project's numpy/
scipy combination, spearmanr (and even plain numpy.corrcoef) crashes the
whole Python process (a hard OS-level exception, not a catchable Python
exception) rather than raising. Reference values below are cross-checked
against pandas' own DataFrame.corr(method="spearman") (confirmed safe --
it's the same batched, Cython-backed path plot_vif_correlation already
uses in charts.py) plus the manual t-distribution p-value formula,
matching what the module under test actually does.

NOTE: pandas' *Series*.corr(method="spearman") (a single pair, not a whole
DataFrame) is NOT safe here -- it internally delegates to
scipy.stats.spearmanr and crashes the same way. Always compute reference
values via `df[[a, b]].corr(method="spearman")`, never `df[a].corr(df[b],
...)`."""

import numpy as np
import pandas as pd
import pytest

from wildfire_susceptibility.viz.spearman_significance import (
    compute_spearman_significance_export,
    _p_value_from_rho,
)

_EXPECTED_COLUMNS = [
    "Var1", "Var2", "Spearman's rho", "test statistic", "significant? (yes/no, alpha=0.95)",
]


def _make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    return pd.DataFrame({
        "a": a,
        "b": a + rng.normal(0, 0.05, n),  # strongly correlated with a
        "c": rng.normal(size=n),  # independent noise
        "landuse_class": rng.integers(0, 3, n),  # categorical -> must be one-hot encoded
        "label": rng.integers(0, 4, n),
    })


def test_output_is_flat_one_row_per_unordered_pair(tmp_path):
    df = _make_df()
    result = compute_spearman_significance_export(df, tmp_path / "figures", season="summer")

    out_df = pd.read_csv(result["out_path"])
    assert list(out_df.columns) == _EXPECTED_COLUMNS

    # 3 numeric (a, b, c) + 3 one-hot dummies from landuse_class (0,1,2) + label = 7 columns
    n = result["n_cols"]
    assert n == 7
    expected_rows = n * (n - 1) // 2
    assert result["n_rows"] == expected_rows
    assert len(out_df) == expected_rows

    # No self-pairs, no duplicated (B, A) alongside (A, B).
    pairs = set(zip(out_df["Var1"], out_df["Var2"]))
    assert all(a != b for a, b in pairs)
    reversed_pairs = {(b, a) for a, b in pairs}
    assert pairs.isdisjoint(reversed_pairs)


def test_output_path_matches_eda_convention(tmp_path):
    df = _make_df()
    result = compute_spearman_significance_export(df, tmp_path / "figures", season="spring")

    assert result["out_path"] == tmp_path / "figures" / "spring" / "eda" / "spearman_significance_full.csv"
    assert result["out_path"].exists()


def test_categorical_column_replaced_by_one_hot_dummies_not_raw_column(tmp_path):
    df = _make_df()
    result = compute_spearman_significance_export(df, tmp_path / "figures", season="summer")

    out_df = pd.read_csv(result["out_path"])
    all_vars = set(out_df["Var1"]) | set(out_df["Var2"])

    assert "landuse_class" not in all_vars
    assert {"landuse_class_0", "landuse_class_1", "landuse_class_2"} <= all_vars
    assert result["onehot_parent_groups"]["landuse_class"] == [
        "landuse_class_0", "landuse_class_1", "landuse_class_2",
    ]


def test_strongly_correlated_pair_flagged_significant_independent_pair_not(tmp_path):
    df = _make_df(n=500)
    result = compute_spearman_significance_export(df, tmp_path / "figures", season="summer")
    out_df = pd.read_csv(result["out_path"])

    def _row(v1, v2):
        match = out_df[
            ((out_df["Var1"] == v1) & (out_df["Var2"] == v2))
            | ((out_df["Var1"] == v2) & (out_df["Var2"] == v1))
        ]
        assert len(match) == 1
        return match.iloc[0]

    ab = _row("a", "b")
    assert ab["Spearman's rho"] > 0.9
    assert ab["significant? (yes/no, alpha=0.95)"] == "yes"

    ac = _row("a", "c")
    assert abs(ac["Spearman's rho"]) < 0.3


def test_values_match_pandas_corr_and_manual_p_value(tmp_path):
    """Reference values come from pandas' own corr(method="spearman") plus
    the module's manual t-distribution p-value formula (_p_value_from_rho)
    -- exactly what compute_spearman_significance_export does internally --
    not scipy.stats.spearmanr, which crashes this environment (see module
    docstring)."""
    df = _make_df()
    result = compute_spearman_significance_export(df, tmp_path / "figures", season="summer")
    out_df = pd.read_csv(result["out_path"])

    row = out_df[(out_df["Var1"] == "a") & (out_df["Var2"] == "c")]
    if row.empty:
        row = out_df[(out_df["Var1"] == "c") & (out_df["Var2"] == "a")]
    expected_rho = df[["a", "c"]].corr(method="spearman").loc["a", "c"]
    expected_p = _p_value_from_rho(expected_rho, n_eff=len(df))

    assert row.iloc[0]["Spearman's rho"] == pytest.approx(expected_rho)
    assert row.iloc[0]["test statistic"] == pytest.approx(expected_p)


def test_pairwise_nan_handling_does_not_drop_unrelated_pairs(tmp_path):
    df = _make_df(n=300)
    df.loc[:50, "c"] = np.nan  # NaNs only in c

    result = compute_spearman_significance_export(df, tmp_path / "figures", season="summer")
    out_df = pd.read_csv(result["out_path"])

    # a-vs-b pair (neither has NaNs) must match the full, un-omitted
    # series -- c's missingness must not leak into it.
    row = out_df[(out_df["Var1"] == "a") & (out_df["Var2"] == "b")]
    if row.empty:
        row = out_df[(out_df["Var1"] == "b") & (out_df["Var2"] == "a")]
    expected_rho = df[["a", "b"]].corr(method="spearman").loc["a", "b"]
    assert row.iloc[0]["Spearman's rho"] == pytest.approx(expected_rho)

    # a-vs-c pair must still compute (not NaN/error) using the overlapping
    # non-null rows only.
    row = out_df[(out_df["Var1"] == "a") & (out_df["Var2"] == "c")]
    if row.empty:
        row = out_df[(out_df["Var1"] == "c") & (out_df["Var2"] == "a")]
    assert pd.notna(row.iloc[0]["Spearman's rho"])


@pytest.mark.parametrize("n_eff", [0, 1, 2])
def test_p_value_nan_below_minimum_sample_size(n_eff):
    assert np.isnan(_p_value_from_rho(0.5, n_eff))


def test_p_value_nan_when_rho_is_nan():
    assert np.isnan(_p_value_from_rho(float("nan"), 100))


@pytest.mark.parametrize("rho", [1.0, -1.0])
def test_p_value_zero_for_perfect_correlation(rho):
    assert _p_value_from_rho(rho, 50) == 0.0


def test_p_value_large_for_zero_correlation():
    assert _p_value_from_rho(0.0, 100) == pytest.approx(1.0)
