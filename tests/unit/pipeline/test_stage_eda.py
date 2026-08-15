"""Unit test for stage_eda.py wiring the new full-dataset Spearman
significance export (viz/spearman_significance.py) alongside the existing
NaN-coverage/VIF/Spearman-heatmap/class-balance figures.

viz.plot_nan_coverage, viz.plot_vif_correlation, and viz.plot_class_balance
are all monkeypatched out below: every one of them is pre-existing code
this branch doesn't touch, and every one independently crashes the whole
Python process in this environment (all three bottom out in a matplotlib
bar/barh bezier-rendering crash, or in plot_vif_correlation's case also a
statsmodels/numpy linear-algebra crash upstream of that) -- none of them
previously covered by any test, and none something this branch is
responsible for. Out of scope to fix here; this test only needs to
exercise the wiring this branch actually added."""

import numpy as np
import pandas as pd
import pytest

from wildfire_susceptibility.pipeline.stage_eda import stage_eda
from wildfire_susceptibility.pipeline import stage_eda as stage_eda_module


def _write_season_csv(path, n=100, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "elevation": rng.normal(100, 20, n),
        "slope": rng.normal(5, 2, n),
        "d_fires": rng.uniform(0, 5000, n),
        "landuse_class": rng.integers(0, 3, n),
        "_x": rng.uniform(0, 10000, n),
        "_y": rng.uniform(0, 10000, n),
        "tas": rng.normal(15, 3, n),
        "tasmin": rng.normal(10, 3, n),
        "label": rng.integers(0, 4, n),
    })
    df.to_csv(path, index=False)
    return path


@pytest.fixture(autouse=True)
def _stub_out_unrelated_plot_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(
        stage_eda_module.viz, "plot_nan_coverage",
        lambda *a, **k: tmp_path / "nan_coverage_stub.png",
    )
    monkeypatch.setattr(
        stage_eda_module.viz, "plot_vif_correlation",
        lambda *a, **k: {
            "vif": tmp_path / "vif_stub.png", "spearman": tmp_path / "spearman_stub.png",
            "vif_csv": tmp_path / "vif_stub.csv", "spearman_csv": tmp_path / "spearman_stub.csv",
        },
    )
    monkeypatch.setattr(
        stage_eda_module.viz, "plot_class_balance",
        lambda *a, **k: tmp_path / "class_balance_stub.png",
    )


@pytest.fixture
def eda_input_paths(tmp_path):
    raw_train = tmp_path / "raw_summer_train.csv"
    clean_train = tmp_path / "clean_summer_train.csv"
    _write_season_csv(raw_train, seed=1)
    _write_season_csv(clean_train, seed=1)
    return {
        "raw": {"summer": {"train": raw_train, "test": raw_train}},
        "clean": {"summer": {"train": clean_train, "test": clean_train}},
    }


def test_stage_eda_writes_spearman_significance_export(tmp_path, eda_input_paths):
    config = {"base": {"figures_dir": str(tmp_path / "figures")}}

    out = stage_eda(config, eda_input_paths)

    assert "spearman_significance_full" in out["summer"]
    sig_path = out["summer"]["spearman_significance_full"]
    assert sig_path.exists()
    assert sig_path == tmp_path / "figures" / "summer" / "eda" / "spearman_significance_full.csv"

    sig_df = pd.read_csv(sig_path)
    assert list(sig_df.columns) == [
        "Var1", "Var2", "Spearman's rho", "test statistic", "significant? (yes/no, alpha=0.95)",
    ]

    # landuse_class one-hot expanded (3 codes), d_fires and coords included --
    # the whole point of this export vs. the used-feature-only VIF/Spearman
    # heatmap, which excludes _x/_y/tas/tasmin (see stage_eda.py's feature_cols).
    all_vars = set(sig_df["Var1"]) | set(sig_df["Var2"])
    assert "landuse_class" not in all_vars
    assert {"landuse_class_0", "landuse_class_1", "landuse_class_2"} <= all_vars
    assert "d_fires" in all_vars
    assert "_x" in all_vars and "_y" in all_vars
    assert "tas" in all_vars and "tasmin" in all_vars
    assert "label" in all_vars
