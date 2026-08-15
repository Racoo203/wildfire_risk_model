"""Unit test for stage_eda.py wiring the new full-dataset Spearman
significance export (viz/spearman_significance.py) alongside the existing
NaN-coverage/VIF/Spearman-heatmap/class-balance figures, and the newer
per-feature factor-map generation (this branch: fix-eda-feature-raster-
generation) that threads stage_static/stage_seasonal/stage_labels' output
into viz.render_all_factor_maps -- the same terrain-backdrop renderer
stage_evaluate.py uses for the susceptibility map.

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


def test_stage_eda_without_static_input_skips_factor_maps_with_warning(tmp_path, eda_input_paths, caplog):
    """No `static`/`seasonal`/`labels` keys in input_paths -- e.g. before
    this branch's run_pipeline.py wiring, or a caller that hasn't updated
    yet -- must degrade gracefully (log + skip), not raise."""
    config = {"base": {"figures_dir": str(tmp_path / "figures")}}

    out = stage_eda(config, eda_input_paths)

    assert out["summer"]["factor_maps"] == {}
    assert out["summer"]["static_factor_maps"] == {}
    assert "skipping feature raster maps" in caplog.text.lower()


def test_stage_eda_renders_static_factor_maps_once_not_once_per_season(
    tmp_path, eda_input_paths, monkeypatch,
):
    """elevation/slope/aspect/etc. don't vary by season -- rendering them
    into every season's folder would just be N identical copies of the
    same figure. render_all_factor_maps must be called with season=None
    for statics exactly once, regardless of how many seasons are active."""
    dem_path = tmp_path / "layers" / "topo_elevation.tif"
    dem_path.parent.mkdir(parents=True)
    dem_path.touch()

    static_paths = {
        "ref_path": dem_path,
        "boundary": tmp_path / "layers" / "boundary.shp",
        "elevation": dem_path,
        "slope": tmp_path / "layers" / "topo_slope.tif",
    }

    calls = []

    def _fake_render_all_factor_maps(feature_paths, dem_path_arg, figures_dir, season=None, class_labels_by_feature=None):
        calls.append({"feature_paths": dict(feature_paths), "season": season, "class_labels_by_feature": class_labels_by_feature})
        return {name: figures_dir for name in feature_paths}

    monkeypatch.setattr(stage_eda_module.viz, "render_all_factor_maps", _fake_render_all_factor_maps)

    # Two seasons sharing the same raw/clean CSVs, to exercise "once across seasons".
    raw_train = eda_input_paths["raw"]["summer"]["train"]
    clean_train = eda_input_paths["clean"]["summer"]["train"]
    input_paths = {
        "raw": {"summer": {"train": raw_train, "test": raw_train}, "spring": {"train": raw_train, "test": raw_train}},
        "clean": {"summer": {"train": clean_train, "test": clean_train}, "spring": {"train": clean_train, "test": clean_train}},
        "static": static_paths,
    }
    config = {"base": {"figures_dir": str(tmp_path / "figures")}}

    stage_eda(config, input_paths)

    static_calls = [c for c in calls if c["season"] is None]
    assert len(static_calls) == 1
    assert set(static_calls[0]["feature_paths"]) == {"elevation", "slope"}  # ref_path/boundary excluded


def test_stage_eda_renders_seasonal_factor_maps_from_test_split(tmp_path, eda_input_paths, monkeypatch):
    """Seasonal (climate/NDVI) features must be rendered from the *test*
    split -- the same split stage_evaluate's full-domain susceptibility
    raster is built from -- not train, so the two are spatially comparable."""
    dem_path = tmp_path / "layers" / "topo_elevation.tif"
    dem_path.parent.mkdir(parents=True)
    dem_path.touch()

    static_paths = {"ref_path": dem_path, "elevation": dem_path}
    seasonal_paths = {
        "summer": {
            "train": {"tasmax": tmp_path / "layers" / "meteo_tasmax_summer_train.tif"},
            "test": {"tasmax": tmp_path / "layers" / "meteo_tasmax_summer_test.tif"},
        },
    }
    labels_paths = {
        "summer": {
            "test": {"d_fires": tmp_path / "layers" / "dist_fires_summer.tif"},
        },
    }

    calls = []

    def _fake_render_all_factor_maps(feature_paths, dem_path_arg, figures_dir, season=None, class_labels_by_feature=None):
        calls.append({"feature_paths": dict(feature_paths), "season": season})
        return {name: figures_dir for name in feature_paths}

    monkeypatch.setattr(stage_eda_module.viz, "render_all_factor_maps", _fake_render_all_factor_maps)

    input_paths = {
        **eda_input_paths,
        "static": static_paths,
        "seasonal": seasonal_paths,
        "labels": labels_paths,
    }
    config = {"base": {"figures_dir": str(tmp_path / "figures")}}

    stage_eda(config, input_paths)

    seasonal_calls = [c for c in calls if c["season"] == "summer"]
    assert len(seasonal_calls) == 1
    rendered_paths = seasonal_calls[0]["feature_paths"]
    assert rendered_paths["tasmax"] == seasonal_paths["summer"]["test"]["tasmax"]  # test split, not train
    assert rendered_paths["d_fires"] == labels_paths["summer"]["test"]["d_fires"]


def test_stage_eda_passes_landuse_class_labels_built_from_human_fclasses(
    tmp_path, eda_input_paths, monkeypatch,
):
    dem_path = tmp_path / "layers" / "topo_elevation.tif"
    dem_path.parent.mkdir(parents=True)
    dem_path.touch()

    static_paths = {
        "ref_path": dem_path,
        "elevation": dem_path,
        "landuse_class": tmp_path / "layers" / "landuse_class.tif",
    }

    captured = {}

    def _fake_render_all_factor_maps(feature_paths, dem_path_arg, figures_dir, season=None, class_labels_by_feature=None):
        if season is None:
            captured.update(class_labels_by_feature or {})
        return {name: figures_dir for name in feature_paths}

    monkeypatch.setattr(stage_eda_module.viz, "render_all_factor_maps", _fake_render_all_factor_maps)

    config = {
        "base": {"figures_dir": str(tmp_path / "figures")},
        "data_sources": {"proximity": {"human_fclasses": ["residential", "farmland"]}},
    }
    input_paths = {**eda_input_paths, "static": static_paths}

    stage_eda(config, input_paths)

    assert captured["landuse_class"] == {0: "No human activity", 1: "Residential", 2: "Farmland"}
