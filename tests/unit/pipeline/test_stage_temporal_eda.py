"""Unit tests for pipeline/stage_temporal_eda.py — the temporal (STL/ADF/
ACF-PACF) and spatial-correlogram EDA extracted from notebooks/01_eda.ipynb
so it's reproducible from src/ and exports its result tables as CSV rather
than notebook-display-only. Uses a synthetic monthly series rather than
real HadUK NetCDF, so these tests don't depend on bronze-tier data."""

import numpy as np
import pandas as pd
import rasterio

from wildfire_susceptibility.pipeline.stage_temporal_eda import (
    temporal_eda_report,
    variance_decomposition,
    _discover_feature_rasters,
    _run_spatial_correlogram,
)
import wildfire_susceptibility.pipeline.stage_temporal_eda as stage_temporal_eda_module

SEASON_MONTHS = {"winter": [12, 1, 2], "spring": [3, 4, 5], "summer": [6, 7, 8], "fall": [9, 10, 11]}


def _make_synthetic_monthly_series(n_years=10):
    idx = pd.date_range("2015-01-01", periods=n_years * 12, freq="MS")
    t = np.arange(len(idx))
    seasonal = 5 * np.sin(2 * np.pi * t / 12)
    trend = 0.05 * t
    noise = np.random.default_rng(0).normal(0, 0.5, len(idx))
    return pd.Series(15 + trend + seasonal + noise, index=idx)


def test_temporal_eda_report_writes_plots_and_returns_adf_stats(tmp_path):
    series = _make_synthetic_monthly_series()
    figures_dir = tmp_path / "figures"

    stats = temporal_eda_report(series, "test_var", figures_dir)

    assert set(stats.keys()) == {"feature", "adf_stat", "adf_p", "stationary"}
    assert stats["feature"] == "test_var"
    assert (figures_dir / "temporal_eda" / "test_var_decomposition.png").exists()
    assert (figures_dir / "temporal_eda" / "test_var_acf_pacf.png").exists()


def test_variance_decomposition_returns_pct_between_seasons():
    series = _make_synthetic_monthly_series()

    result = variance_decomposition(series, SEASON_MONTHS)

    assert "pct_variance_between_seasons" in result
    assert 0 <= result["pct_variance_between_seasons"] <= 100


def test_discover_feature_rasters_finds_only_existing_layers(tmp_path):
    layers_dir = tmp_path / "layers"
    layers_dir.mkdir()
    (layers_dir / "topo_elevation.tif").write_bytes(b"")
    (layers_dir / "ndvi_summer.tif").write_bytes(b"")
    (layers_dir / "meteo_tasmax_summer.tif").write_bytes(b"")
    # rainfall is configured but has no file on disk -> should be excluded

    config = {
        "base": {"output_dir": str(layers_dir)},
        "data_sources": {"haduk": {"sources": ["tasmax", "rainfall"]}},
    }

    found = _discover_feature_rasters(config, "summer")

    assert set(found.keys()) == {"elevation", "ndvi", "tasmax"}


def _write_dummy_raster(path, reference_transform, rng):
    meta = {
        "driver": "GTiff", "height": 10, "width": 10, "count": 1,
        "dtype": "float32", "crs": "EPSG:27700", "transform": reference_transform, "nodata": np.nan,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(rng.normal(size=(10, 10)).astype("float32")[np.newaxis, :, :])


def test_run_spatial_correlogram_writes_csv_and_png(tmp_path, reference_transform):
    layers_dir = tmp_path / "layers"
    layers_dir.mkdir()
    figures_dir = tmp_path / "figures"
    _write_dummy_raster(layers_dir / "topo_elevation.tif", reference_transform, np.random.default_rng(0))

    config = {
        "base": {"output_dir": str(layers_dir)},
        "data_sources": {"haduk": {"sources": []}},
        "seasons": {"active": ["summer"]},
    }

    out_path = _run_spatial_correlogram(config, figures_dir)

    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert (df["layer"] == "elevation").any()
    assert (figures_dir / "eda" / "spatial_correlogram.png").exists()


def test_stage_temporal_eda_end_to_end(tmp_path, monkeypatch, reference_transform):
    series = _make_synthetic_monthly_series()
    monkeypatch.setattr(
        stage_temporal_eda_module, "load_climate_timeseries",
        lambda config, var, year_range=None: series,
    )

    layers_dir = tmp_path / "layers"
    layers_dir.mkdir()
    figures_dir = tmp_path / "figures"
    _write_dummy_raster(layers_dir / "topo_elevation.tif", reference_transform, np.random.default_rng(0))

    config = {
        "base": {"output_dir": str(layers_dir), "figures_dir": str(figures_dir)},
        "data_sources": {"haduk": {"sources": ["tasmax"]}},
        "seasons": {"active": ["summer"], "definitions": SEASON_MONTHS},
        "processing": {"training_years": [2015, 2019]},
    }

    out = stage_temporal_eda_module.stage_temporal_eda(config)

    assert out["stationarity_summary"].exists()
    assert out["spatial_correlogram"].exists()

    summary_df = pd.read_csv(out["stationarity_summary"])
    assert list(summary_df["feature"]) == ["tasmax"]
    assert "pct_variance_between_seasons" in summary_df.columns
