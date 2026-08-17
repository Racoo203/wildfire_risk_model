"""Stage: temporal + spatial-autocorrelation EDA.

Reproduces eda.ipynb's temporal-EDA section (STL decomposition, ADF
stationarity, ACF/PACF per HadUK climate variable, between/within-season
variance decomposition) and its spatial correlogram section (distance-band
Moran's I) as a pipeline stage, with both result tables exported as CSV
rather than notebook-display-only.

The correlogram was originally scoped narrowly to empirically justify
spatial_block_size_m/spatial_buffer_m (configs/modeling.yaml) against just
4 sample layers; here it runs across every discoverable feature raster per
active season, since the underlying spatial_correlogram() function is
generic over any point sample. Each season's result table/plot is saved
under its own {season}/eda/ directory (not pooled into a single top-level
eda/), matching every other per-season figure's layout.

    stage_temporal_eda(config) -> {
        "stationarity_summary": Path,           # csv
        "spatial_correlogram": Dict[str, Path],  # csv per season
    }

Reads raw HadUK NetCDF directly (climate time series) and sampled feature
rasters from config["base"]["output_dir"] (already written by the
static/seasonal/labels stages) — needs no trained models, runs from
feature data alone.
"""
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import xarray as xr

from .. import viz
from ..utils.logger import setup_logger
from ..utils.spatial_autocorr import spatial_correlogram, sample_raster_points

CORRELOGRAM_BAND_EDGES = [
    0, 150, 300, 500, 750, 1000,
    1500, 2000, 3000, 4000, 5500, 7000, 9000,
    11000, 14000, 18000, 23000, 30000,
]
CORRELOGRAM_N_SAMPLE = 4000

_STATIC_FEATURE_FILES = {
    "elevation": "topo_elevation.tif",
    "slope": "topo_slope.tif",
    "aspect": "topo_aspect.tif",
    "d_roads": "dist_roads.tif",
    "d_rivers": "dist_rivers.tif",
    "d_buildings": "dist_buildings.tif",
}


def load_climate_timeseries(config: dict, var_name: str, year_range=None) -> pd.Series:
    """Full monthly time series (pre seasonal-average) for one HadUK
    variable, spatially averaged across Essex."""
    data_dir = Path(config["data_sources"]["haduk"]["data_dir"]) / var_name
    files = sorted(data_dir.glob("*.nc"))
    ds = xr.open_mfdataset(files, combine="by_coords")
    da = ds[var_name]
    if year_range:
        da = da.sel(time=slice(f"{year_range[0]}-01-01", f"{year_range[1]}-12-31"))
    return da.mean(dim=["projection_x_coordinate", "projection_y_coordinate"]).to_series()


def variance_decomposition(series: pd.Series, season_months: Dict[str, list]) -> dict:
    """% of total variance explained by season (between) vs within-season
    month-to-month variation. High between-season % justifies collapsing
    to a seasonal mean rather than keeping full monthly observations."""
    labels = series.index.month.map(
        lambda m: next(s for s, months in season_months.items() if m in months)
    )
    df = pd.DataFrame({"value": series.values, "season": labels})
    grand_mean = df["value"].mean()
    between = df.groupby("season")["value"].mean()
    ss_between = (df.groupby("season").size() * (between - grand_mean) ** 2).sum()
    ss_within = df.groupby("season")["value"].apply(lambda x: ((x - x.mean()) ** 2).sum()).sum()
    return {"pct_variance_between_seasons": 100 * ss_between / (ss_between + ss_within)}


def temporal_eda_report(series: pd.Series, name: str, figures_dir) -> dict:
    """STL decomposition + ACF/PACF plots (saved via viz/charts.py) + ADF
    stationarity test for one feature's time series. Returns summary
    stats; the caller is responsible for collecting them into a table."""
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.stattools import adfuller

    stl = STL(series, period=12, robust=True).fit()
    viz.plot_temporal_decomposition(series, stl, name, figures_dir)
    viz.plot_acf_pacf(series, name, figures_dir)

    adf_stat, adf_p, *_ = adfuller(series.dropna())
    return {"feature": name, "adf_stat": adf_stat, "adf_p": adf_p, "stationary": adf_p < 0.05}


def _discover_feature_rasters(config: dict, season: str) -> Dict[str, Path]:
    """Every feature raster relevant to `season`, by the same naming
    convention reporting/generate_report_figures.py's
    discover_feature_rasters uses (kept independent rather than imported,
    since pipeline stages shouldn't depend on the reporting script) —
    except split is fixed to "train" here, not "test": this correlogram
    exists to justify spatial_block_size_m/spatial_buffer_m, which only
    ever partition the training set, so autocorrelation should be measured
    on the same rasters spatial CV actually folds over.

    Climate/NDVI/diurnal_range builders write a separate raster per split
    (ClimateBuilder/VegetationBuilder), so these need the _train suffix
    that's missing from the static (topography/proximity) filenames."""
    layers_dir = Path(config["base"]["output_dir"])
    haduk_vars = config["data_sources"]["haduk"]["sources"]

    candidates = {name: layers_dir / fname for name, fname in _STATIC_FEATURE_FILES.items()}
    candidates["d_fires"] = layers_dir / f"dist_fires_{season}.tif"
    candidates["ndvi"] = layers_dir / f"ndvi_{season}_train.tif"
    candidates["diurnal_range"] = layers_dir / f"meteo_diurnal_range_{season}_train.tif"
    for var in haduk_vars:
        candidates[var] = layers_dir / f"meteo_{var}_{season}_train.tif"

    return {name: path for name, path in candidates.items() if path.exists()}


def _run_spatial_correlogram(config: dict, figures_dir) -> Dict[str, Path]:
    """Distance-band Moran's I for every discoverable feature raster,
    per active season. Each season's table/plot is written to its own
    {season}/eda/ directory rather than pooled into one top-level eda/."""
    rng = np.random.default_rng(42)

    out_paths = {}
    for season in config["seasons"]["active"]:
        frames = []
        for layer_name, path in _discover_feature_rasters(config, season).items():
            coords, values = sample_raster_points(path, CORRELOGRAM_N_SAMPLE, rng)
            df = spatial_correlogram(coords, values, CORRELOGRAM_BAND_EDGES)
            df["season"], df["layer"] = season, layer_name
            frames.append(df)

        correlogram_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        out_path = Path(figures_dir) / season / "eda" / "spatial_correlogram.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        correlogram_df.to_csv(out_path, index=False)

        if not correlogram_df.empty:
            viz.plot_spatial_correlogram(correlogram_df, figures_dir, season=season)

        out_paths[season] = out_path

    return out_paths


def stage_temporal_eda(config: dict) -> Dict[str, Path]:
    logger = setup_logger()
    figures_dir = Path(config["base"]["figures_dir"])

    climate_vars = config["data_sources"]["haduk"]["sources"]
    season_months = config["seasons"]["definitions"]
    year_range = config["processing"]["training_years"]

    summary_rows = []
    for var in climate_vars:
        logger.info(f"[stage_temporal_eda] Analyzing {var}...")
        series = load_climate_timeseries(config, var, year_range=year_range)
        stats = temporal_eda_report(series, var, figures_dir)
        var_decomp = variance_decomposition(series, season_months)
        summary_rows.append({**stats, **var_decomp})

    stationarity_path = Path(figures_dir) / "temporal_eda" / "stationarity_summary.csv"
    stationarity_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(stationarity_path, index=False)
    logger.info(f"[stage_temporal_eda] Stationarity summary -> {stationarity_path.name}")

    correlogram_paths = _run_spatial_correlogram(config, figures_dir)
    for season, path in correlogram_paths.items():
        logger.info(f"[stage_temporal_eda] Spatial correlogram [{season}] -> {path}")

    return {"stationarity_summary": stationarity_path, "spatial_correlogram": correlogram_paths}
