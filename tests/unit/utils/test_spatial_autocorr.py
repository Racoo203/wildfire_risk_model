"""Unit tests for utils/spatial_autocorr.py — the distance-band Moran's I
correlogram extracted from notebooks/01_eda.ipynb so it's importable from
src/ (and therefore reproducible outside the notebook, and reusable beyond
the original block-size-justification use case)."""

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from wildfire_susceptibility.utils.spatial_autocorr import (
    spatial_correlogram,
    sample_raster_points,
    semivariogram,
    fit_spherical_variogram,
)


def test_spatial_correlogram_detects_strong_autocorrelation_in_a_gradient():
    rng = np.random.default_rng(0)
    n = 150
    coords = rng.uniform(0, 1000, size=(n, 2))
    values = coords[:, 0] + coords[:, 1] + rng.normal(0, 1, n)  # smooth spatial gradient

    result = spatial_correlogram(coords, values, [0, 200, 400, 600, 800, 1000], permutations=99)

    nearest_band = result.iloc[0]
    assert nearest_band["I"] > 0.3
    assert nearest_band["p_sim"] < 0.05


def test_spatial_correlogram_finds_near_zero_autocorrelation_in_pure_noise():
    rng = np.random.default_rng(0)
    n = 150
    coords = rng.uniform(0, 1000, size=(n, 2))
    values = np.random.default_rng(1).normal(0, 1, n)  # no spatial structure at all

    result = spatial_correlogram(coords, values, [0, 200, 400, 600, 800, 1000], permutations=99)

    nearest_band = result.iloc[0]
    assert abs(nearest_band["I"]) < 0.25
    assert nearest_band["p_sim"] > 0.05


def test_spatial_correlogram_handles_empty_band_gracefully():
    rng = np.random.default_rng(2)
    coords = rng.uniform(0, 10, size=(5, 2))  # all points clustered close together
    values = rng.normal(size=5)

    result = spatial_correlogram(coords, values, [1000, 2000], permutations=9)  # band far beyond any pair

    assert np.isnan(result.iloc[0]["I"])
    assert result.iloc[0]["n_links"] == 0


def test_semivariogram_rises_with_distance_for_a_smooth_gradient():
    rng = np.random.default_rng(0)
    n = 150
    coords = rng.uniform(0, 1000, size=(n, 2))
    values = coords[:, 0] + coords[:, 1] + rng.normal(0, 1, n)  # smooth spatial gradient

    result = semivariogram(coords, values, [0, 200, 400, 600, 800, 1000])

    assert (result["n_pairs"] > 0).all()
    # semivariance should grow (roughly monotonically) with lag for a smooth gradient
    assert result.iloc[-1]["gamma"] > result.iloc[0]["gamma"]


def test_semivariogram_is_flat_for_pure_noise():
    rng = np.random.default_rng(0)
    n = 150
    coords = rng.uniform(0, 1000, size=(n, 2))
    values = np.random.default_rng(1).normal(0, 1, n)  # no spatial structure at all

    result = semivariogram(coords, values, [0, 200, 400, 600, 800, 1000])

    # pure noise: semivariance should hover near the variance at every lag,
    # not show a strong trend the way the gradient case does
    gamma = result["gamma"].to_numpy()
    assert gamma.max() / gamma.min() < 3


def test_semivariogram_handles_empty_band_gracefully():
    rng = np.random.default_rng(2)
    coords = rng.uniform(0, 10, size=(5, 2))  # all points clustered close together
    values = rng.normal(size=5)

    result = semivariogram(coords, values, [1000, 2000])  # band far beyond any pair

    assert np.isnan(result.iloc[0]["gamma"])
    assert result.iloc[0]["n_pairs"] == 0


def test_fit_spherical_variogram_recovers_a_known_range():
    rng = np.random.default_rng(3)
    true_range = 500.0
    band_edges = np.linspace(0, 1500, 16)
    h = (band_edges[:-1] + band_edges[1:]) / 2.0
    ratio = np.clip(h / true_range, 0, 1)
    gamma = 1.0 + 4.0 * (1.5 * ratio - 0.5 * ratio**3)  # nugget=1, sill=4
    df = pd.DataFrame({"lag_lo_m": band_edges[:-1], "lag_hi_m": band_edges[1:], "gamma": gamma})

    fit = fit_spherical_variogram(df)

    assert fit["range_m"] == pytest.approx(true_range, rel=0.1)
    assert fit["r_squared"] > 0.95


def test_fit_spherical_variogram_returns_nan_when_underdetermined():
    df = pd.DataFrame({"lag_lo_m": [0, 500], "lag_hi_m": [500, 1000], "gamma": [1.0, 2.0]})

    fit = fit_spherical_variogram(df)

    assert np.isnan(fit["range_m"])


def test_sample_raster_points_skips_nan_and_caps_at_n_sample(tmp_path):
    transform = from_origin(0, 100, 10, 10)
    arr = np.arange(100, dtype="float32").reshape(10, 10)
    arr[0, 0] = np.nan  # one invalid pixel

    path = tmp_path / "test_raster.tif"
    meta = {
        "driver": "GTiff", "height": 10, "width": 10, "count": 1,
        "dtype": "float32", "crs": "EPSG:27700", "transform": transform, "nodata": np.nan,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr[np.newaxis, :, :])

    coords, values = sample_raster_points(path, n_sample=20, rng=np.random.default_rng(0))

    assert coords.shape == (20, 2)
    assert values.shape == (20,)
    assert not np.isnan(values).any()
