"""Distance-band Moran's I spatial autocorrelation, plus an empirical
semivariogram as a second, independent way of sizing spatial CV
block/buffer distances (configs/modeling.yaml's spatial_block_size_m /
spatial_buffer_m). Moran's I answers "is autocorrelation still
significant at this lag" (a bounded index + permutation p-value per
band); the semivariogram answers "how much does dissimilarity keep
growing with distance" (an unbounded, monotonically-rising-to-a-plateau
quantity, fit to a parametric range) — the two can disagree on where
the practical autocorrelation range sits. The underlying functions are
generic over any point sample + values, so they also serve as the
reusable core for a broader per-feature spatial autocorrelation report
(see pipeline/stage_temporal_eda.py)."""

from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio


def spatial_correlogram(
    coords: np.ndarray,
    values: np.ndarray,
    band_edges: Sequence[float],
    permutations: int = 199,
) -> pd.DataFrame:
    """Moran's I per distance band (ring/annulus weights over a point
    sample, row-standardized) rather than one fixed-lag lattice weight.
    band_edges are ring boundaries in metres, e.g. [0, 500, 1000, ...].
    The autocorrelation range is the band where I drops to ~0 / loses
    significance."""
    from libpysal.weights import full2W
    from esda.moran import Moran
    from scipy.spatial.distance import pdist, squareform

    dist_matrix = squareform(pdist(coords))
    rows = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        band = (dist_matrix >= lo) & (dist_matrix < hi)
        np.fill_diagonal(band, False)
        n_links = int(band.sum())
        if n_links == 0:
            rows.append({"lag_lo_m": lo, "lag_hi_m": hi, "I": np.nan, "p_sim": np.nan, "n_links": 0})
            continue
        w = full2W(band.astype(float), silence_warnings=True)
        w.transform = "r"
        mi = Moran(values, w, permutations=permutations)
        rows.append({"lag_lo_m": lo, "lag_hi_m": hi, "I": mi.I, "p_sim": mi.p_sim, "n_links": n_links})
    return pd.DataFrame(rows)


def semivariogram(
    coords: np.ndarray,
    values: np.ndarray,
    band_edges: Sequence[float],
) -> pd.DataFrame:
    """Empirical semivariogram: binned semivariance
    gamma(h) = mean(0.5 * (z_i - z_j)^2) over point pairs whose
    separation falls in each distance band. Uses the same band_edges
    convention as spatial_correlogram (ring boundaries in metres) so the
    two can be run on the same point sample and compared directly.
    Unlike Moran's I, gamma(h) is unbounded and rises with distance until
    it plateaus at the sill; the plateau distance (see
    fit_spherical_variogram) is the semivariogram's analogue of "I drops
    to ~0"."""
    from scipy.spatial.distance import pdist, squareform

    dist_matrix = squareform(pdist(coords))
    sq_diff = squareform(pdist(values.reshape(-1, 1), metric="sqeuclidean"))

    rows = []
    for lo, hi in zip(band_edges[:-1], band_edges[1:]):
        band = (dist_matrix >= lo) & (dist_matrix < hi)
        np.fill_diagonal(band, False)
        n_pairs = int(band.sum())
        gamma = 0.5 * sq_diff[band].mean() if n_pairs else np.nan
        rows.append({"lag_lo_m": lo, "lag_hi_m": hi, "gamma": gamma, "n_pairs": n_pairs})
    return pd.DataFrame(rows)


def fit_spherical_variogram(semivariogram_df: pd.DataFrame) -> dict:
    """Fit the spherical variogram model (the standard choice for this;
    Cressie 1993) to an empirical semivariogram via nonlinear least
    squares:

        gamma(h) = nugget + sill * (1.5*(h/range) - 0.5*(h/range)**3)   h <= range
        gamma(h) = nugget + sill                                        h >  range

    `range_m` is the fitted distance at which spatial dependence
    plateaus (the sill) — the semivariogram's counterpart to
    spatial_block_size_m/spatial_buffer_m. Returns NaNs if there are
    fewer than 3 usable bands (underdetermined for a 3-parameter fit)."""
    from scipy.optimize import curve_fit

    df = semivariogram_df.dropna(subset=["gamma"])
    h = ((df["lag_lo_m"] + df["lag_hi_m"]) / 2.0).to_numpy(dtype=float)
    gamma = df["gamma"].to_numpy(dtype=float)

    if len(h) < 3:
        return {"nugget": np.nan, "sill": np.nan, "range_m": np.nan, "r_squared": np.nan}

    def _spherical(h, nugget, sill, range_):
        ratio = np.clip(h / range_, 0, 1)
        return nugget + sill * (1.5 * ratio - 0.5 * ratio**3)

    nugget0 = max(float(gamma.min()), 0.0)
    sill0 = max(float(gamma.max()) - nugget0, 1e-12)
    range0 = float(np.median(h))

    try:
        popt, _ = curve_fit(
            _spherical, h, gamma,
            p0=[nugget0, sill0, range0],
            bounds=([0, 0, h.min()], [gamma.max(), gamma.max() * 2 + 1e-9, h.max() * 3]),
            maxfev=10000,
        )
    except RuntimeError:
        return {"nugget": np.nan, "sill": np.nan, "range_m": np.nan, "r_squared": np.nan}

    nugget, sill, range_m = (float(v) for v in popt)
    residuals = gamma - _spherical(h, *popt)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((gamma - gamma.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"nugget": nugget, "sill": sill, "range_m": range_m, "r_squared": r_squared}


def sample_raster_points(path, n_sample: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Random sample of up to n_sample valid (non-NaN) pixel centers from a
    raster: returns (coords[N,2] in the raster's CRS units, values[N])."""
    with rasterio.open(path) as src:
        arr = src.read(1)
        transform = src.transform
    rows, cols = np.where(~np.isnan(arr))
    idx = rng.choice(len(rows), size=min(n_sample, len(rows)), replace=False)
    rows, cols = rows[idx], cols[idx]
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    return np.column_stack([xs, ys]), arr[rows, cols].astype(float)
