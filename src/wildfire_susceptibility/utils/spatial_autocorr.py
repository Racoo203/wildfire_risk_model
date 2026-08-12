"""Distance-band Moran's I spatial autocorrelation. Originally written to
empirically justify spatial CV block/buffer sizing
(configs/modeling.yaml's spatial_block_size_m / spatial_buffer_m) — the
underlying functions are generic over any point sample + values, so they
also serve as the reusable core for a broader per-feature spatial
autocorrelation report (see pipeline/stage_temporal_eda.py)."""

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
