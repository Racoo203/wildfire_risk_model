"""Blend NaN pixels into a desaturated hillshade backdrop instead of
rendering them blank/white, so maps read as 'no data here' rather than
'broken here' (Section 9)."""

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt


def render_with_terrain_backdrop(
    data_path: Path,
    dem_path: Path,
    cmap: str,
    ax: plt.Axes,
    backdrop_alpha: float = 0.25,
    title: Optional[str] = None,
    colorbar_label: Optional[str] = None,
) -> plt.Axes:
    """
    Render `data_path` on top of a desaturated hillshade of `dem_path`.
    NaN cells in `data_path` become fully transparent, revealing the
    hillshade underneath instead of a blank patch.
    """
    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
    with rasterio.open(data_path) as src:
        data = src.read(1)

    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(np.nan_to_num(dem, nan=0.0), vert_exag=2)
    ax.imshow(hillshade, cmap="gray", alpha=backdrop_alpha, zorder=0)

    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap=cmap, alpha=1.0, zorder=1)

    if title:
        ax.set_title(title)
    ax.axis("off")

    if colorbar_label:
        cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cbar.set_label(colorbar_label)

    return ax


def save_terrain_map(
    data_path: Path,
    dem_path: Path,
    out_path: Path,
    cmap: str = "viridis",
    title: Optional[str] = None,
    colorbar_label: Optional[str] = None,
    figsize: tuple = (8, 6),
    dpi: int = 150,
) -> Path:
    """Convenience wrapper: render_with_terrain_backdrop + save to disk."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    render_with_terrain_backdrop(
        data_path, dem_path, cmap, ax,
        title=title, colorbar_label=colorbar_label,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path