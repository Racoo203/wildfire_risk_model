"""Blend NaN pixels into a desaturated hillshade backdrop instead of
rendering them blank/white, so maps read as 'no data here' rather than
'broken here' (Section 9). Also adds the cartographic chrome (scale bar,
north arrow, coordinate gridlines) every map figure needs for
publication use."""

from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd
import numpy as np
import rasterio
from matplotlib.colors import LightSource, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.pyplot as plt


def _nice_scalebar_length(span_m: float) -> float:
    """Round `span_m * 0.2` up to the nearest 1/2/5 x 10^n metres, so the
    scale bar reads as a round number rather than an arbitrary fraction
    of the visible extent."""
    target = span_m * 0.2
    if target <= 0:
        return 1.0
    magnitude = 10 ** np.floor(np.log10(target))
    for mult in (1, 2, 5, 10):
        candidate = mult * magnitude
        if candidate >= target:
            return candidate
    return 10 * magnitude


def _add_scalebar(ax: plt.Axes, transform, width_px: int) -> AnchoredSizeBar:
    px_size_m = abs(transform.a)
    length_m = _nice_scalebar_length(width_px * px_size_m)
    length_px = length_m / px_size_m
    label = f"{length_m / 1000:.3g} km" if length_m >= 1000 else f"{length_m:.0f} m"

    scalebar = AnchoredSizeBar(
        ax.transData, length_px, label, loc="lower right",
        pad=0.5, color="black", frameon=True,
        size_vertical=max(1.0, width_px * 0.01),
    )
    ax.add_artist(scalebar)
    return scalebar


def _add_north_arrow(ax: plt.Axes) -> None:
    ax.annotate(
        "N", xy=(0.95, 0.90), xytext=(0.95, 0.72),
        xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center", fontsize=12, fontweight="bold",
        arrowprops=dict(facecolor="black", width=4, headwidth=12, headlength=10),
    )


def _add_gridlines(ax: plt.Axes, transform, height: int, width: int, n_ticks: int = 5) -> None:
    """Coordinate-labeled gridlines. Tick *positions* stay in pixel-index
    units (matching imshow's default, extent-less coordinate system);
    only the tick *labels* are converted to CRS easting/northing via the
    raster's affine transform."""
    row_ticks = np.linspace(0, height - 1, n_ticks)
    col_ticks = np.linspace(0, width - 1, n_ticks)

    xs, _ = rasterio.transform.xy(transform, [0] * len(col_ticks), col_ticks)
    _, ys = rasterio.transform.xy(transform, row_ticks, [0] * len(row_ticks))

    ax.set_xticks(col_ticks)
    ax.set_xticklabels([f"{x:,.0f}" for x in xs], fontsize=7, rotation=45)
    ax.set_yticks(row_ticks)
    ax.set_yticklabels([f"{y:,.0f}" for y in ys], fontsize=7)
    ax.set_xlabel("Easting (m)", fontsize=8)
    ax.set_ylabel("Northing (m)", fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.tick_params(length=0)


def _add_points(
    ax: plt.Axes,
    transform,
    points_gdf: gpd.GeoDataFrame,
    label: Optional[str] = None,
) -> None:
    """Scatter point geometries (assumed already in the raster's CRS) onto
    the map, converted from CRS coordinates to the pixel-index units
    imshow uses by default — the same convention `_add_gridlines` relies
    on for tick placement."""
    if len(points_gdf) == 0:
        return
    rows, cols = rasterio.transform.rowcol(
        transform, points_gdf.geometry.x.values, points_gdf.geometry.y.values
    )
    ax.scatter(
        cols, rows, s=8, facecolor="black", edgecolor="black",
        alpha=0.9, linewidth=0, zorder=2, label=label,
    )
    if label:
        ax.legend(loc="upper left", fontsize=7, framealpha=0.8)


def render_with_terrain_backdrop(
    data_path: Path,
    dem_path: Path,
    cmap: str,
    ax: plt.Axes,
    backdrop_alpha: float = 0.25,
    title: Optional[str] = None,
    colorbar_label: Optional[str] = None,
    show_scalebar: bool = True,
    show_north_arrow: bool = True,
    show_gridlines: bool = True,
    points_gdf: Optional[gpd.GeoDataFrame] = None,
    points_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    discrete_legend: Optional[Dict[int, str]] = None,
) -> plt.Axes:
    """
    Render `data_path` on top of a desaturated hillshade of `dem_path`.
    NaN cells in `data_path` become fully transparent, revealing the
    hillshade underneath instead of a blank patch. If `points_gdf` is
    given, its point geometries are scattered on top (e.g. fire incident
    locations over a predicted susceptibility map).

    `vmin`/`vmax` fix the color scale explicitly (e.g. to a config's full
    class range) instead of matplotlib's default of scaling to this
    raster's own data min/max — needed so a class entirely absent from one
    raster doesn't shift the color mapping relative to a raster where every
    class is present.

    `discrete_legend` (e.g. {0: "No human activity", 1: "Residential", ...})
    renders `data_path`'s integer codes as flat colour bins (one colour per
    key, via BoundaryNorm) with a labelled patch legend instead of a
    continuous colorbar — for genuinely categorical rasters (landuse_class)
    where a continuous colorbar would misleadingly imply the codes are
    ordinal. Mutually exclusive with `vmin`/`vmax` (matplotlib forbids
    passing both `norm` and `vmin`/`vmax` to imshow).
    """
    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
    with rasterio.open(data_path) as src:
        data = src.read(1)
        transform = src.transform
        height, width = src.height, src.width

    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(np.nan_to_num(dem, nan=0.0), vert_exag=2)
    ax.imshow(hillshade, cmap="gray", alpha=backdrop_alpha, zorder=0)

    masked = np.ma.masked_invalid(data)

    legend_cmap = None
    sorted_keys = []
    if discrete_legend:
        sorted_keys = sorted(discrete_legend)
        base_cmap = plt.get_cmap(cmap, len(sorted_keys))
        legend_cmap = ListedColormap([base_cmap(i) for i in range(len(sorted_keys))])
        bounds = [k - 0.5 for k in sorted_keys] + [sorted_keys[-1] + 0.5]
        norm = BoundaryNorm(bounds, legend_cmap.N)
        im = ax.imshow(masked, cmap=legend_cmap, norm=norm, alpha=1.0, zorder=1)
    else:
        im = ax.imshow(masked, cmap=cmap, alpha=1.0, zorder=1, vmin=vmin, vmax=vmax)

    if points_gdf is not None:
        _add_points(ax, transform, points_gdf, label=points_label)

    if title:
        ax.set_title(title)

    if show_gridlines:
        _add_gridlines(ax, transform, height, width)
    else:
        ax.axis("off")

    if show_scalebar:
        _add_scalebar(ax, transform, width)

    if show_north_arrow:
        _add_north_arrow(ax)

    if discrete_legend:
        handles = [
            Patch(facecolor=legend_cmap(i), edgecolor="black", linewidth=0.3, label=discrete_legend[k])
            for i, k in enumerate(sorted_keys)
        ]
        ax.legend(
            handles=handles, loc="lower left", fontsize=6, framealpha=0.85,
            title=colorbar_label, title_fontsize=7,
        )
    elif colorbar_label:
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
    show_scalebar: bool = True,
    show_north_arrow: bool = True,
    show_gridlines: bool = True,
    points_gdf: Optional[gpd.GeoDataFrame] = None,
    points_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    discrete_legend: Optional[Dict[int, str]] = None,
) -> Path:
    """Convenience wrapper: render_with_terrain_backdrop + save to disk."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=figsize)
    render_with_terrain_backdrop(
        data_path, dem_path, cmap, ax,
        title=title, colorbar_label=colorbar_label,
        show_scalebar=show_scalebar, show_north_arrow=show_north_arrow,
        show_gridlines=show_gridlines,
        points_gdf=points_gdf, points_label=points_label,
        vmin=vmin, vmax=vmax, discrete_legend=discrete_legend,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
