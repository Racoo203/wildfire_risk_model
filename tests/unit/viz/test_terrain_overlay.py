"""Unit tests for viz/terrain_overlay.py's cartographic chrome (scale bar,
north arrow, coordinate gridlines) layered on top of the existing hillshade
backdrop rendering."""

from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.pyplot as plt

from wildfire_susceptibility.viz.terrain_overlay import (
    render_with_terrain_backdrop,
    save_terrain_map,
)


def _has_scalebar(ax):
    return any(isinstance(a, AnchoredSizeBar) for a in ax.artists)


def _has_north_arrow(ax):
    return any(t.get_text() == "N" for t in ax.texts)


def test_render_with_terrain_backdrop_adds_all_chrome_by_default(synthetic_dem, synthetic_reference_raster):
    fig, ax = plt.subplots()
    render_with_terrain_backdrop(synthetic_reference_raster, synthetic_dem, "viridis", ax)

    assert _has_scalebar(ax)
    assert _has_north_arrow(ax)
    assert len(ax.get_xticklabels()) > 0
    assert ax.get_xlabel() == "Easting (m)"
    assert ax.get_ylabel() == "Northing (m)"
    plt.close(fig)


def test_show_scalebar_false_suppresses_scalebar(synthetic_dem, synthetic_reference_raster):
    fig, ax = plt.subplots()
    render_with_terrain_backdrop(
        synthetic_reference_raster, synthetic_dem, "viridis", ax, show_scalebar=False,
    )
    assert not _has_scalebar(ax)
    plt.close(fig)


def test_show_north_arrow_false_suppresses_arrow(synthetic_dem, synthetic_reference_raster):
    fig, ax = plt.subplots()
    render_with_terrain_backdrop(
        synthetic_reference_raster, synthetic_dem, "viridis", ax, show_north_arrow=False,
    )
    assert not _has_north_arrow(ax)
    plt.close(fig)


def test_show_gridlines_false_falls_back_to_axis_off(synthetic_dem, synthetic_reference_raster):
    fig, ax = plt.subplots()
    render_with_terrain_backdrop(
        synthetic_reference_raster, synthetic_dem, "viridis", ax, show_gridlines=False,
    )
    assert ax.axison is False
    plt.close(fig)


def test_save_terrain_map_writes_nonempty_png(tmp_path, synthetic_dem, synthetic_reference_raster):
    out_path = tmp_path / "map.png"
    result = save_terrain_map(synthetic_reference_raster, synthetic_dem, out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
