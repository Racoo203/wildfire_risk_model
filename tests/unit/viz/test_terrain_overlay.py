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


def test_vmin_vmax_fix_the_imshow_color_range(synthetic_dem, synthetic_reference_raster):
    """vmin/vmax let a caller (e.g. render_susceptibility_map) pin the color
    scale to a fixed class range instead of matplotlib's default of scaling
    to this raster's own data min/max -- needed so a raster missing a class
    doesn't shift the color mapping relative to one where every class is
    present."""
    fig, ax = plt.subplots()
    render_with_terrain_backdrop(
        synthetic_reference_raster, synthetic_dem, "viridis", ax, vmin=0, vmax=4,
    )
    im = [c for c in ax.get_images()][-1]
    assert im.get_clim() == (0, 4)
    plt.close(fig)


def test_save_terrain_map_writes_nonempty_png(tmp_path, synthetic_dem, synthetic_reference_raster):
    out_path = tmp_path / "map.png"
    result = save_terrain_map(synthetic_reference_raster, synthetic_dem, out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_points_gdf_overlays_scatter_and_legend(synthetic_dem, synthetic_reference_raster, synthetic_fire_points):
    fig, ax = plt.subplots()
    render_with_terrain_backdrop(
        synthetic_reference_raster, synthetic_dem, "viridis", ax,
        points_gdf=synthetic_fire_points, points_label="Test-period fires",
    )

    collections_with_offsets = [c for c in ax.collections if len(c.get_offsets()) > 0]
    assert len(collections_with_offsets) >= 1
    assert collections_with_offsets[0].get_offsets().shape[0] == len(synthetic_fire_points)

    legend = ax.get_legend()
    assert legend is not None
    assert legend.get_texts()[0].get_text() == "Test-period fires"
    plt.close(fig)


def test_no_points_gdf_means_no_legend(synthetic_dem, synthetic_reference_raster):
    fig, ax = plt.subplots()
    render_with_terrain_backdrop(synthetic_reference_raster, synthetic_dem, "viridis", ax)

    assert ax.get_legend() is None
    plt.close(fig)


def test_points_are_small_and_highly_opaque(synthetic_dem, synthetic_reference_raster, synthetic_fire_points):
    """Fire-incident circles should read as small, solid, high-opacity
    markers -- not the previous larger hollow-ring style."""
    fig, ax = plt.subplots()
    render_with_terrain_backdrop(
        synthetic_reference_raster, synthetic_dem, "viridis", ax,
        points_gdf=synthetic_fire_points, points_label="Test-period fires",
    )

    collections_with_offsets = [c for c in ax.collections if len(c.get_offsets()) > 0]
    points_collection = collections_with_offsets[0]

    assert points_collection.get_sizes()[0] < 20  # smaller than the old s=20
    assert points_collection.get_alpha() is not None and points_collection.get_alpha() >= 0.8
    facecolor = points_collection.get_facecolor()[0]
    assert facecolor[3] > 0.0  # filled, not fully transparent facecolor="none"
    plt.close(fig)


def test_save_terrain_map_with_points_gdf(tmp_path, synthetic_dem, synthetic_reference_raster, synthetic_fire_points):
    out_path = tmp_path / "map_with_points.png"
    result = save_terrain_map(
        synthetic_reference_raster, synthetic_dem, out_path,
        points_gdf=synthetic_fire_points, points_label="Test-period fires",
    )

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
