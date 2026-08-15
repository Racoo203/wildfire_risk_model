"""Unit tests for viz/maps.py::render_susceptibility_map's class-count
handling — regression coverage for the bug where `n_classes` was inferred
from the raster's own predicted values (`nanmax(labels_data) + 1`) instead
of the fixed schema in config (`labels.n_classes`). A raster whose
predictions happen to collapse to fewer classes than the configured schema
(e.g. zero "High" pixels this season) must still render successfully with
the full, correctly-labeled legend — not raise
`No class-name scheme defined for n_classes=<collapsed count>`."""

from pathlib import Path

import numpy as np
import pytest
import rasterio

from wildfire_susceptibility.viz import maps
from wildfire_susceptibility.modeling.class_labels import class_names_for


def _write_labels_raster(path: Path, transform, max_class: int):
    """A small labels raster whose classes run 0..max_class (inclusive)."""
    size = 10
    data = np.tile(np.arange(size, dtype="float32") % (max_class + 1), (size, 1))
    meta = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:27700",
        "transform": transform,
        "nodata": np.nan,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data[np.newaxis, :, :])
    return path


@pytest.fixture
def _capture_save_terrain_map(monkeypatch):
    captured = {}

    def _fake_save_terrain_map(data_path, dem_path, out_path, **kwargs):
        captured.update(kwargs)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.touch()
        return out_path

    monkeypatch.setattr(maps, "save_terrain_map", _fake_save_terrain_map)
    return captured


@pytest.mark.parametrize("n_classes", [3, 4, 5])
def test_susceptibility_map_colorbar_label_matches_configured_n_classes(
    tmp_path, reference_transform, synthetic_dem, n_classes, _capture_save_terrain_map,
):
    """Full-range raster (every configured class present) — the label
    should match the configured n_classes, same as before this fix."""
    labels_path = _write_labels_raster(tmp_path / "labels.tif", reference_transform, n_classes - 1)

    maps.render_susceptibility_map(
        labels_path, synthetic_dem, tmp_path / "figures",
        n_classes=n_classes, season="summer", model_name="catboost",
    )

    expected_names = class_names_for(n_classes)
    expected_label = f"Class (0={expected_names[0]} .. {n_classes - 1}={expected_names[-1]})"
    assert _capture_save_terrain_map["colorbar_label"] == expected_label


def test_susceptibility_map_5class_label_is_not_the_stale_4class_string(
    tmp_path, reference_transform, synthetic_dem, _capture_save_terrain_map,
):
    """The literal regression this bug caused: a 5-class raster (the active
    baseline.yaml config) rendered a colorbar claiming '3=Very High' as the
    top class, silently misrepresenting the real top class (index 4)."""
    labels_path = _write_labels_raster(tmp_path / "labels.tif", reference_transform, 4)

    maps.render_susceptibility_map(
        labels_path, synthetic_dem, tmp_path / "figures", n_classes=5, season="summer",
    )

    assert _capture_save_terrain_map["colorbar_label"] != "Class (0=Low .. 3=Very High)"
    assert _capture_save_terrain_map["colorbar_label"] == "Class (0=Very Low .. 4=Very High)"


def test_susceptibility_map_renders_when_predictions_collapse_below_configured_classes(
    tmp_path, reference_transform, synthetic_dem, _capture_save_terrain_map,
):
    """The actual bug: a raster whose predictions only span 2 of the 3
    configured classes (e.g. no pixels predicted "High" this season) used
    to infer n_classes=2 from the data and crash inside class_names_for().
    It must instead render using the full configured 3-class schema."""
    labels_path = _write_labels_raster(tmp_path / "labels.tif", reference_transform, max_class=1)

    out_path = maps.render_susceptibility_map(
        labels_path, synthetic_dem, tmp_path / "figures", n_classes=3, season="spring",
    )

    assert out_path.exists()
    expected_names = class_names_for(3)
    assert _capture_save_terrain_map["colorbar_label"] == (
        f"Class (0={expected_names[0]} .. 2={expected_names[-1]})"
    )


def test_susceptibility_map_uses_fixed_color_range_regardless_of_which_classes_are_present(
    tmp_path, reference_transform, synthetic_dem, _capture_save_terrain_map,
):
    """The color scale (and therefore the legend) must span the full
    configured class range even when a class has zero pixels — otherwise a
    collapsed raster would silently remap its present classes onto the
    full color range and misrepresent them, even after the n_classes fix
    above stops the crash."""
    labels_path = _write_labels_raster(tmp_path / "labels.tif", reference_transform, max_class=1)

    maps.render_susceptibility_map(
        labels_path, synthetic_dem, tmp_path / "figures", n_classes=3, season="spring",
    )

    assert _capture_save_terrain_map["vmin"] == 0
    assert _capture_save_terrain_map["vmax"] == 2
