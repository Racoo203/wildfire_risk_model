"""Unit tests for viz/maps.py::render_susceptibility_map's colorbar label —
regression coverage for the same hardcoded-4-class bug already fixed in
metrics.py, evaluate.py, training/evaluation.py, and
charts.py::plot_class_balance (all now derive class names from
class_labels.py::class_names_for()). This file was missed in that pass:
the colorbar label was hardcoded to "Class (0=Low .. 3=Very High)"
regardless of the raster's actual class count."""

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


@pytest.mark.parametrize("n_classes", [3, 4, 5])
def test_susceptibility_map_colorbar_label_matches_configured_n_classes(
    tmp_path, monkeypatch, reference_transform, synthetic_dem, n_classes
):
    labels_path = _write_labels_raster(tmp_path / "labels.tif", reference_transform, n_classes - 1)

    captured = {}

    def _fake_save_terrain_map(data_path, dem_path, out_path, **kwargs):
        captured["colorbar_label"] = kwargs.get("colorbar_label")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.touch()
        return out_path

    monkeypatch.setattr(maps, "save_terrain_map", _fake_save_terrain_map)

    maps.render_susceptibility_map(
        labels_path, synthetic_dem, tmp_path / "figures", season="summer", model_name="catboost",
    )

    expected_names = class_names_for(n_classes)
    expected_label = f"Class (0={expected_names[0]} .. {n_classes - 1}={expected_names[-1]})"
    assert captured["colorbar_label"] == expected_label


def test_susceptibility_map_5class_label_is_not_the_stale_4class_string(
    tmp_path, monkeypatch, reference_transform, synthetic_dem
):
    """The literal regression this bug caused: a 5-class raster (the active
    baseline.yaml config) rendered a colorbar claiming '3=Very High' as the
    top class, silently misrepresenting the real top class (index 4)."""
    labels_path = _write_labels_raster(tmp_path / "labels.tif", reference_transform, 4)

    captured = {}

    def _fake_save_terrain_map(data_path, dem_path, out_path, **kwargs):
        captured["colorbar_label"] = kwargs.get("colorbar_label")
        return Path(out_path)

    monkeypatch.setattr(maps, "save_terrain_map", _fake_save_terrain_map)

    maps.render_susceptibility_map(labels_path, synthetic_dem, tmp_path / "figures", season="summer")

    assert captured["colorbar_label"] != "Class (0=Low .. 3=Very High)"
    assert captured["colorbar_label"] == "Class (0=Very Low .. 4=Very High)"
