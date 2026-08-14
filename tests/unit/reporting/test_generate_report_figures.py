"""Unit tests for reporting/generate_report_figures.py's susceptibility-map
raster discovery and model-selection logic — regression coverage for the
bug where the 'susceptibility map' category rendered stage_labels' ground-
truth label raster instead of a model's predicted output, and never
overlaid test-period fire points."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from wildfire_susceptibility.reporting.generate_report_figures import (
    discover_predicted_susceptibility_rasters,
    discover_selected_model,
    discover_fire_test_points,
    generate_susceptibility_map,
)


def _write_dummy_raster(path, reference_transform, value: float = 2.0):
    """Uniform-value raster. Default value=2.0 (not 0.0/1.0) so the max
    class index is 2 -> class_names_for(3) resolves to a real scheme;
    render_susceptibility_map derives n_classes from the raster's own
    max value, and class_names_for only supports {3, 4, 5}."""
    meta = {
        "driver": "GTiff", "height": 10, "width": 10, "count": 1,
        "dtype": "float32", "crs": "EPSG:27700",
        "transform": reference_transform, "nodata": np.nan,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(np.full((10, 10), value, dtype="float32")[np.newaxis, :, :])


@pytest.fixture
def report_cfg(tmp_path):
    return {
        "base": {
            "output_dir": str(tmp_path / "layers"),
            "figures_dir": str(tmp_path / "figures"),
        },
        "labels": {"include_d_fires_as_feature": False},
    }


def test_discover_predicted_susceptibility_rasters_finds_all_models(report_cfg, reference_transform):
    layers_dir = Path(report_cfg["base"]["output_dir"])
    layers_dir.mkdir(parents=True)
    _write_dummy_raster(layers_dir / "susceptibility_random_forest_summer.tif", reference_transform)
    _write_dummy_raster(layers_dir / "susceptibility_catboost_summer.tif", reference_transform)
    _write_dummy_raster(layers_dir / "susceptibility_random_forest_spring.tif", reference_transform)  # other season

    found = discover_predicted_susceptibility_rasters(report_cfg, "summer")

    assert set(found.keys()) == {"random_forest", "catboost"}
    assert found["random_forest"].name == "susceptibility_random_forest_summer.tif"


def test_discover_predicted_susceptibility_rasters_empty_when_none_exist(report_cfg):
    Path(report_cfg["base"]["output_dir"]).mkdir(parents=True)
    assert discover_predicted_susceptibility_rasters(report_cfg, "summer") == {}


def test_discover_selected_model_reads_selection_summary(report_cfg):
    figures_dir = Path(report_cfg["base"]["figures_dir"])
    figures_dir.mkdir(parents=True)
    (figures_dir / "selection_summary.json").write_text(json.dumps({
        "summer": {"selected_model": "catboost"},
    }))

    assert discover_selected_model(report_cfg, "summer") == "catboost"
    assert discover_selected_model(report_cfg, "spring") is None  # season not in summary


def test_discover_selected_model_returns_none_when_no_summary_yet(report_cfg):
    assert discover_selected_model(report_cfg, "summer") is None


def test_discover_fire_test_points_finds_gpkg(report_cfg):
    layers_dir = Path(report_cfg["base"]["output_dir"])
    layers_dir.mkdir(parents=True)
    (layers_dir / "fire_points_test_summer.gpkg").write_bytes(b"")

    assert discover_fire_test_points(report_cfg, "summer").name == "fire_points_test_summer.gpkg"
    assert discover_fire_test_points(report_cfg, "spring") is None


def test_generate_susceptibility_map_renders_only_selected_model(
    tmp_path, synthetic_dem, reference_transform, synthetic_fire_points,
):
    layers_dir = tmp_path / "layers"
    figures_dir = tmp_path / "figures"
    layers_dir.mkdir()
    figures_dir.mkdir()

    (layers_dir / "topo_elevation.tif").write_bytes(Path(synthetic_dem).read_bytes())
    _write_dummy_raster(layers_dir / "susceptibility_random_forest_summer.tif", reference_transform)
    _write_dummy_raster(layers_dir / "susceptibility_catboost_summer.tif", reference_transform)
    (figures_dir / "selection_summary.json").write_text(json.dumps({
        "summer": {"selected_model": "catboost"},
    }))
    synthetic_fire_points.to_file(layers_dir / "fire_points_test_summer.gpkg", driver="GPKG")

    cfg = {
        "base": {"output_dir": str(layers_dir), "figures_dir": str(figures_dir)},
        "labels": {"include_d_fires_as_feature": False},
    }

    generate_susceptibility_map(cfg, "summer")

    assert (figures_dir / "summer" / "susceptibility" / "catboost.png").exists()
    assert not (figures_dir / "summer" / "susceptibility" / "random_forest.png").exists()


def test_generate_susceptibility_map_falls_back_to_all_models_without_selection(
    tmp_path, synthetic_dem, reference_transform,
):
    layers_dir = tmp_path / "layers"
    figures_dir = tmp_path / "figures"
    layers_dir.mkdir()

    (layers_dir / "topo_elevation.tif").write_bytes(Path(synthetic_dem).read_bytes())
    _write_dummy_raster(layers_dir / "susceptibility_random_forest_summer.tif", reference_transform)
    _write_dummy_raster(layers_dir / "susceptibility_catboost_summer.tif", reference_transform)

    cfg = {
        "base": {"output_dir": str(layers_dir), "figures_dir": str(figures_dir)},
        "labels": {"include_d_fires_as_feature": False},
    }

    generate_susceptibility_map(cfg, "summer")

    assert (figures_dir / "summer" / "susceptibility" / "random_forest.png").exists()
    assert (figures_dir / "summer" / "susceptibility" / "catboost.png").exists()


def test_generate_susceptibility_map_skips_when_no_predicted_rasters(tmp_path, synthetic_dem):
    layers_dir = tmp_path / "layers"
    figures_dir = tmp_path / "figures"
    layers_dir.mkdir()
    (layers_dir / "topo_elevation.tif").write_bytes(Path(synthetic_dem).read_bytes())

    cfg = {
        "base": {"output_dir": str(layers_dir), "figures_dir": str(figures_dir)},
        "labels": {"include_d_fires_as_feature": False},
    }

    generate_susceptibility_map(cfg, "summer")  # should not raise

    assert not (figures_dir / "summer").exists()
