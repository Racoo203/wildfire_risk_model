"""Unit tests for reporting/generate_report_figures.py's susceptibility-map
raster discovery and model-selection logic -- regression coverage for the
bug where the 'susceptibility map' category rendered stage_labels' ground-
truth label raster instead of a model's predicted output, and never
overlaid test-period fire points.

Also covers discover_feature_rasters -- regression coverage for the
fix-eda-feature-raster-generation branch, where this function was found to
be stale against the current feature builders: it looked for a
dist_activity.tif that ProximityBuilder no longer writes (replaced by
d_buildings + landuse_class), never listed landuse_class/d_buildings at
all, and assumed climate/NDVI rasters have no train/test split suffix when
ClimateBuilder/VegetationBuilder actually write one file per split."""

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio

from wildfire_susceptibility.reporting.generate_report_figures import (
    discover_predicted_susceptibility_rasters,
    discover_selected_model,
    discover_fire_test_points,
    discover_feature_rasters,
    generate_susceptibility_map,
)


def _write_dummy_raster(path, reference_transform, value: float = 2.0):
    """Uniform-value raster. Value doesn't need to span the full class
    range any more -- render_susceptibility_map now reads n_classes from
    cfg["labels"]["n_classes"] (see report_cfg fixture) rather than
    inferring it from the raster's own max value."""
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
        "labels": {"n_classes": 3},
        "data_sources": {"haduk": {"sources": ["tas", "tasmax", "tasmin", "rainfall", "sfcWind", "hurs"]}},
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
        "labels": {"n_classes": 3},
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
        "labels": {"n_classes": 3},
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
        "labels": {"n_classes": 3},
    }

    generate_susceptibility_map(cfg, "summer")  # should not raise

    assert not (figures_dir / "summer").exists()


class TestDiscoverFeatureRasters:
    def _touch(self, layers_dir: Path, name: str):
        layers_dir.mkdir(parents=True, exist_ok=True)
        (layers_dir / name).write_bytes(b"")

    def test_finds_static_features_including_landuse_class_and_d_buildings(self, report_cfg):
        """Regression: landuse_class and d_buildings replaced the old
        merged d_activity layer (see features/proximity.py's own
        docstring) but were never added as candidates -- silently dropping
        two real on-disk rasters from every factor-map run."""
        layers_dir = Path(report_cfg["base"]["output_dir"])
        for name in ("topo_elevation.tif", "topo_slope.tif", "topo_aspect.tif",
                     "dist_roads.tif", "dist_rivers.tif", "dist_buildings.tif", "landuse_class.tif"):
            self._touch(layers_dir, name)

        found = discover_feature_rasters(report_cfg, "summer")

        assert "landuse_class" in found
        assert "d_buildings" in found
        assert "d_activity" not in found  # dead layer, must not be a candidate at all

    def test_does_not_find_dead_d_activity_layer_even_if_present(self, report_cfg):
        layers_dir = Path(report_cfg["base"]["output_dir"])
        self._touch(layers_dir, "dist_activity.tif")  # stale leftover, if any

        found = discover_feature_rasters(report_cfg, "summer")

        assert "d_activity" not in found

    def test_climate_and_ndvi_use_the_split_suffixed_filename(self, report_cfg):
        """ClimateBuilder/VegetationBuilder write meteo_<var>_<season>_<split>.tif
        and ndvi_<season>_<split>.tif -- a candidate path without the split
        suffix would never match any file these builders actually produce."""
        layers_dir = Path(report_cfg["base"]["output_dir"])
        self._touch(layers_dir, "meteo_tasmax_summer_test.tif")
        self._touch(layers_dir, "meteo_tasmax_summer_train.tif")  # wrong split, must not be picked by default
        self._touch(layers_dir, "ndvi_summer_test.tif")

        found = discover_feature_rasters(report_cfg, "summer")

        assert found["tasmax"].name == "meteo_tasmax_summer_test.tif"
        assert found["ndvi"].name == "ndvi_summer_test.tif"

    def test_split_argument_selects_the_other_split(self, report_cfg):
        layers_dir = Path(report_cfg["base"]["output_dir"])
        self._touch(layers_dir, "meteo_tasmax_summer_train.tif")

        found = discover_feature_rasters(report_cfg, "summer", split="train")

        assert found["tasmax"].name == "meteo_tasmax_summer_train.tif"

    def test_d_fires_has_no_split_suffix(self, report_cfg):
        """d_fires is built once from fire_train regardless of split (see
        stage_labels.py) -- unlike climate/NDVI, it must not require a
        split suffix to be found."""
        layers_dir = Path(report_cfg["base"]["output_dir"])
        self._touch(layers_dir, "dist_fires_summer.tif")

        found = discover_feature_rasters(report_cfg, "summer")

        assert found["d_fires"].name == "dist_fires_summer.tif"

    def test_missing_files_are_skipped_not_raised(self, report_cfg):
        Path(report_cfg["base"]["output_dir"]).mkdir(parents=True)

        found = discover_feature_rasters(report_cfg, "summer")

        assert found == {}
