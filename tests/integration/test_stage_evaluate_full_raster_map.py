"""Integration coverage: stage_evaluate must (1) predict the
susceptibility raster over the full in-domain pixel set -- not just the
label-filtered test split -- and (2) auto-render the terrain-backdrop PNG
map, without requiring a separate manual reporting-script run."""

import shutil

import numpy as np
import pandas as pd
import pytest
import rasterio

from wildfire_susceptibility.pipeline.stage_train import stage_train
from wildfire_susceptibility.pipeline.stage_evaluate import stage_evaluate


@pytest.fixture
def stage_config(tmp_path, monkeypatch, fast_modeling_config, synthetic_dem):
    monkeypatch.chdir(tmp_path)

    test_db = tmp_path / "mlflow_test.db"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.trainer.MLFLOW_TRACKING_URI",
        f"sqlite:///{test_db}",
    )

    config = fast_modeling_config
    config["base"]["figures_dir"] = str(tmp_path / "figures")
    config["modeling"]["models"] = ["random_forest"]
    config["modeling"]["mlflow_experiment"] = "test-full-raster-map"
    config["modeling"]["cv_strategy"] = "standard"
    config["modeling"]["excluded_features"] = []

    # stage_evaluate looks for the DEM at <output_dir>/topo_elevation.tif
    layers_dir = tmp_path / "layers"
    layers_dir.mkdir()
    shutil.copy(synthetic_dem, layers_dir / "topo_elevation.tif")
    config["base"]["output_dir"] = str(layers_dir)
    return config


@pytest.mark.slow
def test_stage_evaluate_predicts_full_domain_and_renders_map(
    stage_config, synthetic_reference_raster, tmp_path
):
    rng = np.random.default_rng(0)

    def _random_rows(n):
        return pd.DataFrame({
            "elevation": rng.normal(size=n),
            "slope": rng.normal(size=n),
            "ndvi": rng.normal(size=n),
            "label": rng.integers(0, 4, size=n).astype(float),
            "_x": rng.uniform(0, 10_000, size=n),
            "_y": rng.uniform(0, 10_000, size=n),
        })

    train_csv = tmp_path / "dataset_train_summer.csv"
    test_csv = tmp_path / "dataset_test_summer.csv"
    full_csv = tmp_path / "dataset_full_summer.csv"
    _random_rows(200).to_csv(train_csv, index=False)

    # "full" = every in-domain pixel of the reference raster; "test" = a
    # small labeled subset of it -- mirrors the real prepare_test vs
    # prepare_full_domain split produced by stage_preprocessing.
    with rasterio.open(synthetic_reference_raster) as ref:
        data = ref.read(1)
        transform = ref.transform
    valid_rows, valid_cols = np.where(~np.isnan(data))
    n_domain = len(valid_rows)
    xs, ys = rasterio.transform.xy(transform, valid_rows, valid_cols)

    df_full = pd.DataFrame({
        "elevation": rng.normal(size=n_domain),
        "slope": rng.normal(size=n_domain),
        "ndvi": rng.normal(size=n_domain),
        "label": (np.arange(n_domain) % 4).astype(float),  # all 4 classes present, deterministic
        "_x": xs,
        "_y": ys,
    })
    df_full.to_csv(full_csv, index=False)
    df_full.iloc[:10].to_csv(test_csv, index=False)  # smaller "labeled" subset

    train_input = {
        "ref_path": synthetic_reference_raster,
        "summer": {"train": train_csv, "test": test_csv},
    }
    train_out = stage_train(stage_config, train_input)

    eval_input = {
        "ref_path": synthetic_reference_raster,
        "summer": {
            "test": test_csv,
            "full": full_csv,
            "artifacts": train_out["summer"],
        },
    }
    stage_evaluate(stage_config, eval_input)

    raster_path = tmp_path / "layers" / "susceptibility_random_forest_summer.tif"
    assert raster_path.exists()
    with rasterio.open(raster_path) as src:
        raster_data = src.read(1)
    assert int(np.sum(~np.isnan(raster_data))) == n_domain, (
        "raster must cover every in-domain pixel, not just the 10-row "
        "labeled test subset"
    )

    map_path = tmp_path / "figures" / "summer" / "susceptibility" / "random_forest.png"
    assert map_path.exists(), "susceptibility map PNG must be auto-rendered by stage_evaluate"
