"""End-to-end regression coverage for the landuse_class categorical-encoding
fit/apply split (dataset_prep.py::fit_categorical_encoding /
apply_categorical_encoding).

Mirrors test_stage_train_evaluate_smoke.py's stage_train -> stage_evaluate
round trip, but adds a landuse_class column and includes CatBoost (native
categorical support) alongside random_forest (one-hot) in the model mix —
neither the original smoke test nor any other integration test exercised
landuse_class, which is exactly why the stage_evaluate categorical-layout
bug (raw/unencoded data reloaded from CSV never re-passed through the
training-time encoding) went uncaught. Confirms the full round trip —
manifest persistence, SHAP, and the full-domain susceptibility raster —
completes without crashing or a column-count/order mismatch for either
model family."""

import json

import numpy as np
import pandas as pd
import pytest
import rasterio

from wildfire_susceptibility.pipeline.stage_train import stage_train
from wildfire_susceptibility.pipeline.stage_evaluate import stage_evaluate


def _make_df(rng, n, landuse_codes=(0, 1, 2, 5)):
    return pd.DataFrame({
        "elevation": rng.normal(size=n),
        "slope": rng.normal(size=n),
        "ndvi": rng.normal(size=n),
        "landuse_class": rng.choice(landuse_codes, size=n).astype("float64"),
        "label": rng.integers(0, 4, size=n).astype(float),
        "_x": rng.uniform(0, 10_000, size=n),
        "_y": rng.uniform(0, 10_000, size=n),
    })


@pytest.fixture
def stage_config(tmp_path, monkeypatch, fast_modeling_config, synthetic_dem):
    import shutil

    monkeypatch.chdir(tmp_path)

    test_db = tmp_path / "mlflow_test.db"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.trainer.MLFLOW_TRACKING_URI",
        f"sqlite:///{test_db}",
    )

    config = fast_modeling_config
    config["base"]["figures_dir"] = str(tmp_path / "figures")
    config["modeling"]["models"] = ["random_forest", "catboost"]
    config["modeling"]["mlflow_experiment"] = "test-categorical-roundtrip"
    config["modeling"]["cv_strategy"] = "standard"
    config["modeling"]["excluded_features"] = []

    layers_dir = tmp_path / "layers"
    layers_dir.mkdir()
    shutil.copy(synthetic_dem, layers_dir / "topo_elevation.tif")
    config["base"]["output_dir"] = str(layers_dir)
    return config


@pytest.mark.slow
def test_categorical_encoding_survives_train_to_evaluate_round_trip(
    stage_config, synthetic_reference_raster, tmp_path
):
    rng = np.random.default_rng(3)

    train_csv = tmp_path / "dataset_train_summer.csv"
    test_csv = tmp_path / "dataset_test_summer.csv"
    full_csv = tmp_path / "dataset_full_summer.csv"

    _make_df(rng, 200).to_csv(train_csv, index=False)

    # "full" = every in-domain pixel of the reference raster (mirrors
    # stage_preprocessing's prepare_full_domain output); "test" = a smaller
    # labeled subset of it, same landuse_class code set as train.
    with rasterio.open(synthetic_reference_raster) as ref:
        data = ref.read(1)
        transform = ref.transform
    valid_rows, valid_cols = np.where(~np.isnan(data))
    n_domain = len(valid_rows)
    xs, ys = rasterio.transform.xy(transform, valid_rows, valid_cols)

    df_full = _make_df(rng, n_domain)
    df_full["_x"] = xs
    df_full["_y"] = ys
    df_full.to_csv(full_csv, index=False)
    df_full.iloc[:15].to_csv(test_csv, index=False)

    train_input = {
        "ref_path": synthetic_reference_raster,
        "summer": {"train": train_csv, "test": test_csv},
    }
    train_out = stage_train(stage_config, train_input)

    assert "random_forest" in train_out["summer"]
    assert "catboost" in train_out["summer"]

    rf_manifest = json.loads((train_out["summer"]["random_forest"] / "manifest.json").read_text())
    cb_manifest = json.loads((train_out["summer"]["catboost"] / "manifest.json").read_text())

    assert rf_manifest["categorical_encoding"]["kind"] == "onehot"
    assert rf_manifest["categorical_encoding"]["categories"]["landuse_class"] == [0, 1, 2, 5]
    assert cb_manifest["categorical_encoding"]["kind"] == "native"
    assert cb_manifest["categorical_encoding"]["columns"] == ["landuse_class"]

    eval_input = {
        "ref_path": synthetic_reference_raster,
        "summer": {
            "test": test_csv,
            "full": full_csv,
            "artifacts": train_out["summer"],
        },
    }
    eval_out = stage_evaluate(stage_config, eval_input)

    for model_name in ("random_forest", "catboost"):
        result = eval_out["summer"][model_name]
        assert result["full_metrics_path"] is not None
        assert result["shap_path"] is not None, (
            f"{model_name}: SHAP plot failed/was skipped — check for a "
            f"column-count or dtype mismatch against the persisted encoding."
        )

        raster_path = tmp_path / "layers" / f"susceptibility_{model_name}_summer.tif"
        assert raster_path.exists()
        with rasterio.open(raster_path) as src:
            raster_data = src.read(1)
        assert int(np.sum(~np.isnan(raster_data))) == n_domain, (
            f"{model_name}: full-domain raster must cover every in-domain pixel"
        )

        map_path = tmp_path / "figures" / "summer" / "susceptibility" / f"{model_name}.png"
        assert map_path.exists()
