"""End-to-end smoke test for stage_train -> stage_evaluate using tiny
synthetic on-disk datasets, confirming the stage-contract implementation
preserves the same training/evaluation behavior as the pre-refactor
in-process pipeline (pipeline/train.py + ModelTrainer.train_one)."""

import numpy as np
import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import Point

from wildfire_susceptibility.pipeline.stage_train import stage_train
from wildfire_susceptibility.pipeline.stage_evaluate import stage_evaluate


@pytest.fixture
def tiny_season_csvs(tmp_path):
    rng = np.random.default_rng(0)
    n = 200

    def _make_df():
        return pd.DataFrame({
            "elevation": rng.normal(size=n),
            "slope": rng.normal(size=n),
            "ndvi": rng.normal(size=n),
            "label": rng.integers(0, 4, size=n).astype(float),
            "_x": rng.uniform(0, 10_000, size=n),
            "_y": rng.uniform(0, 10_000, size=n),
        })

    train_path = tmp_path / "dataset_train_test_season.csv"
    test_path = tmp_path / "dataset_test_test_season.csv"
    _make_df().to_csv(train_path, index=False)
    _make_df().to_csv(test_path, index=False)
    return train_path, test_path


@pytest.fixture
def tiny_fire_gpkg(tmp_path):
    rng = np.random.default_rng(1)
    n = 10
    gdf = gpd.GeoDataFrame(
        geometry=[Point(x, y) for x, y in zip(rng.uniform(0, 10_000, n), rng.uniform(0, 10_000, n))],
        crs="EPSG:27700",
    )
    train_path = tmp_path / "fire_train.gpkg"
    test_path = tmp_path / "fire_test.gpkg"
    gdf.to_file(train_path, driver="GPKG")
    gdf.to_file(test_path, driver="GPKG")
    return train_path, test_path


@pytest.fixture
def stage_config(tmp_path, monkeypatch, fast_modeling_config):
    monkeypatch.chdir(tmp_path)  # isolate models/ artifact writes to tmp_path

    test_db = tmp_path / "mlflow_test.db"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.trainer.MLFLOW_TRACKING_URI",
        f"sqlite:///{test_db}",
    )

    config = fast_modeling_config
    config["base"]["figures_dir"] = str(tmp_path / "figures")
    config["modeling"]["models"] = ["random_forest", "svm"]
    config["modeling"]["mlflow_experiment"] = "test-stage-train-evaluate"
    config["modeling"]["cv_strategy"] = "standard"
    config["modeling"]["excluded_features"] = []
    return config


@pytest.mark.slow
def test_stage_train_then_stage_evaluate_round_trip(
    stage_config, tiny_season_csvs, tiny_fire_gpkg, tmp_path
):
    train_csv, test_csv = tiny_season_csvs
    fire_train, fire_test = tiny_fire_gpkg

    train_input = {
        "ref_path": tmp_path / "ref.tif",  # unused by evaluate.py's fail-soft paths at this scale
        "test_season": {
            "train": train_csv,
            "test": test_csv,
            "fire_train": fire_train,
            "fire_test": fire_test,
        },
    }

    train_out = stage_train(stage_config, train_input)

    assert "test_season" in train_out
    assert "random_forest" in train_out["test_season"]
    artifact_dir = train_out["test_season"]["random_forest"]
    assert (artifact_dir / "model.joblib").exists()
    assert (artifact_dir / "best_params.json").exists()
    assert (artifact_dir / "manifest.json").exists()

    import json
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    assert manifest["season"] == "test_season"
    assert manifest["model_name"] == "random_forest"
    assert "cfg_sig" in manifest
    assert "config_snapshot" in manifest
    assert 0.0 <= manifest["val_auc"] <= 1.0

    eval_input = {
        "ref_path": tmp_path / "ref.tif",
        "test_season": {
            "test": test_csv,
            "fire_test": fire_test,
            "artifacts": train_out["test_season"],
        },
    }

    eval_out = stage_evaluate(stage_config, eval_input)

    assert "test_season" in eval_out
    assert "random_forest" in eval_out["test_season"]
    assert "svm" in eval_out["test_season"]
    result = eval_out["test_season"]["random_forest"]
    assert "full_metrics_path" in result

    # ROC comparison across models — regression coverage for the previously-
    # unwired viz.plot_roc_curves() call.
    roc_path = tmp_path / "figures" / "test_season" / "models" / "roc_comparison.png"
    assert roc_path.exists()