"""End-to-end regression coverage for the scaler-persistence fix
(fix-ordinal-lr-scaler-persistence): trainer.py used to fit a StandardScaler
at training time and immediately discard it
(`X_tr, X_va, _ = dataset_prep.scale_for_model_family(...)`), so any
scale-sensitive model (needs_scaling() -> True: ordinal_lr, neural_net) was
evaluated in stage_evaluate.py against raw, unscaled features reloaded fresh
from CSV -- completely different from the StandardScaler-transformed data it
was actually fit on. RF/CatBoost (needs_scaling() -> False) were unaffected.

Mirrors test_stage_train_evaluate_categorical_roundtrip.py's
fit-time-persist / eval-time-reload-and-apply pattern, but for feature
scaling instead of categorical encoding.
"""
import json

import numpy as np
import pandas as pd
import pytest

from wildfire_susceptibility.modeling.dataset_prep import DatasetPrep
from wildfire_susceptibility.pipeline.stage_train import stage_train
from wildfire_susceptibility.pipeline.stage_evaluate import stage_evaluate


def _make_df(rng, n):
    return pd.DataFrame({
        "elevation": rng.normal(size=n),
        "slope": rng.normal(size=n),
        "ndvi": rng.normal(size=n),
        "label": rng.integers(0, 4, size=n).astype(float),
        "_x": rng.uniform(0, 10_000, size=n),
        "_y": rng.uniform(0, 10_000, size=n),
    })


@pytest.fixture
def stage_config(tmp_path, monkeypatch, fast_modeling_config):
    monkeypatch.chdir(tmp_path)

    test_db = tmp_path / "mlflow_test.db"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.trainer.MLFLOW_TRACKING_URI",
        f"sqlite:///{test_db}",
    )

    config = fast_modeling_config
    config["base"]["figures_dir"] = str(tmp_path / "figures")
    config["modeling"]["models"] = ["random_forest", "ordinal_lr"]
    config["modeling"]["mlflow_experiment"] = "test-scaler-persistence"
    config["modeling"]["cv_strategy"] = "standard"
    config["modeling"]["excluded_features"] = []
    return config


@pytest.fixture
def trained_artifacts(stage_config, tmp_path):
    rng = np.random.default_rng(5)
    train_csv = tmp_path / "dataset_train_summer.csv"
    test_csv = tmp_path / "dataset_test_summer.csv"
    _make_df(rng, 200).to_csv(train_csv, index=False)
    _make_df(rng, 40).to_csv(test_csv, index=False)

    train_input = {
        # Not backed by a real file -- evaluate.py's SHAP/time-forward paths
        # fail soft when ref_path doesn't resolve to a real raster, same as
        # test_stage_train_evaluate_smoke.py's fixture at this scale.
        "ref_path": tmp_path / "ref.tif",
        "summer": {"train": train_csv, "test": test_csv},
    }
    train_out = stage_train(stage_config, train_input)
    return stage_config, train_out, test_csv, tmp_path


@pytest.mark.slow
class TestScalerPersistedAtTrainingTime:
    def test_ordinal_lr_manifest_and_scaler_artifact(self, trained_artifacts):
        _, train_out, _, _ = trained_artifacts
        artifact_dir = train_out["summer"]["ordinal_lr"]

        assert (artifact_dir / "scaler.joblib").exists()
        manifest = json.loads((artifact_dir / "manifest.json").read_text())
        assert manifest["scaling"]["needs_scaling"] is True
        assert manifest["scaling"]["scaler_path"] == "scaler.joblib"
        assert set(manifest["scaling"]["columns"]) == {"elevation", "slope", "ndvi"}

    def test_random_forest_writes_no_scaler(self, trained_artifacts):
        _, train_out, _, _ = trained_artifacts
        artifact_dir = train_out["summer"]["random_forest"]

        assert not (artifact_dir / "scaler.joblib").exists()
        manifest = json.loads((artifact_dir / "manifest.json").read_text())
        assert manifest["scaling"]["needs_scaling"] is False
        assert manifest["scaling"]["columns"] == []


@pytest.mark.slow
class TestScalerAppliedAtEvalTime:
    def test_apply_scaling_called_for_ordinal_lr_not_random_forest(
        self, trained_artifacts, monkeypatch
    ):
        stage_config, train_out, test_csv, tmp_path = trained_artifacts
        calls = []
        original = DatasetPrep.apply_scaling

        def _spy(self, X, scaler):
            calls.append(scaler)
            return original(self, X, scaler)

        monkeypatch.setattr(DatasetPrep, "apply_scaling", _spy)

        eval_input = {
            "ref_path": tmp_path / "ref.tif",
            "summer": {"test": test_csv, "artifacts": train_out["summer"]},
        }
        stage_evaluate(stage_config, eval_input)

        # Only "test" supplied (no "full") -> exactly one apply_scaling
        # call, and only for ordinal_lr; random_forest's eval path must
        # never touch scaling at all.
        assert len(calls) == 1

    def test_missing_scaler_on_pre_fix_artifact_fails_loudly(self, trained_artifacts):
        """Simulates an artifact trained before this fix: manifest has no
        "scaling" block at all (the pre-fix manifest schema) and no
        scaler.joblib on disk. stage_evaluate must raise clearly -- driven
        by the reloaded model's own needs_scaling(), not the manifest --
        rather than silently evaluating ordinal_lr against unscaled data."""
        stage_config, train_out, test_csv, tmp_path = trained_artifacts
        artifact_dir = train_out["summer"]["ordinal_lr"]

        manifest_path = artifact_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        del manifest["scaling"]
        manifest_path.write_text(json.dumps(manifest))
        (artifact_dir / "scaler.joblib").unlink()

        eval_input = {
            "ref_path": tmp_path / "ref.tif",
            "summer": {"test": test_csv, "artifacts": {"ordinal_lr": artifact_dir}},
        }
        with pytest.raises(FileNotFoundError, match="scaler"):
            stage_evaluate(stage_config, eval_input)
