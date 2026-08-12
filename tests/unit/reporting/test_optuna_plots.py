"""Unit tests for the Optuna report-plot wrapper functions in viz/charts.py
and the study-name reconstruction generate_report_figures.py uses to find
them — regression coverage for the gap where no code anywhere called
optuna.visualization, even though studies were already fully queryable
from the existing SQLite backend without any new instrumentation."""

import json

import optuna
import pytest

from wildfire_susceptibility import viz
from wildfire_susceptibility.core.registry import MODELS
from wildfire_susceptibility.modeling import models as _models  # noqa: F401 — registers wrappers
from wildfire_susceptibility.modeling.training.search import HyperparamSearch, OBJECTIVE_VERSION


def _make_tiny_study(storage_url: str, study_name: str) -> optuna.Study:
    study = optuna.create_study(direction="maximize", storage=storage_url, study_name=study_name)

    def objective(trial):
        x = trial.suggest_float("x", -5, 5)
        return -((x - 1) ** 2)

    study.optimize(objective, n_trials=8)
    return study


def test_plot_optuna_optimization_history_writes_and_indexes_png(tmp_path):
    storage_url = f"sqlite:///{tmp_path / 'optuna_test.db'}"
    study = _make_tiny_study(storage_url, "history_test")
    figures_dir = tmp_path / "figures"

    out_path = viz.plot_optuna_optimization_history(
        study, figures_dir, season="summer", model_name="random_forest"
    )

    assert out_path.exists()
    manifest = json.loads((figures_dir / "manifest.json").read_text())
    assert any(e["category"] == "optuna_optimization_history" for e in manifest)


def test_plot_optuna_param_importances_writes_and_indexes_png(tmp_path):
    storage_url = f"sqlite:///{tmp_path / 'optuna_test.db'}"
    study = _make_tiny_study(storage_url, "importance_test")
    figures_dir = tmp_path / "figures"

    out_path = viz.plot_optuna_param_importances(
        study, figures_dir, season="summer", model_name="random_forest"
    )

    assert out_path.exists()
    manifest = json.loads((figures_dir / "manifest.json").read_text())
    assert any(e["category"] == "optuna_param_importance" for e in manifest)


def test_reconstructed_study_name_matches_get_or_create_study(tmp_path, monkeypatch):
    """Guards against generate_report_figures.py's study-name reconstruction
    drifting from HyperparamSearch.get_or_create_study()'s actual naming
    scheme — both must derive from the same OBJECTIVE_VERSION constant and
    _param_space_signature() static method rather than each hardcoding
    their own copy of the formula."""
    storage_url = f"sqlite:///{tmp_path / 'optuna_test.db'}"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.search.OPTUNA_STORAGE", storage_url,
    )

    config = {"modeling": {"optuna_n_trials": 1}}
    search = HyperparamSearch(config, cv_strategy=None)
    study = search.get_or_create_study("summer", "random_forest")

    model_cls = MODELS["random_forest"]
    sig = HyperparamSearch._param_space_signature(model_cls)
    reconstructed_name = f"summer_random_forest_{sig}_{OBJECTIVE_VERSION}"

    assert reconstructed_name == study.study_name

    reloaded = optuna.load_study(study_name=reconstructed_name, storage=storage_url)
    assert reloaded.study_name == study.study_name
