"""Guards against the 'CategoricalDistribution does not support dynamic
value space' failure: changing a model's param_space() must produce a
different study name, never reuse a study with incompatible persisted
distributions."""

import optuna
import pytest


def _dummy_model_factory(kernel_choices):
    class _DummySVM:
        def param_space(self, trial):
            return {"kernel": trial.suggest_categorical("kernel", kernel_choices)}
    return _DummySVM


def test_param_space_signature_changes_with_categorical_choices():
    from wildfire_susceptibility.modeling.training.search import HyperparamSearch

    sig_old = HyperparamSearch._param_space_signature(_dummy_model_factory(["rbf", "poly"]))
    sig_new = HyperparamSearch._param_space_signature(_dummy_model_factory(["rbf", "linear"]))

    assert sig_old != sig_new, (
        "Changing categorical choices in param_space() must change the "
        "signature — otherwise a stale study with incompatible distributions "
        "will be reused and Optuna will raise on the first differing trial."
    )


def test_param_space_signature_stable_for_unchanged_space():
    from wildfire_susceptibility.modeling.training.search import HyperparamSearch

    sig_a = HyperparamSearch._param_space_signature(_dummy_model_factory(["rbf", "linear"]))
    sig_b = HyperparamSearch._param_space_signature(_dummy_model_factory(["rbf", "linear"]))

    assert sig_a == sig_b  # same space -> same study reused, trials preserved


def test_get_or_create_study_survives_param_space_change(tmp_path, monkeypatch, fast_modeling_config):
    """..."""
    from wildfire_susceptibility.modeling.training.search import HyperparamSearch
    from wildfire_susceptibility.modeling.cv import FoldStrategy
    from wildfire_susceptibility.modeling.training.resampling import SMOTEResampler
    from wildfire_susceptibility.core.registry import MODELS
    from wildfire_susceptibility.modeling import models  # noqa: F401 — registers svm

    db_path = tmp_path / "optuna_test.db"
    monkeypatch.setattr(
        "wildfire_susceptibility.modeling.training.search.OPTUNA_STORAGE",
        f"sqlite:///{db_path}",
    )

    config = fast_modeling_config
    config["modeling"]["optuna_n_trials"] = 1
    resampler = SMOTEResampler(config)
    fold_strategy = FoldStrategy(config, resampler)
    search = HyperparamSearch(config, fold_strategy)

    # First call establishes a study under today's param_space.
    study1 = search.get_or_create_study("test_season", "svm")
    assert study1.trials == []

    original_param_space = MODELS["svm"].param_space

    def _changed_param_space(self, trial):
        return {"kernel": trial.suggest_categorical("kernel", ["rbf", "poly", "sigmoid"])}

    monkeypatch.setattr(MODELS["svm"], "param_space", _changed_param_space)
    try:
        study2 = search.get_or_create_study("test_season", "svm")  # must not raise
        assert study2.study_name != study1.study_name
    finally:
        monkeypatch.setattr(MODELS["svm"], "param_space", original_param_space)