import numpy as np
from wildfire_susceptibility.modeling.resampling import SMOTEResampler

def test_smote_disabled_passthrough(minimal_modeling_config):
    from wildfire_susceptibility.modeling.resampling import SMOTEResampler
    config = minimal_modeling_config
    config["modeling"]["use_smote"] = False
    resampler = SMOTEResampler(config)
    X, y = np.zeros((10, 3)), np.array([0]*8 + [1]*2)
    X_out, y_out = resampler.resample(X, y)
    assert X_out is X and y_out is y


def test_smote_skips_on_degenerate_class(minimal_modeling_config):
    from wildfire_susceptibility.modeling.resampling import SMOTEResampler
    config = minimal_modeling_config
    config["modeling"]["use_smote"] = True
    config["modeling"]["smote_k_neighbors"] = 5
    resampler = SMOTEResampler(config)
    X, y = np.zeros((10, 3)), np.array([0]*9 + [1])  # minority class has 1 sample
    X_out, y_out = resampler.resample(X, y, context="test")
    assert len(y_out) == len(y)  # unchanged, not resampled


def test_cost_weighted_forces_smote_off_even_when_use_smote_true(minimal_modeling_config):
    """The whole point of imbalance_strategy='cost_weighted' is that it's
    mutually exclusive with SMOTE, to avoid double-correcting for the same
    class imbalance. use_smote=True alone must not be enough to trigger
    resampling once a model_name resolves to 'cost_weighted' — this is the
    gate cv/base.py's fit_and_score_full and trainer.py's final refit both
    rely on."""
    from wildfire_susceptibility.modeling.resampling import SMOTEResampler
    config = minimal_modeling_config
    config["modeling"]["use_smote"] = True  # would enable SMOTE under the old/default path
    # minimal_modeling_config dumps the schema's real default (None) for
    # smote_sampling_strategy explicitly, which imblearn's SMOTE rejects as
    # a sampling_strategy value — set the legacy "auto" string so this test
    # exercises SMOTE actually running, not the ImportError/None-handling path.
    config["modeling"]["smote_sampling_strategy"] = "auto"
    config["modeling"]["imbalance_strategy_by_model"] = {"catboost": "cost_weighted"}
    resampler = SMOTEResampler(config)

    X = np.zeros((10, 3))
    y = np.array([0]*6 + [1]*2 + [2]*2)

    # Untouched: gated off for the model resolved to cost_weighted.
    X_out, y_out = resampler.resample(X, y, context="test", model_name="catboost")
    assert X_out is X and y_out is y

    # A model with no per-model override falls back to the global default
    # ('smote' unless explicitly set) and DOES get resampled/considered.
    config["modeling"]["imbalance_strategy"] = "smote"
    resampler_default = SMOTEResampler(config)
    X_out2, y_out2 = resampler_default.resample(X, y, context="test", model_name="random_forest")
    assert not (X_out2 is X and y_out2 is y), "expected SMOTE to actually run for the non-overridden model"


def test_explicit_imbalance_strategy_smote_enables_without_use_smote(minimal_modeling_config):
    """Reproduces the real dissertation.yaml bug: imbalance_strategy: "smote"
    set explicitly, use_smote left at its config default (False) and never
    touched. Before this fix, SMOTEResampler.enabled was driven purely by
    use_smote/search_resample_target_size/smote_sampling_strategy, so this
    combination silently resampled nothing — the dissertation run's "SMOTE"
    condition was actually running with zero imbalance handling, identical
    to imbalance_strategy="none". An explicit imbalance_strategy: "smote"
    must be sufficient on its own now, exactly like "cost_weighted" needs
    no second flag."""
    from wildfire_susceptibility.modeling.resampling import SMOTEResampler
    config = minimal_modeling_config
    config["modeling"]["use_smote"] = False  # exactly dissertation.yaml's real inherited value
    config["modeling"]["smote_sampling_strategy"] = "auto"  # see note in other tests re: None being rejected by imblearn
    config["modeling"]["imbalance_strategy"] = "smote"
    resampler = SMOTEResampler(config)

    X = np.zeros((300, 3))
    y = np.array([0] * 250 + [1] * 30 + [2] * 15 + [3] * 5)

    X_out, y_out = resampler.resample(X, y, context="test", model_name="random_forest")
    assert len(y_out) != len(y), "expected explicit imbalance_strategy='smote' to actually resample"

    # Same config but truly unset (nobody wrote imbalance_strategy at all)
    # must still be governed by use_smote — legacy back-compat preserved.
    config_unset = minimal_modeling_config
    config_unset["modeling"]["use_smote"] = False
    config_unset["modeling"]["imbalance_strategy"] = None
    resampler_unset = SMOTEResampler(config_unset)
    X_out2, y_out2 = resampler_unset.resample(X, y, context="test", model_name="random_forest")
    assert X_out2 is X and y_out2 is y, "unset imbalance_strategy must still defer to use_smote=False"


def test_force_disable_overrides_explicit_imbalance_strategy_smote(minimal_modeling_config):
    """trainer.py uses force_disable to keep search-time resampling off
    unless smote_during_search/stratified_spatial_block opts in — that
    suppression must hold even when imbalance_strategy explicitly requests
    "smote", since it's a decision about *when* to resample, not *whether*
    SMOTE is the configured mechanism."""
    from wildfire_susceptibility.modeling.resampling import SMOTEResampler
    config = minimal_modeling_config
    config["modeling"]["use_smote"] = True
    config["modeling"]["smote_sampling_strategy"] = "auto"
    config["modeling"]["imbalance_strategy"] = "smote"
    resampler = SMOTEResampler(config, force_disable=True)

    X = np.zeros((10, 3))
    y = np.array([0] * 6 + [1] * 2 + [2] * 2)
    X_out, y_out = resampler.resample(X, y, context="test", model_name="random_forest")
    assert X_out is X and y_out is y


def test_model_name_none_preserves_legacy_behavior(minimal_modeling_config):
    """Ad-hoc/legacy callers that don't pass model_name (e.g. this file's
    other tests, or any code calling SMOTEResampler directly) must be
    unaffected by imbalance_strategy — only the model_name-aware call
    sites (cv/base.py, trainer.py) opt into the new gating."""
    from wildfire_susceptibility.modeling.resampling import SMOTEResampler
    config = minimal_modeling_config
    config["modeling"]["imbalance_strategy"] = "cost_weighted"  # would gate off SMOTE if model_name were passed
    config["modeling"]["use_smote"] = True
    config["modeling"]["smote_sampling_strategy"] = "auto"  # see note in the test above
    resampler = SMOTEResampler(config)

    X = np.zeros((10, 3))
    y = np.array([0]*6 + [1]*2 + [2]*2)
    X_out, y_out = resampler.resample(X, y, context="test")  # no model_name
    assert not (X_out is X and y_out is y), "expected SMOTE to run when model_name is not supplied"