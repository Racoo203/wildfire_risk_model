# tests/unit/config/test_signature.py
from wildfire_susceptibility.config.signature import compute_cfg_sig


def test_cfg_sig_stable_for_identical_training_relevant_config(minimal_modeling_config):
    sig_a = compute_cfg_sig(minimal_modeling_config)
    sig_b = compute_cfg_sig(minimal_modeling_config)
    assert sig_a == sig_b


def test_cfg_sig_changes_when_labels_change(minimal_modeling_config):
    sig_before = compute_cfg_sig(minimal_modeling_config)
    minimal_modeling_config["labels"]["classify_method"] = "jenks"
    sig_after = compute_cfg_sig(minimal_modeling_config)
    assert sig_before != sig_after


def test_cfg_sig_unchanged_when_only_excluded_fields_change(minimal_modeling_config):
    sig_before = compute_cfg_sig(minimal_modeling_config)
    minimal_modeling_config["base"]["figures_dir"] = "/some/other/path"
    minimal_modeling_config["modeling"]["mlflow_experiment"] = "totally-different-name"
    minimal_modeling_config["logging"]["level"] = "DEBUG"
    sig_after = compute_cfg_sig(minimal_modeling_config)
    assert sig_before == sig_after


def test_cfg_sig_unaffected_by_dict_key_order(minimal_modeling_config):
    import json
    reordered = json.loads(json.dumps(minimal_modeling_config))  # trivial reorder isn't guaranteed by this, but...
    # More reliable: manually construct a reordered dict at the top level
    reordered = {k: minimal_modeling_config[k] for k in reversed(list(minimal_modeling_config.keys()))}
    assert compute_cfg_sig(minimal_modeling_config) == compute_cfg_sig(reordered)