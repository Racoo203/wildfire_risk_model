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
    reordered = {k: minimal_modeling_config[k] for k in reversed(list(minimal_modeling_config.keys()))}
    assert compute_cfg_sig(minimal_modeling_config) == compute_cfg_sig(reordered)

def test_cfg_sig_stable_across_path_representations(minimal_modeling_config):
    """Guards against WindowsPath vs PosixPath producing different hashes
    for what is logically the same path."""
    from pathlib import PurePosixPath, PureWindowsPath

    sig_native = compute_cfg_sig(minimal_modeling_config)

    swapped = dict(minimal_modeling_config)
    swapped["data_sources"] = dict(swapped["data_sources"])
    swapped["data_sources"]["srtm"] = dict(swapped["data_sources"]["srtm"])
    # Simulate the same logical path rendered as the *other* OS's Path type
    original = swapped["data_sources"]["srtm"]["data_dir"]
    if isinstance(original, PureWindowsPath):
        swapped["data_sources"]["srtm"]["data_dir"] = PurePosixPath(original.as_posix())
    else:
        swapped["data_sources"]["srtm"]["data_dir"] = PureWindowsPath(str(original))

    sig_swapped = compute_cfg_sig(swapped)
    assert sig_native == sig_swapped

def test_cfg_sig_normalizes_path_objects_consistently():
    """Guards against Path serialization being OS-dependent, in case a
    future training-relevant field becomes Path-typed."""
    from pathlib import PurePosixPath, PureWindowsPath

    cfg_a = {"labels": {"some_future_path_field": PurePosixPath("data/bronze/x")}}
    cfg_b = {"labels": {"some_future_path_field": PureWindowsPath("data\\bronze\\x")}}
    assert compute_cfg_sig(cfg_a) == compute_cfg_sig(cfg_b)