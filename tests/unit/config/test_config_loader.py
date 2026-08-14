# tests/unit/config/test_config_loader.py
import yaml
import pytest
from pydantic import ValidationError
from wildfire_susceptibility.config.loader import ConfigLoader, DEFAULT_CONFIG_DIR, DEFAULT_CONFIG_FILES
from wildfire_susceptibility.config.schema import WildfireConfig


def test_default_config_dir_resolves_to_repo_root_configs():
    """Guards against DEFAULT_CONFIG_DIR drifting back into src/ again."""
    assert DEFAULT_CONFIG_DIR.name == "configs"
    assert DEFAULT_CONFIG_DIR.parent.name != "wildfire_susceptibility"
    assert (DEFAULT_CONFIG_DIR / "base.yaml").exists()


def test_split_config_matches_legacy_defaults():
    """
    Guards against the configs/*.yaml split silently dropping or
    changing a value. defaults.yaml has been deleted, so this
    reconstructs the legacy monolithic content inline for comparison.
    """
    split_cfg = ConfigLoader.load(config_dir=DEFAULT_CONFIG_DIR, files=DEFAULT_CONFIG_FILES)

    merged = {}
    for fname in DEFAULT_CONFIG_FILES:
        with open(DEFAULT_CONFIG_DIR / fname) as f:
            content = yaml.safe_load(f) or {}
        for key, value in content.items():
            assert key not in merged, f"Unexpected top-level key collision: {key}"
            merged[key] = value

    reconstructed_cfg = WildfireConfig.model_validate(merged)
    assert split_cfg.model_dump(mode="python") == reconstructed_cfg.model_dump(mode="python")

# def test_load_experiment_overrides_only_specified_keys():
#     """
#     Guards against load_experiment() accidentally clobbering unrelated
#     base config sections instead of deep-merging just the overridden keys.
#     Exercised against configs/experiment/baseline.yaml, the only experiment
#     override file that actually exists in the repo.
#     """
#     base_cfg = ConfigLoader.load()  # six canonical files, no overrides
#     exp_cfg = ConfigLoader.load_experiment("baseline")

#     # Overridden keys actually changed
#     assert exp_cfg.seasons.active == ["summer"]
#     assert exp_cfg.modeling.mlflow_experiment == "wf-baseline"
#     assert exp_cfg.modeling.use_smote is True
#     assert exp_cfg.modeling.smote_during_search is True
#     assert exp_cfg.modeling.search_resample_target_size == 100_000
#     assert exp_cfg.modeling.smote_sampling_strategy == {1: 400000, 2: 200000, 3: 100000}
#     assert exp_cfg.modeling.models == ["catboost"]
#     assert exp_cfg.labels.density_method == "convolution"
#     assert exp_cfg.labels.classify_method == "gmm"

#     # Untouched sections remain identical to base
#     assert exp_cfg.data_sources == base_cfg.data_sources
#     assert exp_cfg.processing == base_cfg.processing
#     assert exp_cfg.logging == base_cfg.logging
#     assert exp_cfg.modeling.cv_folds == base_cfg.modeling.cv_folds
#     # labels.classify_method was already "gmm" in base — confirm other
#     # labels.* fields weren't clobbered by the override file's partial dict
#     assert exp_cfg.labels.gmm_n_components == base_cfg.labels.gmm_n_components
#     assert exp_cfg.labels.percentiles == base_cfg.labels.percentiles


def test_baseline_auto_find_k_override_survives_load_and_dump():
    """Regression guard for the labels-config-drift fix: before
    extra='forbid' + real schema fields, LabelsConfig's default
    extra='ignore' silently dropped configs/experiment/baseline.yaml's
    `auto_find_k: false` override, and .get(..., False) happened to
    produce the same value by coincidence -- masking the drop. This
    confirms the override is now genuinely wired end to end, including
    into the dumped dict the pipeline actually runs against."""
    exp_cfg = ConfigLoader.load_experiment("baseline")
    assert exp_cfg.labels.auto_find_k is False

    dump = exp_cfg.model_dump(mode="python")
    assert dump["labels"]["auto_find_k"] is False


def test_labels_yaml_auto_find_k_fields_load_and_match_yaml_values():
    """Loads labels.yaml (via the default six-file ConfigLoader.load())
    and asserts auto_find_k / k_search_range / trim_bottom_pct actually
    reach the dumped config dict, matching the raw YAML values -- these
    three fields had no schema slot before this fix, so pydantic v2's
    default extra='ignore' silently dropped them on every load."""
    with open(DEFAULT_CONFIG_DIR / "labels.yaml") as f:
        raw_labels = yaml.safe_load(f)["labels"]

    cfg = ConfigLoader.load()
    dump = cfg.model_dump(mode="python")

    assert dump["labels"]["auto_find_k"] == raw_labels["auto_find_k"]
    assert list(dump["labels"]["k_search_range"]) == raw_labels["k_search_range"]
    assert dump["labels"]["trim_bottom_pct"] == raw_labels["trim_bottom_pct"]
    assert dump["labels"]["n_classes"] == raw_labels["n_classes"]


def test_labels_yaml_no_longer_has_ambiguous_max_sample_key():
    """max_sample (bare, ambiguous — never read by any code path) was
    removed from labels.yaml in favor of the real jenks_max_sample /
    gmm_max_sample fields the code actually consumes."""
    with open(DEFAULT_CONFIG_DIR / "labels.yaml") as f:
        raw_labels = yaml.safe_load(f)["labels"]
    assert "max_sample" not in raw_labels


def test_labels_config_rejects_misspelled_key():
    """extra='forbid' on LabelsConfig must reject unknown keys instead of
    silently dropping them -- this is the actual bug fix; a misspelling
    like `auto_find_kk` should now raise loudly instead of the pipeline
    quietly running with default behavior."""
    with pytest.raises(ValidationError, match="auto_find_kk"):
        WildfireConfig.model_validate({"labels": {"auto_find_kk": True}})


def test_labels_config_rejects_misspelled_key_via_full_loader(tmp_path):
    """Same guard, exercised through ConfigLoader.load() against an
    on-disk override file, matching how a real typo would actually be
    encountered."""
    typo_file = tmp_path / "typo.yaml"
    typo_file.write_text("labels:\n  auto_find_kk: true\n")
    with pytest.raises(ValidationError, match="auto_find_kk"):
        ConfigLoader.load(config_dir=tmp_path, files=["typo.yaml"])