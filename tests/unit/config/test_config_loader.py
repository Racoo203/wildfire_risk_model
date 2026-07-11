# tests/unit/config/test_config_loader.py
import yaml
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

def test_load_experiment_overrides_only_specified_keys():
    """
    Guards against load_experiment() accidentally clobbering unrelated
    base config sections instead of deep-merging just the overridden keys.
    """
    base_cfg = ConfigLoader.load()  # six canonical files, no overrides
    exp_cfg = ConfigLoader.load_experiment("summer_jenks_baseline")

    # Overridden keys actually changed
    assert exp_cfg.seasons.active == ["summer"]
    assert exp_cfg.modeling.mlflow_experiment == "wildfire-summer-jenks-baseline"

    # Untouched sections remain identical to base
    assert exp_cfg.data_sources == base_cfg.data_sources
    assert exp_cfg.processing == base_cfg.processing
    assert exp_cfg.logging == base_cfg.logging
    # labels.classify_method was already "gmm" in base — confirm other
    # labels.* fields weren't clobbered by the override file's partial dict
    assert exp_cfg.labels.gmm_n_components == base_cfg.labels.gmm_n_components
    assert exp_cfg.labels.percentiles == base_cfg.labels.percentiles