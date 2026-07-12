import yaml
from pathlib import Path
from typing import Union, List
from wildfire_susceptibility.config.schema import WildfireConfig

DEFAULT_CONFIG_DIR = Path("./configs")
DEFAULT_CONFIG_FILES = [
    "base.yaml", "processing.yaml", "data_sources.yaml",
    "labels.yaml", "modeling.yaml", "logging.yaml",
]

class ConfigLoader:
    @staticmethod
    def load(
        config_dir: Union[str, Path] = DEFAULT_CONFIG_DIR,
        files: List[str] = DEFAULT_CONFIG_FILES,
    ) -> WildfireConfig:
        """
        Loads and deep-merges a set of topic-scoped YAML files from
        config_dir, in the order given by `files`, then validates the
        merged result into a WildfireConfig. Missing keys fall back to
        pydantic schema defaults, same as before — an empty or absent
        file for a topic is not an error.
        """
        merged: dict = {}
        config_dir = Path(config_dir)

        for fname in files:
            fpath = config_dir / fname
            if not fpath.exists():
                continue  # topic file optional; schema defaults cover it
            with open(fpath, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f) or {}
            merged = ConfigLoader._deep_merge(merged, content)

        return WildfireConfig.model_validate(merged)

    @staticmethod
    def load_experiment(
        name: str,
        config_dir: Union[str, Path] = DEFAULT_CONFIG_DIR,
        base_files: List[str] = DEFAULT_CONFIG_FILES,
    ) -> WildfireConfig:
        """
        Load the six canonical config files, then layer an experiment
        override file on top. `name` is the experiment filename without
        the 'experiment/' prefix or '.yaml' suffix, e.g. "summer_gmm_baseline".
        """
        experiment_file = f"experiment/{name}.yaml"
        return ConfigLoader.load(
            config_dir=config_dir,
            files=[*base_files, experiment_file],
        )

    @staticmethod
    def _deep_merge(base: dict, override: dict) -> dict:
        """Recursively merge override into base; override wins on conflicts."""
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        return result