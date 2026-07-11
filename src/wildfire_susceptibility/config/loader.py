import yaml
from pathlib import Path
from typing import Union
from wildfire_susceptibility.config.schema import WildfireConfig

class ConfigLoader:
    @staticmethod
    def load(path: Union[str, Path]) -> WildfireConfig:
        """Loads a YAML configuration file and parses it into a strongly typed WildfireConfig object."""
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {file_path.resolve()}")
            
        with open(file_path, "r", encoding="utf-8") as f:
            yaml_content = yaml.safe_load(f) or {}
            
        # Pydantic v2 model validation
        return WildfireConfig.model_validate(yaml_content)