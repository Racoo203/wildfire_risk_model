from pathlib import Path
from wildfire_susceptibility.config.signature import compute_cfg_sig

def artifact_dir(
    config: dict, 
    season: str, 
    model_name: str, 
    models_root: Path = Path("models")
) -> Path:
    """models/<season>/<model_name>/<cfg_sig>/"""
    cfg_sig = compute_cfg_sig(config)
    path = models_root / season / model_name / cfg_sig
    path.mkdir(parents=True, exist_ok=True)
    return path