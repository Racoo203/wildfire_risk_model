# config/signature.py
"""Deterministic training-relevant config hashing, used for model
artifact path scoping: models/<season>/<model_name>/<cfg_sig>/...

Distinct from HyperparamSearch._param_space_signature (search.py),
which hashes only a model's Optuna param_space shape — cfg_sig hashes
the full training-relevant config so runs with different labels/
seasons/features/search settings land in different artifact dirs.
"""

import hashlib
import json
from typing import Any, Dict
from pathlib import Path

# Top-level sections and keys excluded from the hash — infra/bookkeeping
# fields that don't affect what gets trained or how.
_EXCLUDED_SECTIONS = {"base", "logging"}
_EXCLUDED_KEYS = {
    ("modeling", "mlflow_experiment"),
}
_EXCLUDED_NESTED_PATTERNS = {
    # any key literally named "data_dir" anywhere under data_sources
    ("data_sources", "*", "data_dir"),
}


def _prune(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of config with excluded sections/keys removed."""
    pruned = {
        section: value
        for section, value in config.items()
        if section not in _EXCLUDED_SECTIONS
    }

    for section, key in _EXCLUDED_KEYS:
        if section in pruned and key in pruned[section]:
            pruned[section] = {k: v for k, v in pruned[section].items() if k != key}

    if "data_sources" in pruned:
        pruned["data_sources"] = {
            source_name: {k: v for k, v in source_cfg.items() if k != "data_dir"}
            for source_name, source_cfg in pruned["data_sources"].items()
        }

    return pruned

def _json_default(value: Any) -> str:
    """Normalize non-JSON-native values for canonical hashing.
    Path objects are rendered as POSIX strings so cfg_sig is identical
    across Windows/Linux/Docker regardless of which OS computed it."""
    if isinstance(value, Path):
        return value.as_posix()
    return str(value)


def compute_cfg_sig(config: Dict[str, Any]) -> str:
    pruned = _prune(config)
    canonical = json.dumps(pruned, sort_keys=True, default=_json_default)
    return hashlib.sha1(canonical.encode()).hexdigest()[:8]