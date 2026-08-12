"""Checkpoint utilities to avoid recomputing pipeline steps"""
from pathlib import Path
from typing import Any, Union, Dict, Iterable
import json
import logging

from ..config.signature import compute_cfg_sig

logger = logging.getLogger(__name__)

def output_exists(
    paths: Union[Dict[str, Union[str, Path]], Iterable[Union[str, Path]]]
) -> bool:
    """
    Check whether all given output files already exist on disk.

    Args:
        paths: dict of name -> path, or an iterable of paths

    Returns:
        True only if every path exists and the collection is non-empty.
    """
    path_list = list(paths.values()) if isinstance(paths, dict) else list(paths)
    if not path_list:
        return False
    return all(Path(p).exists() for p in path_list)

def log_cached(
    step_name: str,
    paths: Dict[str, Union[str, Path]]
) -> None:
    logger.info(f"[CACHED] {step_name}: outputs already exist, skipping computation")
    for name, path in paths.items():
        logger.debug(f"  [cached] {name} -> {path}")   # was INFO


def _path_list(paths: Union[Dict[str, Any], Iterable[Any]]) -> list:
    return list(paths.values()) if isinstance(paths, dict) else list(paths)


def compute_cache_signature(
    config: dict,
    input_paths: Union[Dict[str, Union[str, Path]], Iterable[Union[str, Path]]],
) -> dict:
    """
    Signature capturing everything a stage's cached output actually depends
    on: the full run config (via the same compute_cfg_sig used for model
    artifact scoping) plus each upstream input file's mtime.

    Meant to be written alongside a stage's output right after computing it,
    then compared against the *current* config/inputs on the next run via
    cache_is_valid() -- an existence-only check (output file is present)
    can't tell a fresh cache apart from one left over from a run with
    different config or, since the underlying data changed upstream and
    was regenerated, different content at the same path.
    """
    paths = [Path(p) for p in _path_list(input_paths)]
    return {
        "cfg_sig": compute_cfg_sig(config),
        "input_mtimes": {
            p.as_posix(): p.stat().st_mtime for p in paths if p.exists()
        },
    }


def write_cache_signature(sig_path: Union[str, Path], signature: dict) -> None:
    Path(sig_path).write_text(json.dumps(signature, indent=2))


def cache_is_valid(
    output_paths: Union[Dict[str, Any], Iterable[Any]],
    sig_path: Union[str, Path],
    config: dict,
    input_paths: Union[Dict[str, Any], Iterable[Any]],
) -> bool:
    """
    True only if every output exists AND the signature recorded when it was
    written (config content + input file mtimes, see compute_cache_signature)
    matches the current one.

    A cache with no sig file (e.g. written before this check existed, or by
    a stage that hasn't been updated to call write_cache_signature) is
    treated as invalid rather than trusted -- recomputing once is cheap
    compared to silently training on stale data.
    """
    if not output_exists(output_paths):
        return False

    sig_path = Path(sig_path)
    if not sig_path.exists():
        return False

    try:
        recorded = json.loads(sig_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False

    return recorded == compute_cache_signature(config, input_paths)

