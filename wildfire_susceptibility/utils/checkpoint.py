"""Checkpoint utilities to avoid recomputing pipeline steps"""
from pathlib import Path
from typing import Union, Dict, Iterable
import logging

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
    """
    Log that a step's outputs were found cached and computation was skipped
    """
    logger.info(f"[CACHED] {step_name}: outputs already exist, skipping computation")
    for name, path in paths.items():
        logger.info(f"  [cached] {name} -> {path}")

