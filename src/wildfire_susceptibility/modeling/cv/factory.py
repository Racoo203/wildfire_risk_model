"""Single place that resolves a `cv_strategy` config string to a
CVStrategy instance — replaces the old if/elif embedded in FoldStrategy."""

from typing import Dict, Type

from .base import CVStrategy
from .standard import StandardKFoldCV
from .spatial import SpatialGroupKFoldCV
from .stratified_spatial import StratifiedSpatialBlockCV
from ..resampling import SMOTEResampler

_STRATEGIES: Dict[str, Type[CVStrategy]] = {
    "standard": StandardKFoldCV,
    "spatial": SpatialGroupKFoldCV,
    "stratified_spatial_block": StratifiedSpatialBlockCV,
}


def get_cv_strategy(name: str, config: dict, resampler: SMOTEResampler) -> CVStrategy:
    if name not in _STRATEGIES:
        raise ValueError(f"Unknown cv_strategy '{name}' (options: {sorted(_STRATEGIES)})")
    return _STRATEGIES[name](config, resampler)


def requires_spatial_groups(name: str) -> bool:
    return name in ("spatial", "stratified_spatial_block")