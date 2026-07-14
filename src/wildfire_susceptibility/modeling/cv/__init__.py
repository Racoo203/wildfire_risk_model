"""CV strategy package. `cv_strategy` config values map 1:1 to a
CVStrategy subclass via get_cv_strategy() — no string-switch logic lives
outside factory.py."""

from .base import CVStrategy
from .standard import StandardKFoldCV
from .spatial import SpatialGroupKFoldCV
from .stratified_spatial import StratifiedSpatialBlockCV
from .factory import get_cv_strategy, requires_spatial_groups

FoldStrategy = StandardKFoldCV

__all__ = [
    "CVStrategy", "StandardKFoldCV", "SpatialGroupKFoldCV", "StratifiedSpatialBlockCV",
    "get_cv_strategy", "requires_spatial_groups", "FoldStrategy",
]