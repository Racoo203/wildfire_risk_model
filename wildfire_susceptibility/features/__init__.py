"""Feature (variable) builders: topography, climate, vegetation, and
proximity layers used as model inputs. Each builder subclasses VarBuilder
and registers itself into FEATURE_BUILDERS."""

from ..core.base import VarBuilder
from .boundary import BoundaryBuilder
from .topography import TopographyBuilder
from .climate import ClimateBuilder
from .vegetation import VegetationBuilder
from .proximity import ProximityBuilder, FireProximityBuilder

__all__ = [
    "VarBuilder",
    "BoundaryBuilder",
    "TopographyBuilder",
    "ClimateBuilder",
    "VegetationBuilder",
    "ProximityBuilder",
    "FireProximityBuilder",
]