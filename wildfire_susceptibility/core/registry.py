# wildfire_susceptibility/core/registry.py
from typing import Callable, TypeVar

T = TypeVar("T")

class Registry(dict):
    def register(self, name: str) -> Callable[[T], T]:
        def _wrap(obj: T) -> T:
            if name in self:
                raise ValueError(f"'{name}' already registered")
            self[name] = obj
            return obj
        return _wrap

FEATURE_BUILDERS = Registry()   # name -> VarBuilder subclass
MODELS = Registry()             # name -> BaseWildfireModel subclass
CLASSIFY_METHODS = Registry()   # name -> callable(density, valid, mask) -> (labels, breakpoints)
DENSITY_METHODS = Registry()    # name -> callable(fire_gdf, ref_path) -> density array