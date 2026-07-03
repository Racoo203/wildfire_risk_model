"""Pipeline orchestrator — thin coordination layer.

Static (non-seasonal) feature builders are resolved via FEATURE_BUILDERS
so that adding a new static feature means registering a new builder class,
not editing this file. Climate and seasonal-NDVI are still called
explicitly because they require months/season arguments the registry loop
can't infer — this becomes fully declarative in Phase 3's pipeline/stages.py.

NOTE: BoundaryBuilder is run for its side effect only (writing boundary.shp,
which VarBuilder._boundary() reads for every other builder's clip step) and
is deliberately excluded from the raster-merging loop below — it returns a
vector path, not a feature raster, and must never end up in the feature
stack that gets passed to rasterio.open() downstream.
"""

from pathlib import Path
from typing import Dict, Tuple
import logging

from ..core.registry import FEATURE_BUILDERS
from ..utils.logger import setup_logger

# Static builders whose process() output IS a raster feature and belongs
# in the merged static_features dict. BoundaryBuilder is intentionally
# NOT in this list — see module docstring.
_STATIC_RASTER_BUILDER_ORDER = ("proximity",)


class FeatureOrchestrator:
    """
    Resolves and runs the static (non-seasonal) portion of the feature
    pipeline via the FEATURE_BUILDERS registry.

    Seasonal builders (climate, seasonal NDVI, fire/label pipeline) remain
    the responsibility of WildfirePreprocessor, which needs per-season
    control flow that a flat registry loop doesn't express well.
    """

    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger(
            log_file=config["logging"]["log_path"],
            level=config["logging"]["level"],
        )

    def build_static_features(self) -> Dict[str, Path]:
        self.logger.info("Building static features via FEATURE_BUILDERS registry...")

        # Boundary must run first (side effect only — writes boundary.shp
        # that _boundary()/_clip_to_boundary() depend on). Its output is
        # NOT a feature raster and must not be merged below.
        boundary_builder = FEATURE_BUILDERS["boundary"](self.config)
        boundary_builder.process()

        # Topography runs next: no ref_path dependency, produces the
        # reference raster every other builder needs.
        topo_builder = FEATURE_BUILDERS["topography"](self.config)
        topo_features = topo_builder.process()
        ref_path = topo_features["elevation"]

        static_features = dict(topo_features)

        for name in _STATIC_RASTER_BUILDER_ORDER:
            builder_cls = FEATURE_BUILDERS.get(name)
            if builder_cls is None:
                self.logger.warning(f"No FEATURE_BUILDERS entry for '{name}', skipping.")
                continue

            builder = builder_cls(self.config, ref_path)
            outputs = builder.process()
            if outputs:
                static_features.update(outputs)

        return static_features, ref_path