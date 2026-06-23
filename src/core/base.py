"""Abstract base class for all variable builders."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Optional, Dict
import logging

from ..utils.checkpoint import output_exists, log_cached

logger = logging.getLogger(__name__)

class VarBuilder(ABC):
    """
    Abstract base for all variable types.

    Supports checkpointing: if force_recompute if False (default), 
    and all expected output files already exist, process() returns the cached
    paths immediately instead of recomputing.

    ref_path and boundary_path are optional because TopographyBuilder 
    produces the reference raster itself rather than consuming one.

    Subclasses implement:
    - process(): transform to aligned 30m GeoTIFF
    """

    def __init__(
        self,
        config,
        ref_path: Optional[Union[str, Path]] = None,
    ):  
        self.config = config
        self.output_dir = Path(config["base"]["output_dir"])
        self.output_dir.mkdir(parents = True, exist_ok = True)

        self.ref_path = Path(ref_path) if ref_path else None
        self.boundary_path = Path(config["base"]["boundary_shapefile"])

        self.force_recompute = config["processing"]["force_recompute"]

    @abstractmethod
    def process(self, **kwargs) -> Path:
        """
        Transform raw data into aligned 30m GeoTIFF

        Returns:
            Path to output GeoTIFF
        """

        pass
    
    def _check_cache(
        self,
        step_name: str,
        output_paths: Dict[str, Path]
    ) -> bool:
        if self.force_recompute:
            return False
        if output_exists(output_paths):
            log_cached(step_name, output_paths)
            return True
        return False
        
    def _validate_reference(self) -> None:
        """Check that reference raster exists and is valid."""
        if self.ref_path is None:
            return
        if not self.ref_path.exists():
            raise FileNotFoundError(f"Reference raster not found: {self.ref_path}")

    @staticmethod
    def _seasonal_name(
        base_name: str,
        season: Optional[str]
    ) -> str:
        """
        e.g. _seasonal_name('avg_temp_essex_30m', 'summer') -> 
        """

        return f"{base_name}_{season}" if season else base_name