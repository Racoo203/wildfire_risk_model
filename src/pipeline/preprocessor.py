import yaml
from pathlib import Path
from typing import Union, Dict, Tuple
import logging
import pandas as pd
import numpy as np
import rasterio

from ..features.boundary import BoundaryBuilder
from ..features.topography import TopographyBuilder
from ..features.climate import ClimateBuilder
from ..features.vegetation import VegetationBuilder
from ..features.proximity import ProximityBuilder

from ..labels.fire_incidents import FireBuilder
from ..labels.kernel_density import KernelDensityClassifier
from ..labels.classification import LabelCleaner

from ..core.raster import RasterManager
from ..utils.logger import setup_logger

logger = logging.getLogger(__name__)

class WildfirePreprocessor:
    """
    Orchestrates the complete preprocessing pipeline:
    1. Build all features
    2. Load fire data
    3. Compute KDE and classify labels
    4. Clean labels with k-means
    5. Stack into tabular datasets
    """

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        self.output_dir = Path(self.config["base"]["output_dir"])

        self.force_recompute = self.config["processing"]["force_recompute"]

        self.logger = setup_logger(
            log_file = self.config["logging"]["log_path"],
            level = self.config["logging"]["level"]
        )

    def _load_config(self) -> dict:
        """Load configuration from YAML."""
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def run_full_pipeline(self) -> Dict[str, Path]:
        """
        1. Split fire records by sets (training, validation).
        2. Compute climate averages from training years only.
        3. Compute fire density labels from training fires only.
        """

        logger.info("Starting wildfire preprocessing pipeline (seasonal)...")

        static_features = self._build_static_features()
        ref_path = static_features["elevation"]
        # print(ref_path)

        season_defs = self.config["seasons"]["definitions"]
        active_seasons = self.config["seasons"]["active"]
        dataset_paths = {}

        for season in active_seasons:
            months = tuple(season_defs[season])
            seasonal_features = self._build_seasonal_features(
                season, months, ref_path
            )

            all_features = {**static_features, **seasonal_features}

            labels_clean, fire_val, fire_test = self._build_seasonal_labels(
                months, season, all_features, ref_path
            )

            dataset_paths[season] = self._assemble_seasonal_dataset(
                season, all_features, labels_clean, ref_path
            )

        logger.info("Pipeline complete for all active seasons.")
        return dataset_paths

    def _build_static_features(self) -> Dict[str, Path]:
        logger.info("Building static features...")

        bound_builder = BoundaryBuilder(self.config)
        bound_builder.process()
        
        topo_builder = TopographyBuilder(self.config)
        topo_features = topo_builder.process()

        ref_path = topo_features["elevation"]

        prox_builder = ProximityBuilder(self.config, ref_path)
        prox_features = prox_builder.process()

        static_features = {**topo_features, **prox_features}

        if not self.config["seasons"].get("seasonal_ndvi", False):
            veg_builder = VegetationBuilder(self.config, ref_path)
            veg_features = veg_builder.process()

            static_features.update(veg_features)

        return static_features

    def _build_seasonal_features(
        self,
        season: str,
        months: Tuple[int, ...],
        ref_path: Path
    ) -> Dict[str, Path]:
        climate_builder = ClimateBuilder(self.config, ref_path)
        climate_features = climate_builder.process(months = months, season = season)

        seasonal_features = {**climate_features}

        if self.config["seasons"].get("seasonal_ndvi", False):
            veg_builder = VegetationBuilder(self.config, ref_path)
            veg_features = veg_builder.process(months = months, season = season)

            seasonal_features.update(veg_features)
        
        return seasonal_features

    def _build_seasonal_labels(
        self,
        months,
        season: str, 
        all_features: Dict[str, Path],
        ref_path,
    ) -> np.ndarray:
        # 1. Split first
        fire_builder = FireBuilder(self.config)
        fire_train, fire_val, fire_test = fire_builder.process(
            months = months,
            season = season,
        )

        # 2. Density and labels from training fires only
        kde = KernelDensityClassifier(self.config, ref_path)
        # print(fire_train.type)
        density = kde.compute_kde(
            fire_train,
            season = season,
        )

        labels = kde.classify(
            density,
            season = season,
        )

        # 3. K-means cleaning (only training)
        # feature_arrays = self._load_feature_arrays(all_features)
        # cleaner = LabelCleaner(self.config, ref_path)
        # labels_clean = cleaner.clean(
        #     labels,
        #     feature_arrays,
        #     season = season
        # )

        # return labels_clean, fire_val, fire_test
        return labels, fire_val, fire_test


    @staticmethod
    def _load_feature_arrays(
        feature_paths: Dict[str, Path]
    ) -> Dict[str, np.ndarray]:
        arrays = {}
        for name, path in feature_paths.items():
            with rasterio.open(path) as src:
                arrays[name] = src.read(1)

        return arrays

    def _assemble_seasonal_dataset(
        self,
        season: str,
        features: Dict[str, Path],
        labels: np.ndarray,
        ref_path: Path
    ) -> Path:
        model_data_path = Path(self.config["base"]["model_data_dir"])
        model_data_path.mkdir(parents = True, exist_ok = True)
        out_csv = model_data_path / f"dataset_clean_{season}.csv"

        if not self.force_recompute and out_csv.exists():
            logger.info(f"[CACHED] Dataset for {season} already exists: {out_csv}")
            return out_csv
        
        label_path = self.output_dir / f"_labels_temp_{season}.tif"
        with rasterio.open(ref_path) as ref:
            meta = ref.meta.copy()

        meta.update({"dtype": "float32", "nodata": np.nan, "count": 1})
        with rasterio.open(label_path, "w", **meta) as dst:
            dst.write(labels[np.newaxis, :, :])

        df = RasterManager.stack_to_dataframe(
            {**features, "label": label_path}, ref_path, out_csv
        )
        label_path.unlink()
        return out_csv
