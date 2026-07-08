import yaml
from pathlib import Path
from typing import Union, Tuple, Dict, Optional

import numpy as np
import rasterio

# Static
from ..features.boundary import BoundaryBuilder
from ..features.topography import TopographyBuilder
from ..features.proximity import ProximityBuilder

# Seasonal
from ..features.climate import ClimateBuilder
from ..features.vegetation import VegetationBuilder
from ..features.proximity import FireProximityBuilder

# Labels
from ..labels.fire_incidents import FireBuilder
from ..labels.kernel_density import KernelDensityClassifier

from ..utils.logger import setup_logger
from ..core.raster import RasterManager

class DatasetBuilder:
    """
    Orchestrates the Dataset Integration stage only:
        1. Build static features (topography, proximity, boundary)
        2. Per active season: build train/test climate+NDVI+labels, stack to CSV

    Label cleaning, imputation-for-modeling, scaling, and resampling are
    NOT run here anymore — see modeling/dataset_prep.py, invoked from the
    training pipeline once both dataset_train_<season>.csv and
    dataset_test_<season>.csv exist.
    """

    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.output_dir = Path(self.config["base"]["output_dir"])
        self.model_data_dir = Path(self.config["base"]["model_data_dir"])
        self.model_data_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(
            log_file=self.config["logging"]["log_path"],
            level=self.config["logging"]["level"],
        )

    def _load_config(self) -> dict:
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def run_full_pipeline(self) -> Dict[str, Dict[str, Path]]:
        """Returns {season: {"train": path, "test": path}}."""
        self.logger.info("Starting wildfire dataset-integration pipeline (seasonal)...")

        static_features, ref_path = self._build_static_features()

        season_defs = self.config["seasons"]["definitions"]
        active_seasons = self.config["seasons"]["active"]

        dataset_paths = {}
        for season in active_seasons:
            months = tuple(season_defs[season])
            dataset_paths[season] = self._build_season(season, months, static_features, ref_path)

        self.logger.info("Dataset integration complete for all active seasons.")
        return dataset_paths
    
    def _build_static_features(self) -> Dict[str, Path]:
        self.logger.info("Building static features via FEATURE_BUILDERS registry...")

        boundary_builder = BoundaryBuilder(self.config)
        boundary_builder.process()

        topo_builder = TopographyBuilder(self.config)
        topo_features = topo_builder.process()
        ref_path = topo_features["elevation"]

        prox_builder = ProximityBuilder(self.config, ref_path)
        prox_features = prox_builder.process()

        static_features = {**topo_features, **prox_features}

        if not self.config["seasons"].get("seasonal_ndvi", False):
            veg_builder = VegetationBuilder(self.config, ref_path)
            static_features.update(veg_builder.process(split="static"))

        return static_features, ref_path
    
    def _build_season(
        self,
        season: str,
        months: Tuple[int, ...],
        static_features: Dict[str, Path],
        ref_path: Path,
    ) -> Dict[str, object]:
        """Returns {"train": path, "test": path, "fire_train": gdf, "fire_test": gdf}."""

        fire_builder = FireBuilder(self.config)
        fire_train, fire_test = fire_builder.process(months=months, season=season)

        fire_prox_builder = FireProximityBuilder(self.config, ref_path)
        fire_prox_features = fire_prox_builder.process(fire_train, season=season)

        # dict order (train, test) matters here: train must be classified FIRST
        # so its fit artifact can be frozen and reused for test — see label_fit.
        splits = {
            "train": (self.config["processing"]["training_years"], fire_train),
            "test": (self.config["processing"]["test_years"], fire_test),
        }

        out_paths = {"fire_train": fire_train, "fire_test": fire_test}
        label_fit = None  # populated after the train split classifies; reused, not refit, for test

        for split, (year_range, fire_gdf) in splits.items():
            seasonal_features = self._build_seasonal_features(season, months, year_range, split, ref_path)
            all_features = {**static_features, **seasonal_features}

            if self.config["labels"].get("include_d_fires_as_feature", True):
                all_features.update(fire_prox_features)

            raw_labels, label_fit = self._build_raw_labels(fire_gdf, season, split, ref_path, fitted=label_fit)
            out_paths[split] = self._assemble_dataset(season, split, all_features, raw_labels, ref_path)

        return out_paths
    
    def _build_seasonal_features(self, season, months, year_range, split, ref_path) -> Dict[str, Path]:
        climate_builder = ClimateBuilder(self.config, ref_path)
        climate_features = climate_builder.process(
            months=months, year_range=year_range, split=split, season=season
        )

        seasonal_features = {**climate_features}
        if self.config["seasons"].get("seasonal_ndvi", False):
            veg_builder = VegetationBuilder(self.config, ref_path)
            veg_features = veg_builder.process(
                months=months, year_range=year_range, split=split, season=season
            )
            seasonal_features.update(veg_features)

        return seasonal_features
    
    def _build_raw_labels(
        self, fire_gdf, season: str, split: str, ref_path: Path, fitted: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """
        Density + classification. `fitted` is None for the train split (fits a
        new classifier) and the train split's returned fit artifact for the
        test split (applies the frozen boundaries — never refits on test
        density, which would otherwise silently redefine what "High" means).
        """
        density_method = self.config["labels"].get("density_method", "convolution")
        classify_method = self.config["labels"].get("classify_method", "percentile")

        kde = KernelDensityClassifier(self.config, ref_path)
        density = kde.compute_density(fire_gdf, season=f"{season}_{split}", method=density_method)
        labels, fit_artifact = kde.classify(
            density,
            season=f"{season}_{split}",
            method=density_method,
            classify_method=classify_method,
            fitted=fitted,
        )
        return labels, fit_artifact

    def _assemble_dataset(
        self, season: str, split: str, features: Dict[str, Path], labels: np.ndarray, ref_path: Path
    ) -> Path:
        out_csv = self.model_data_dir / f"dataset_{split}_{season}.csv"

        if not self.config["processing"]["force_recompute"] and out_csv.exists():
            self.logger.info(f"[CACHED] {split} dataset for {season} already exists: {out_csv}")
            return out_csv

        label_path = self.output_dir / f"_labels_temp_{split}_{season}.tif"
        with rasterio.open(ref_path) as ref:
            meta = ref.meta.copy()
        meta.update({"dtype": "float32", "nodata": np.nan, "count": 1})
        with rasterio.open(label_path, "w", **meta) as dst:
            dst.write(labels[np.newaxis, :, :])

        RasterManager.stack_to_dataframe({**features, "label": label_path}, ref_path, out_csv)
        label_path.unlink()

        self.logger.info(f"[{season}][{split}] Dataset assembled → {out_csv.name}")
        return out_csv