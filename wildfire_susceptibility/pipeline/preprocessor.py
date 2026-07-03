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
from ..features.proximity import ProximityBuilder, FireProximityBuilder

from ..labels.fire_incidents import FireBuilder
from ..labels.kernel_density import KernelDensityClassifier
from ..labels.classification import LabelCleaner

from .orchestrator import FeatureOrchestrator
from ..core.registry import FEATURE_BUILDERS

from ..modeling.dataset_prep import DatasetPrep  # add this import

from ..core.raster import RasterManager
from ..utils.logger import setup_logger

class WildfirePreprocessor:
    """
    Orchestrates the complete preprocessing pipeline:
    1. Build all features
    2. Load fire data
    3. Compute density and classify labels
    4. Clean ambiguous Low labels via pairwise k-means (+ sensitivity analysis)
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
        4. Clean ambiguous Low labels via pairwise k-means.
        """

        self.logger.info("Starting wildfire preprocessing pipeline (seasonal)...")

        static_features = self._build_static_features()
        ref_path = static_features["elevation"]

        season_defs = self.config["seasons"]["definitions"]
        active_seasons = self.config["seasons"]["active"]
        dataset_paths = {}

        for season in active_seasons:
            months = tuple(season_defs[season])
            seasonal_features = self._build_seasonal_features(
                season, months, ref_path
            )

            all_features = {**static_features, **seasonal_features}

            train_labels, fires, fire_prox_features = self._build_seasonal_labels(
                months, season, ref_path
            )

            all_features.update(fire_prox_features)

            train_labels = self._clean_seasonal_labels(
                train_labels, all_features, season, ref_path
            )

            dataset_paths[season] = self._assemble_seasonal_dataset(
                season, all_features, train_labels, ref_path
            )

        self.logger.info("Pipeline complete for all active seasons.")
        return dataset_paths

    def _build_static_features(self) -> Dict[str, Path]:
        orchestrator = FeatureOrchestrator(self.config)
        static_features, ref_path = orchestrator.build_static_features()
        self._ref_path = ref_path  # stash for _build_seasonal_features / labels

        if not self.config["seasons"].get("seasonal_ndvi", False):
            veg_builder = FEATURE_BUILDERS["vegetation"](self.config, ref_path)
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

    def _build_seasonal_labels(self, months, season, ref_path):
        # 1. Split first
        fire_builder = FireBuilder(self.config)
        fires = fire_builder.process(months=months, season=season)
        fire_train = fires[0]

        # 2. d_fires — training fires only, built strictly after the split
        fire_prox_builder = FireProximityBuilder(self.config, ref_path)
        fire_prox_features = fire_prox_builder.process(fire_train, season=season)

        # 3. Density surface (shared across all classify_methods below)
        density_method = self.config["labels"].get("density_method", "convolution")
        classify_method = self.config["labels"].get("classify_method", "percentile")

        kde = KernelDensityClassifier(self.config, ref_path)
        density = kde.compute_density(fire_train, season=season, method=density_method)

        # 4. Optionally render every classify_method for side-by-side comparison
        #    (figures + label rasters), without changing which one feeds the dataset.
        for compare_method in self.config["labels"].get("compare_classify_methods", []):
            if compare_method != classify_method:
                kde.classify(density, season=season, method=density_method, classify_method=compare_method)

        # 5. The method that actually feeds the model dataset
        train_labels = kde.classify(
            density, season=season, method=density_method, classify_method=classify_method
        )

        return train_labels, fires, fire_prox_features

    def _clean_seasonal_labels(
        self,
        train_labels: np.ndarray,
        all_features: Dict[str, Path],
        season: str,
        ref_path: Path,
    ) -> np.ndarray:
        if not self.config["labels"].get("clean_labels", True):
            return train_labels

        feature_arrays = self._load_feature_arrays(all_features)

        # Resolve missing data BEFORE k-means cleaning (Section 7.3 fix).
        prep = DatasetPrep(self.config)
        feature_df = pd.DataFrame({k: v.ravel() for k, v in feature_arrays.items()})
        feature_df = prep.resolve_missing(
            feature_df,
            slope_aspect_cols=("slope", "aspect"),
            ndvi_col="ndvi",
            drop_if_any_nan_in=(),
        )
        ref_shape = next(iter(feature_arrays.values())).shape
        feature_arrays = {k: feature_df[k].values.reshape(ref_shape) for k in feature_arrays}

        # NEW: any pixel where a feature still has an unresolved NaN (climate
        # resampling edges, sea/estuary pixels outside the land mask, etc.)
        # cannot be fed to KMeans. Rather than drop rows from a dataframe
        # (which would break the raster shape), exclude those pixels from
        # labelled training data by setting their label to NaN — this has the
        # same practical effect as "drop" (LabelCleaner._flatten_valid already
        # filters on ~np.isnan(labels)) without touching the raster grid.
        still_nan_mask = np.zeros(ref_shape, dtype=bool)
        for name, arr in feature_arrays.items():
            still_nan_mask |= np.isnan(arr)

        n_excluded = int(np.sum(still_nan_mask & ~np.isnan(train_labels)))
        if n_excluded:
            self.logger.info(
                f"[{season}] Excluding {n_excluded:,} otherwise-labelled pixels from "
                f"cleaning/training due to unresolved feature NaNs (e.g. climate "
                f"resampling edges) — see per-feature breakdown above."
            )
        train_labels = train_labels.copy()
        train_labels[still_nan_mask] = np.nan

        cleaner = LabelCleaner(self.config, ref_path)
        return cleaner.clean(train_labels, feature_arrays, season=season)

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
            self.logger.info(f"[CACHED] Dataset for {season} already exists: {out_csv}")
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