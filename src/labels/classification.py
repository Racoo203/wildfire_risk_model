"""Load and filter fire occurence data."""

import numpy as np
from typing import Union, Tuple, Optional
import logging

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import rasterio

from ..core.base import VarBuilder

logger = logging.getLogger(__name__)

class LabelCleaner(VarBuilder):

    def process(self):
        return

    def clean(
        self,
        labels,
        feature_arrays,
        season
    ):
        out_paths = {
            "risk_labels_clean": self.output_dir / f"{self._seasonal_name('risk_labels_clean', season)}.tif"
        }

        if self._check_cache(f"LabelCleaner[{season}]", out_paths):
            logger.info(f"[CACHED] Cleaned labels ({season}) already exist")
            with rasterio.open(out_paths["risk_labels_clean"]) as src:
                return src.read(1)

        max_sample_per_class = self.config["labels"]["max_sample_per_class"]

        valid_mask = ~np.isnan(labels)
        flat_labels = labels[valid_mask]
        feature_names = list(feature_arrays.keys())
        flat_features = np.column_stack(
            [feature_arrays[name][valid_mask] for name in feature_names]
        )

        scaler = StandardScaler()
        low_idx = np.where(flat_labels == 0)[0]
        to_remove = set()

        for high_class in [1, 2, 3]:
            high_idx = np.where(flat_labels == high_class)[0]
            if len(high_idx) == 0:
                continue

            low_sample = (
                low_idx if len(low_idx) <= max_sample_per_class
                else np.random.choice(
                    low_idx, max_sample_per_class, replace = False
                )
            )

            high_sample = (
                high_idx if len(high_idx) <= max_sample_per_class
                else np.random.choice(
                    high_idx, max_sample_per_class, replace = False
                )
            )

            subset_idx = np.concatenate([low_sample, high_sample])
            X = scaler.fit_transform(flat_features[subset_idx])
            km = KMeans(
                n_clusters = self.config["labels"]["kmeans_k"],
                n_init = self.config["labels"]["kmeans_n_init"],
                max_iter = self.config["labels"]["kmeans_max_iter"],
                random_state = self.config["labels"]["random_state"]
            )
            km.fit(X)

            n_low = len(low_sample)
            high_centroid = np.bincount(km.labels_[n_low:]).argmax()
            flagged = low_sample[km.labels_[:n_low] == high_centroid]
            to_remove.update(flagged.toList())

        
        cleaned_flat = flat_labels.copy().astype("float32")
        if to_remove:
            cleaned_flat[np.array(list(to_remove))] = np.nan
        
        cleaned_labels = np.full(labels.shape, np.nan, dtype = "float32")
        cleaned_labels[valid_mask] = cleaned_flat

        with rasterio.open(self.ref_path) as ref:
            meta = ref.meta.copy()
            meta.update({"dtype": "float32", "nodata": np.nan, "count": 1})
        
        with rasterio.open(out_paths["risk_labels_clean"], "w", **meta) as dst:
            dst.write(cleaned_labels[np.newaxis, :, :])
        
        return cleaned_labels