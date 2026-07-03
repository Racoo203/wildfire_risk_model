from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import rasterio
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ..core.base import VarBuilder

HIGH_RISK_CLASSES = (1, 2, 3)  # Medium, High, Very High — each paired against Low (0)

class LabelCleaner(VarBuilder):
    """
    Flags and removes ambiguous Low-risk pixels via pairwise k-means,
    replicating the paper's methodology (k=2, StandardScaler, Euclidean;
    n_init=10, max_iter=300, random_state=42), and runs a sensitivity
    analysis (alternate n_init / seeds / feature reweighting) to confirm
    the flagged set is stable rather than an artifact of one random run.
    """

    def process(self):
        return

    def _build_clustering_view(
        self, feature_arrays: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Feature set used ONLY for k-means geometry — not for modeling."""
        view = {k: v for k, v in feature_arrays.items() if k not in self.CLUSTERING_EXCLUDE}
        for name, fn in self.CLUSTERING_DERIVED.items():
            if all(dep in feature_arrays for dep in ("tasmax", "tasmin")):
                view[name] = fn(feature_arrays)
        return view
    
    def _flag_low_pixels(
        self, flat_labels, flat_features, *, n_init, max_iter, random_state,
        max_sample, sample_random_state=None, feature_weights=None,
        feature_names=None,
        population_std=None,
    ):
        sample_random_state = sample_random_state if sample_random_state is not None else random_state
        rng = np.random.default_rng(sample_random_state)
        low_idx = np.where(flat_labels == 0)[0]
        low_sample = self._subsample(low_idx, max_sample, rng)

        if population_std is None:
            population_std = flat_features.std(axis=0)

        scaler = StandardScaler()
        flagged: Set[int] = set()

        for high_class in HIGH_RISK_CLASSES:
            high_idx = np.where(flat_labels == high_class)[0]
            if len(high_idx) == 0:
                self.logger.warning(f"No pixels in class {high_class}; skipping pair (Low, {high_class}).")
                continue

            high_sample = self._subsample(high_idx, max_sample, rng)
            subset_idx = np.concatenate([low_sample, high_sample])
            raw_subset = flat_features[subset_idx]

            # --- variance-floor guard -------------------------------
            subset_std = raw_subset.std(axis=0)
            degenerate = subset_std < 0.01 * np.maximum(population_std, 1e-8)
            if degenerate.any() and feature_names is not None:
                names = np.array(feature_names)[degenerate].tolist()
                self.logger.warning(
                    f"[Low vs {high_class}] excluding near-constant-in-subsample "
                    f"feature(s) from clustering (std < 1% of population std): {names}"
                )

            X = scaler.fit_transform(raw_subset[:, ~degenerate])
            if feature_weights is not None:
                X = X * feature_weights[~degenerate]

            km = KMeans(n_clusters=2, n_init=n_init, max_iter=max_iter, random_state=random_state)
            cluster_ids = km.fit_predict(X)

            n_low = len(low_sample)
            high_centroid = np.bincount(cluster_ids[n_low:]).argmax()
            flagged.update(low_sample[cluster_ids[:n_low] == high_centroid].tolist())

        return flagged

    def sensitivity_analysis(
        self,
        flat_labels: np.ndarray,
        flat_features: np.ndarray,
        baseline_flagged: Set[int],
        feature_names=None,
        season: Optional[str] = None,
    ) -> pd.DataFrame:
        cfg = self.config["labels"]

        max_sample = cfg["max_sample_per_class"]
        low_size = int(np.sum(flat_labels == 0))
        n_features = flat_features.shape[1]

        variants = self._build_sensitivity_variants(cfg)
        rows = []

        for variant in variants:
            rng = np.random.default_rng(variant["random_state"])
            weights = None
            if variant["reweight"]:
                noise_std = cfg.get("sensitivity_feature_noise_std", 0.05)
                weights = rng.normal(loc=1.0, scale=noise_std, size=n_features)

            flagged = self._flag_low_pixels(
                flat_labels, flat_features,
                n_init=variant["n_init"],
                max_iter=cfg["kmeans_max_iter"],
                random_state=variant["random_state"],
                sample_random_state=cfg["random_state"],
                max_sample=max_sample,
                feature_weights=weights,
                feature_names=feature_names,
            )

            symmetric_diff = len(baseline_flagged ^ flagged)
            variance_pct = 100 * symmetric_diff / low_size if low_size else 0.0

            rows.append({
                "label": variant["label"],
                "n_init": variant["n_init"],
                "random_state": variant["random_state"],
                "reweighted": variant["reweight"],
                "n_flagged": len(flagged),
                "pct_low_flagged": 100 * len(flagged) / low_size if low_size else 0.0,
                "variance_vs_baseline_pct": round(variance_pct, 4),
            })

        report = pd.DataFrame(rows)
        fully_flagged = report["n_flagged"] >= int(0.999 * max_sample)
        if fully_flagged.any():
            self.logger.warning(
                f"[{season}] {int(fully_flagged.sum())}/{len(report)} sensitivity variants flagged "
                f"~100% of the subsampled Low class — this indicates near-total separability "
                f"between Low and high-risk classes in feature space (a dominant feature, likely "
                f"d_fires, may be driving this), not genuine k-means robustness. Low variance here "
                f"is not the same as the paper's <0.2% stability result."
            )

        figures_dir = Path(self.config["base"]["figures_dir"])
        figures_dir.mkdir(parents=True, exist_ok=True)
        out_path = figures_dir / f"label_cleaning_sensitivity_{season or 'static'}.csv"
        report.to_csv(out_path, index=False)

        max_variance = report["variance_vs_baseline_pct"].max()
        self.logger.info(
            f"[{season}] Sensitivity analysis: {len(variants)} variants, "
            f"max variance vs. baseline = {max_variance:.4f}% -> {out_path.name}"
        )
        if max_variance > 1.0:
            self.logger.warning(
                f"[{season}] Sensitivity variance ({max_variance:.4f}%) exceeds 1% — "
                f"flagged set may be sensitive to k-means initialization."
            )

        return report

    def clean(
        self,
        labels: np.ndarray,
        feature_arrays: Dict[str, np.ndarray],
        season: Optional[str] = None,
    ) -> np.ndarray:
        out_paths = {
            "risk_labels_clean": self.output_dir / f"{self._seasonal_name('risk_labels_clean', season)}.tif"
        }

        if self._check_cache(f"LabelCleaner[{season}]", out_paths):
            self.logger.info(f"[CACHED] Cleaned labels ({season}) already exist")
            with rasterio.open(out_paths["risk_labels_clean"]) as src:
                return src.read(1)
            
        cfg = self.config["labels"]

        self.CLUSTERING_EXCLUDE = cfg.CLUSTERING_EXCLUDE
        self.CLUSTERING_DERIVED = cfg.CLUSTERING_DERIVED

        clustering_arrays = self._build_clustering_view(feature_arrays)
        valid_mask, flat_labels, flat_features, feature_names = self._flatten_valid(
            labels, clustering_arrays
        )

        flagged = self._flag_low_pixels(
            flat_labels, flat_features,
            n_init=cfg["kmeans_n_init"],
            max_iter=cfg["kmeans_max_iter"],
            random_state=cfg["random_state"],
            sample_random_state=cfg["random_state"],   # fixed baseline sample
            max_sample=cfg["max_sample_per_class"],
            feature_names=feature_names
        )

        cleaned_labels = self._apply_flags(labels, valid_mask, flat_labels, flagged)
        self._log_removal(flat_labels, flagged, season)
        self._write_raster(cleaned_labels, out_paths["risk_labels_clean"])

        if cfg.get("run_sensitivity", True):
            self.sensitivity_analysis(
                flat_labels, 
                flat_features, 
                baseline_flagged=flagged, 
                season=season,
                feature_names=feature_names
            )

        return cleaned_labels

    @staticmethod
    def _build_sensitivity_variants(cfg: dict) -> List[dict]:
        base_seed = cfg["random_state"]
        n_inits = cfg.get("sensitivity_n_inits", [10, 20])
        alt_seeds = cfg.get("sensitivity_seeds", [7, 123])
        n_reweight_trials = cfg.get("sensitivity_n_reweight_trials", 3)

        variants = []
        for n_init in n_inits:
            variants.append({"label": f"n_init={n_init}", "n_init": n_init,
                              "random_state": base_seed, "reweight": False})
        for seed in alt_seeds:
            variants.append({"label": f"seed={seed}", "n_init": cfg["kmeans_n_init"],
                              "random_state": seed, "reweight": False})
        for i in range(n_reweight_trials):
            variants.append({"label": f"reweight_trial_{i}", "n_init": cfg["kmeans_n_init"],
                              "random_state": base_seed + i + 1, "reweight": True})
        return variants

    @staticmethod
    def _flatten_valid(labels: np.ndarray, feature_arrays: Dict[str, np.ndarray]):
        valid_mask = ~np.isnan(labels)
        flat_labels = labels[valid_mask]
        feature_names = list(feature_arrays.keys())
        flat_features = np.column_stack([feature_arrays[name][valid_mask] for name in feature_names])
        return valid_mask, flat_labels, flat_features, feature_names

    @staticmethod
    def _subsample(idx: np.ndarray, max_n: int, rng: np.random.Generator) -> np.ndarray:
        if len(idx) <= max_n:
            return idx
        return rng.choice(idx, max_n, replace=False)

    def _apply_flags(
        self,
        labels: np.ndarray,
        valid_mask: np.ndarray,
        flat_labels: np.ndarray,
        flagged: Set[int],
    ) -> np.ndarray:
        cleaned_flat = flat_labels.copy().astype("float32")
        if flagged:
            cleaned_flat[np.array(list(flagged))] = np.nan

        cleaned_labels = np.full(labels.shape, np.nan, dtype="float32")
        cleaned_labels[valid_mask] = cleaned_flat
        return cleaned_labels

    def _log_removal(self, flat_labels: np.ndarray, flagged: Set[int], season: Optional[str]) -> None:
        low_size = int(np.sum(flat_labels == 0))
        pct = 100 * len(flagged) / low_size if low_size else 0.0
        self.logger.info(
            f"[{season}] LabelCleaner: flagged {len(flagged):,} / {low_size:,} "
            f"Low-class pixels ({pct:.2f}%) as ambiguous."
        )

    def _write_raster(self, cleaned_labels: np.ndarray, out_path: Path) -> None:
        with rasterio.open(self.ref_path) as ref:
            meta = ref.meta.copy()
        meta.update({"dtype": "float32", "nodata": np.nan, "count": 1})
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(cleaned_labels[np.newaxis, :, :])