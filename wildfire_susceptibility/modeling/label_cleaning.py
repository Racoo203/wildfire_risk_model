# wildfire_susceptibility/modeling/label_cleaning.py

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

HIGH_RISK_CLASSES = (1, 2, 3)

_CLUSTERING_DERIVED_FEATURES = {
    "diurnal_range": lambda arrays: arrays["tasmax"] - arrays["tasmin"],
}


class LabelCleaner:
    """
    Flags and removes ambiguous Low-risk pixels via pairwise k-means.

    Two entry points, same core logic:
        clean()      — 2D raster in, 2D raster out (legacy / raster-stage use)
        clean_flat()  — stacked dataframe in, cleaned label Series out
                         (used by modeling/dataset_prep.py on the raw
                         dataset_train_<season>.csv table — this is now
                         the only caller in the standard pipeline)
    """

    def __init__(self, config: dict, *args, **kwargs):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def clean_flat(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "label",
        season: Optional[str] = None,
    ) -> pd.Series:
        """
        Clean the label column of a stacked (one-row-per-pixel) dataframe.

        Only rows with a non-NaN label participate in k-means; rows with
        NaN labels (zero-density / unlabelled pixels) pass through
        unchanged. Returns a new Series (same index as `df`) — caller
        decides whether to assign it back into `df[label_col]`.

        `feature_cols` should be the modeling feature set (post
        resolve_missing, so no NaNs remain in-domain); this method does NOT
        do its own imputation.
        """
        clustering_df = self._build_clustering_view_df(df, feature_cols)
        valid_mask = df[label_col].notna().to_numpy()

        flat_labels = df.loc[valid_mask, label_col].to_numpy()
        feature_names = list(clustering_df.columns)
        flat_features = clustering_df.loc[valid_mask].to_numpy()

        cfg = self.config["labels"]
        self.CLUSTERING_VARIANCE_FLOOR_PCT = cfg.get("clustering_variance_floor_pct", 0.01)

        flagged = self._flag_low_pixels(
            flat_labels,
            flat_features,
            n_init=cfg["kmeans_n_init"],
            max_iter=cfg["kmeans_max_iter"],
            random_state=cfg["random_state"],
            sample_random_state=cfg["random_state"],
            max_sample=cfg["max_sample_per_class"],
            feature_names=feature_names,
        )

        cleaned_flat = flat_labels.copy().astype("float32")
        if flagged:
            cleaned_flat[np.array(list(flagged))] = np.nan

        self._log_removal(flat_labels, flagged, season)

        if cfg.get("run_sensitivity", True):
            self.sensitivity_analysis(
                flat_labels, flat_features, baseline_flagged=flagged,
                season=season, feature_names=feature_names,
            )

        clean_df = df.copy()
        clean_df.loc[valid_mask, label_col] = cleaned_flat
        return clean_df.dropna(subset=label_col)

    def _build_clustering_view_df(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Feature set used ONLY for k-means geometry — flat-dataframe variant."""
        exclude = set(self.config["labels"].get("clustering_exclude_features", ["tas", "tasmin"]))
        keep_cols = [c for c in feature_cols if c not in exclude]
        view = df[keep_cols].copy()
        if "tasmax" in df.columns and "tasmin" in df.columns:
            view["diurnal_range"] = df["tasmax"] - df["tasmin"]
        return view

    def _flag_low_pixels(
        self, flat_labels, flat_features, *, n_init, max_iter, random_state, max_sample,
        sample_random_state=None, feature_weights=None, feature_names=None, population_std=None,
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

            subset_std = raw_subset.std(axis=0)
            variance_floor = getattr(self, "CLUSTERING_VARIANCE_FLOOR_PCT", 0.01)
            degenerate = subset_std < variance_floor * np.maximum(population_std, 1e-8)

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
        self, flat_labels, flat_features, baseline_flagged, feature_names=None, season=None,
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
                n_init=variant["n_init"], max_iter=cfg["kmeans_max_iter"],
                random_state=variant["random_state"], sample_random_state=cfg["random_state"],
                max_sample=max_sample, feature_weights=weights, feature_names=feature_names,
            )

            symmetric_diff = len(baseline_flagged ^ flagged)
            variance_pct = 100 * symmetric_diff / low_size if low_size else 0.0

            rows.append({
                "label": variant["label"], "n_init": variant["n_init"],
                "random_state": variant["random_state"], "reweighted": variant["reweight"],
                "n_flagged": len(flagged),
                "pct_low_flagged": 100 * len(flagged) / low_size if low_size else 0.0,
                "variance_vs_baseline_pct": round(variance_pct, 4),
            })

        report = pd.DataFrame(rows)
        fully_flagged = report["n_flagged"] >= int(0.999 * max_sample)
        if fully_flagged.any():
            self.logger.warning(
                f"[{season}] {int(fully_flagged.sum())}/{len(report)} sensitivity variants flagged "
                f"~100% of the subsampled Low class — likely near-total separability, not genuine "
                f"k-means robustness. Low variance here is not the paper's <0.2% stability result."
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

    @staticmethod
    def _build_sensitivity_variants(cfg: dict) -> List[dict]:
        base_seed = cfg["random_state"]
        n_inits = cfg.get("sensitivity_n_inits", [10, 20])
        alt_seeds = cfg.get("sensitivity_seeds", [7, 123])
        n_reweight_trials = cfg.get("sensitivity_n_reweight_trials", 3)

        variants = []
        for n_init in n_inits:
            variants.append({"label": f"n_init={n_init}", "n_init": n_init, "random_state": base_seed, "reweight": False})
        for seed in alt_seeds:
            variants.append({"label": f"seed={seed}", "n_init": cfg["kmeans_n_init"], "random_state": seed, "reweight": False})
        for i in range(n_reweight_trials):
            variants.append({"label": f"reweight_trial_{i}", "n_init": cfg["kmeans_n_init"], "random_state": base_seed + i + 1, "reweight": True})
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

    def _log_removal(self, flat_labels: np.ndarray, flagged: Set[int], season: Optional[str]) -> None:
        low_size = int(np.sum(flat_labels == 0))
        pct = 100 * len(flagged) / low_size if low_size else 0.0
        self.logger.info(
            f"[{season}] LabelCleaner: flagged {len(flagged):,} / {low_size:,} "
            f"Low-class pixels ({pct:.2f}%) as ambiguous."
        )