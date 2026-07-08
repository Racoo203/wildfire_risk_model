from pathlib import Path
from typing import Dict, Optional
import logging

import numpy as np
import geopandas as gpd
import mlflow
import rasterio

from ..metrics import compute_full_metrics, log_metrics_to_mlflow, save_metrics_sidecar
from ..evaluate import evaluate_on_test, generate_susceptibility_raster

logger = logging.getLogger(__name__)

CLASS_NAMES = ["Low", "Medium", "High", "Very High"]

class PostTrainingEvaluator:
    def __init__(self, config: dict):
        self.config = config

    def evaluate(
            self, 
            final_model, 
            X_val, 
            y_val, 
            season, 
            model_name, 
            ref_path, 
            fire_test_gdf, 
            x_coords, 
            y_coords
        ):
        defaults = {
            "shap_path": None, 
            "time_forward_validation": None, 
            "susceptibility_map_path": None, 
            "full_metrics_path": None
        }

        y_proba = final_model.predict_proba(X_val.values)
        full_metrics = compute_full_metrics(y_val.values, y_proba)
        log_metrics_to_mlflow(full_metrics)

        figures_dir = Path(self.config["base"]["figures_dir"])
        defaults["full_metrics_path"] = save_metrics_sidecar(full_metrics, figures_dir, season, model_name)

        from ...viz.charts import plot_confusion_matrix
        plot_confusion_matrix(full_metrics["confusion_matrix"], CLASS_NAMES, figures_dir, season=season, model_name=model_name)

        if ref_path is None:
            logger.info(f"[{season}][{model_name}] No ref_path supplied; skipping remaining post-training evaluation.")
            return defaults

        results = evaluate_on_test(
            final_model, X_val.values, y_val.values, list(X_val.columns),
            season, model_name, self.config,
            fire_test_gdf=fire_test_gdf, x_coords=x_coords, y_coords=y_coords,
        )

        results["susceptibility_map_path"] = None
        if x_coords is not None and y_coords is not None:
            results["susceptibility_map_path"] = generate_susceptibility_raster(
                final_model, X_val.values, x_coords, y_coords, ref_path, season, model_name, self.config,
            )

        return results
    
    @staticmethod
    def evaluate_on_test(
        final_model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list,
        season: str,
        model_name: str,
        config: dict,
        fire_test_gdf: Optional[gpd.GeoDataFrame] = None,
        x_coords: Optional[np.ndarray] = None,
        y_coords: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Runs the three post-training checks and logs everything to mlflow +
        figures/manifest.json. Returns a dict merged into train_one()'s result.

        Any individual check that fails logs a warning and is skipped rather
        than aborting the whole training run — a partial evaluation is more
        useful than losing an otherwise-successful training result.
        """
        results: Dict = {
            "shap_path": None,
            "time_forward_validation": None,
        }

        results["shap_path"] = PostTrainingEvaluator._try_shap_summary(
            final_model, X_test, feature_names, season, model_name, config
        )

        if fire_test_gdf is not None and x_coords is not None and y_coords is not None:
            results["time_forward_validation"] = PostTrainingEvaluator._try_time_forward_validation(
                final_model, X_test, x_coords, y_coords, fire_test_gdf, season, model_name, config
            )

        return results


    @staticmethod
    def _try_shap_summary(final_model, X_test, feature_names, season, model_name, config) -> Optional[Path]:
        try:
            import shap
            from ...viz.charts import plot_shap_summary

            # Tree models: TreeExplainer (fast, exact). SVM/NN: KernelExplainer
            # on a small background sample (expensive — cap sample size hard).
            underlying = getattr(final_model, "model", final_model)
            sample = X_test if len(X_test) <= 2000 else X_test[
                np.random.default_rng(42).choice(len(X_test), 2000, replace=False)
            ]

            if model_name in ("random_forest", "xgboost"):
                explainer = shap.TreeExplainer(underlying)
                shap_values = explainer.shap_values(sample)
            else:
                background = sample[: min(100, len(sample))]
                explainer = shap.KernelExplainer(final_model.predict_proba, background)
                shap_values = explainer.shap_values(sample[: min(200, len(sample))])

            figures_dir = Path(config["base"]["figures_dir"])
            path = plot_shap_summary(shap_values, feature_names, figures_dir, season=season, model_name=model_name)
            logger.info(f"[{season}][{model_name}] SHAP summary written -> {path}")
            return path

        except Exception as exc:
            logger.warning(f"[{season}][{model_name}] SHAP summary failed, skipping: {exc}")
            return None


    @staticmethod
    def _try_time_forward_validation(
        final_model, X_test, x_coords, y_coords, fire_test_gdf, season, model_name, config
    ) -> Optional[dict]:
        """
        Replicates the paper's headline validation figure (§1.5): predict the
        susceptibility class at every test-period fire location and report the
        % falling in each class. A well-calibrated map should show the large
        majority of real fires landing in Medium-Very High, mirroring the
        paper's 87.73% Medium-VeryHigh / 12.27% missed result.
        """
        try:
            if len(fire_test_gdf) == 0:
                logger.warning(f"[{season}][{model_name}] No test-period fires to validate against; skipping.")
                return None

            proba = final_model.predict_proba(X_test)
            pred_class = np.argmax(proba, axis=1)

            from scipy.spatial import cKDTree

            tree = cKDTree(np.column_stack([x_coords, y_coords]))
            fire_xy = np.column_stack([fire_test_gdf.geometry.x.values, fire_test_gdf.geometry.y.values])
            _, nearest_idx = tree.query(fire_xy, k=1)

            matched_classes = pred_class[nearest_idx]
            counts = {name: int(np.sum(matched_classes == i)) for i, name in enumerate(CLASS_NAMES)}
            total = len(matched_classes)
            pct = {f"pct_{name.lower().replace(' ', '_')}": 100 * c / total for name, c in counts.items()}
            pct["pct_medium_plus"] = 100 * int(np.sum(matched_classes >= 1)) / total

            for k, v in pct.items():
                mlflow.log_metric(f"tf_validation_{k}", v)

            logger.info(
                f"[{season}][{model_name}] Time-forward validation ({total} test fires): "
                f"{ {k: round(v, 2) for k, v in pct.items()} }"
            )
            return {"n_fires": total, "counts": counts, **pct}

        except Exception as exc:
            logger.warning(f"[{season}][{model_name}] Time-forward validation failed, skipping: {exc}")
            return None


    def generate_susceptibility_raster(
        final_model,
        X_full: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        ref_path: Path,
        season: str,
        model_name: str,
        config: dict,
    ) -> Optional[Path]:
        """
        Predicts the 0-3 class over every valid pixel in the reference grid and
        writes it as a GeoTIFF. This is the actual dissertation deliverable —
        everything else in this module is diagnostic/supporting evidence.
        """
        try:
            proba = final_model.predict_proba(X_full)
            pred_class = np.argmax(proba, axis=1).astype("float32")

            with rasterio.open(ref_path) as ref:
                meta = ref.meta.copy()
                height, width, transform = ref.height, ref.width, ref.transform

            out_grid = np.full((height, width), np.nan, dtype="float32")
            rows, cols = rasterio.transform.rowcol(transform, x_coords, y_coords)
            rows, cols = np.array(rows), np.array(cols)
            valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
            out_grid[rows[valid], cols[valid]] = pred_class[valid]

            meta.update({"dtype": "float32", "nodata": np.nan, "count": 1})
            out_dir = Path(config["base"]["output_dir"])
            out_path = out_dir / f"susceptibility_{model_name}_{season}.tif"
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(out_grid[np.newaxis, :, :])

            logger.info(f"[{season}][{model_name}] Susceptibility raster written -> {out_path.name}")
            return out_path

        except Exception as exc:
            logger.warning(f"[{season}][{model_name}] Susceptibility raster generation failed: {exc}")
            return None