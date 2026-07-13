from pathlib import Path
from typing import Dict, Optional
import logging

import mlflow

from ..metrics import compute_full_metrics, log_metrics_to_mlflow, save_metrics_sidecar
from ..evaluate import evaluate_on_test, generate_susceptibility_raster

logger = logging.getLogger(__name__)
CLASS_NAMES = ["Low", "Medium", "High", "Very High"]

class PostTrainingEvaluator:
    def __init__(self, config: dict):
        self.config = config

    def evaluate(
        self, final_model, X_val, y_val, season, model_name,
        ref_path, fire_test_gdf, x_coords, y_coords,
    ):
        """
        If called from ModelTrainer.train_one(), an mlflow run is already
        active and metrics attach to it. If called standalone (e.g. from
        stage_evaluate, reloading an already-trained model), no run is
        active — start and properly close one here instead of letting
        mlflow.log_metric() implicitly auto-start a run that never gets
        ended, which leaks into every subsequent test/run in-process.
        """
        if mlflow.active_run() is None:
            with mlflow.start_run(run_name=f"{season}_{model_name}_eval"):
                return self._evaluate(
                    final_model, X_val, y_val, season, model_name,
                    ref_path, fire_test_gdf, x_coords, y_coords,
                )
        return self._evaluate(
            final_model, X_val, y_val, season, model_name,
            ref_path, fire_test_gdf, x_coords, y_coords,
        )

    def _evaluate(
        self, final_model, X_val, y_val, season, model_name,
        ref_path, fire_test_gdf, x_coords, y_coords,
    ):
        defaults = {
            "shap_path": None, "time_forward_validation": None,
            "susceptibility_map_path": None, "full_metrics_path": None,
        }

        y_proba = final_model.predict_proba(X_val.values)
        full_metrics = compute_full_metrics(y_val.values, y_proba)
        log_metrics_to_mlflow(full_metrics)

        figures_dir = Path(self.config["base"]["figures_dir"])
        defaults["full_metrics_path"] = save_metrics_sidecar(full_metrics, figures_dir, season, model_name)

        try:
            from ...viz.charts import plot_confusion_matrix
            plot_confusion_matrix(
                full_metrics["confusion_matrix"], CLASS_NAMES, figures_dir,
                season=season, model_name=model_name,
            )
        except Exception as exc:
            logger.warning(f"[{season}][{model_name}] Confusion matrix plot failed, skipping: {exc}")

        if ref_path is None:
            logger.info(f"[{season}][{model_name}] No ref_path supplied; skipping remaining post-training evaluation.")
            return defaults

        results = evaluate_on_test(
            final_model=final_model,
            X_test=X_val.values,
            y_test=y_val.values,
            feature_names=list(X_val.columns),
            season=season,
            model_name=model_name,
            config=self.config,
            fire_test_gdf=fire_test_gdf,
            x_coords=x_coords,
            y_coords=y_coords,
        )

        results["susceptibility_map_path"] = None
        if x_coords is not None and y_coords is not None:
            results["susceptibility_map_path"] = generate_susceptibility_raster(
                final_model=final_model,
                X_full=X_val.values,
                x_coords=x_coords,
                y_coords=y_coords,
                ref_path=ref_path,
                season=season,
                model_name=model_name,
                config=self.config,
            )

        return {**defaults, **results}