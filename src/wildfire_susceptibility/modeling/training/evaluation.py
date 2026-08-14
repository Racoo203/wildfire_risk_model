from pathlib import Path
from typing import Dict, Optional
import logging

import mlflow

from ..categorical import cat_features_of, to_model_array
from ..class_labels import class_names_for
from ..metrics import compute_full_metrics, log_metrics_to_mlflow, save_metrics_sidecar
from ..evaluate import evaluate_on_test, generate_susceptibility_raster

logger = logging.getLogger(__name__)

class PostTrainingEvaluator:
    def __init__(self, config: dict):
        self.config = config

    def evaluate(
        self, final_model, X_val, y_val, season, model_name,
        ref_path, fire_test_gdf, x_coords, y_coords,
        X_full=None, full_x_coords=None, full_y_coords=None,
    ):
        """
        If called from ModelTrainer.train_one(), an mlflow run is already
        active and metrics attach to it. If called standalone (e.g. from
        stage_evaluate, reloading an already-trained model), no run is
        active — start and properly close one here instead of letting
        mlflow.log_metric() implicitly auto-start a run that never gets
        ended, which leaks into every subsequent test/run in-process.

        X_full/full_x_coords/full_y_coords, when given, scope the
        susceptibility *raster* to every in-domain pixel (regardless of
        whether it has a ground-truth label) — separate from X_val/
        x_coords/y_coords, which stay label-filtered because metrics need
        real ground truth. When omitted, the raster falls back to the
        label-filtered X_val/x_coords/y_coords, matching prior behavior.

        X_val and X_full (when given) must already be encoded consistently
        with how `final_model` was trained — i.e. already passed through
        DatasetPrep.apply_categorical_encoding using the training-fitted
        encoding_meta. This method only handles the DataFrame -> model
        array conversion (to_model_array, via cat_features_of(final_model)
        to recover which column(s) CatBoost needs as raw categoricals
        rather than a plain float array).
        """
        if mlflow.active_run() is None:
            with mlflow.start_run(run_name=f"{season}_{model_name}_eval"):
                return self._evaluate(
                    final_model, X_val, y_val, season, model_name,
                    ref_path, fire_test_gdf, x_coords, y_coords,
                    X_full, full_x_coords, full_y_coords,
                )
        return self._evaluate(
            final_model, X_val, y_val, season, model_name,
            ref_path, fire_test_gdf, x_coords, y_coords,
            X_full, full_x_coords, full_y_coords,
        )

    def _evaluate(
        self, final_model, X_val, y_val, season, model_name,
        ref_path, fire_test_gdf, x_coords, y_coords,
        X_full=None, full_x_coords=None, full_y_coords=None,
    ):
        defaults = {
            "shap_path": None, "shap_dependence_path": None, "time_forward_validation": None,
            "susceptibility_map_path": None, "full_metrics_path": None,
        }

        cat_positions = cat_features_of(final_model)
        X_val_arr = to_model_array(X_val, cat_positions)

        y_proba = final_model.predict_proba(X_val_arr)
        full_metrics = compute_full_metrics(y_val.values, y_proba)
        log_metrics_to_mlflow(full_metrics)

        figures_dir = Path(self.config["base"]["figures_dir"])
        defaults["full_metrics_path"] = save_metrics_sidecar(full_metrics, figures_dir, season, model_name)

        try:
            from ...viz.charts import plot_confusion_matrix
            class_names = class_names_for(len(full_metrics["confusion_matrix"]))
            plot_confusion_matrix(
                full_metrics["confusion_matrix"], class_names, figures_dir,
                season=season, model_name=model_name,
            )
        except Exception as exc:
            logger.warning(f"[{season}][{model_name}] Confusion matrix plot failed, skipping: {exc}")

        if ref_path is None:
            logger.info(f"[{season}][{model_name}] No ref_path supplied; skipping remaining post-training evaluation.")
            return defaults

        results = evaluate_on_test(
            final_model=final_model,
            X_test=X_val_arr,
            y_test=y_val.values,
            feature_names=list(X_val.columns),
            season=season,
            model_name=model_name,
            config=self.config,
            fire_test_gdf=fire_test_gdf,
            x_coords=x_coords,
            y_coords=y_coords,
        )

        raster_X = to_model_array(X_full, cat_positions) if X_full is not None else X_val_arr
        raster_x_coords = full_x_coords if full_x_coords is not None else x_coords
        raster_y_coords = full_y_coords if full_y_coords is not None else y_coords

        results["susceptibility_map_path"] = None
        if raster_x_coords is not None and raster_y_coords is not None:
            results["susceptibility_map_path"] = generate_susceptibility_raster(
                final_model=final_model,
                X_full=raster_X,
                x_coords=raster_x_coords,
                y_coords=raster_y_coords,
                ref_path=ref_path,
                season=season,
                model_name=model_name,
                config=self.config,
            )

        return {**defaults, **results}