"""Full classification metric suite for one (season, model) validation
pass. Called once from PostTrainingEvaluator after the final refit,
logged to mlflow, and written to figures/manifest.json as a JSON sidecar
so the results dashboard can render a metrics table without recomputing."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support, confusion_matrix,
    log_loss, cohen_kappa_score, balanced_accuracy_score, matthews_corrcoef,
)

from .class_labels import class_names_for

logger = logging.getLogger(__name__)

def compute_full_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Returns a flat dict of scalar metrics (mlflow-loggable) plus a
    'confusion_matrix' key (2D list, not scalar — logged separately as
    a figure, not an mlflow metric).
    """
    y_pred = np.argmax(y_proba, axis=1)
    n_classes = y_proba.shape[1]
    if class_names is None:
        class_names = class_names_for(n_classes)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(n_classes), zero_division=0
    )

    auc_ovr_macro = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    auc_ovr_weighted = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
    gini_macro = 2 * auc_ovr_macro - 1  # Gini = 2*AUC - 1, standard credit-risk-style transform

    metrics = {
        "accuracy": float(np.mean(y_pred == y_true)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "auc_ovr_macro": float(auc_ovr_macro),
        "auc_ovr_weighted": float(auc_ovr_weighted),
        "gini_macro": float(gini_macro),
        "log_loss": float(log_loss(y_true, y_proba, labels=range(n_classes))),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        # Quadratic-weighted kappa: classes are an ordinal risk discretization
        # (Low < Medium < High < Very High), so distant misclassifications
        # (Low predicted as Very High) are penalized more than adjacent ones
        # (Low predicted as Medium) — unlike unweighted cohen_kappa above.
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
    }

    for i, name in enumerate(class_names[:n_classes]):
        key = name.lower().replace(" ", "_")
        metrics[f"precision_{key}"] = float(precision[i])
        metrics[f"recall_{key}"] = float(recall[i])
        metrics[f"f1_{key}"] = float(f1[i])
        metrics[f"support_{key}"] = int(support[i])
        try:
            metrics[f"auc_{key}_vs_rest"] = float(
                roc_auc_score((y_true == i).astype(int), y_proba[:, i])
            )
        except ValueError:
            logger.warning(f"Could not compute one-vs-rest AUC for class '{name}' (likely a single-class fold).")

    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes)).tolist()

    return {"scalars": metrics, "confusion_matrix": cm}


def log_metrics_to_mlflow(metrics: Dict) -> None:
    import mlflow
    for k, v in metrics["scalars"].items():
        mlflow.log_metric(k, v)


def save_metrics_sidecar(metrics: Dict, figures_dir: Path, season: str, model_name: str) -> Path:
    out_path = Path(figures_dir) / (season or "static") / "models" / f"metrics_{model_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    return out_path