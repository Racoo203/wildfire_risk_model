"""
Extract MLflow metrics + artifacts for dissertation final-fit models.

Usage:
    python export_dissertation_results.py \
        --experiment wf-dissertation-final \
        --out dissertation_results/

Notes:
- run.data.metrics only returns the LATEST logged value per key. If you
  logged metrics per-step (e.g. per Optuna trial, per epoch), use
  client.get_metric_history(run_id, key) instead to get the full series
  (useful for Optuna optimization-history / HPO convergence plots).
- Adjust `get_final_runs` matching logic to however you actually tag
  final-fit runs (run name, a "stage" tag/param, etc.) — the substring
  match on run name is a placeholder.
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from mlflow.tracking import MlflowClient


MODEL_NAMES = ("summer_ordinal_lr", "summer_random_forest", "summer_catboost", "summer_xgboost")

# RQ1: standard vs spatial CV optimism gap, per metric
RQ1_METRIC_KEYS = [
    "cv_auc_standard", "cv_auc_spatial", "cv_auc_optimism_gap",
    "cv_pr_auc_macro_standard", "cv_pr_auc_macro_spatial", "cv_pr_auc_macro_optimism_gap",
    "cv_f1_macro_standard", "cv_f1_macro_spatial", "cv_f1_macro_optimism_gap",
    "cv_qwk_standard", "cv_qwk_spatial", "cv_qwk_optimism_gap",
]

# Headline results table (main text)
HEADLINE_METRIC_KEYS = [
    "cv_auc_spatial", "cv_pr_auc_macro_spatial", "cv_f1_macro_spatial", "cv_qwk_spatial",
]

# Per-class breakdown (appendix). Names come from
# modeling/class_labels.py's 3-class scheme (Low/Medium/High), which is
# what configs/experiment/baseline.yaml and dissertation.yaml both run
# (labels.n_classes: 3) — update this if n_classes changes.
PERCLASS_METRIC_KEYS = [
    "precision_low", "recall_low", "f1_low", "support_low", "auc_low_vs_rest",
    "precision_medium", "recall_medium", "f1_medium", "support_medium", "auc_medium_vs_rest",
    "precision_high", "recall_high", "f1_high", "support_high", "auc_high_vs_rest",
]

# Time-forward validation (evaluate.py::_try_time_forward_validation) —
# replicates the paper's headline validation figure: % of held-out
# test-period fires landing in each predicted class. Logged with a
# "tf_validation_" prefix; pct_medium_plus mirrors the paper's
# "% Medium-VeryHigh" headline number. Also keyed off the 3-class scheme.
TF_VALIDATION_METRIC_KEYS = [
    "tf_validation_pct_low", "tf_validation_pct_medium", "tf_validation_pct_high",
    "tf_validation_pct_medium_plus",
]


def get_final_runs(client, experiment_name, model_names=MODEL_NAMES):
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["start_time DESC"],
    )
    final_runs = {}
    for r in runs:
        name = r.data.tags.get("mlflow.runName", "").lower()
        for m in model_names:
            if m in name and m not in final_runs:
                final_runs[m] = r
    missing = set(model_names) - set(final_runs)
    if missing:
        print(f"WARNING: no matching run found for: {missing}. "
              f"Check run-naming assumption in get_final_runs().")
    return final_runs


def export_metrics(run, model_name, out_dir):
    metrics = run.data.metrics
    df = pd.DataFrame(sorted(metrics.items()), columns=["metric", "value"])
    df.to_csv(out_dir / f"{model_name}_metrics.csv", index=False)
    return metrics


def export_artifacts(client, run_id, model_name, out_dir):
    dest = out_dir / model_name / "artifacts"
    dest.mkdir(parents=True, exist_ok=True)
    downloaded = []

    def _walk(path=""):
        for a in client.list_artifacts(run_id, path):
            if a.is_dir:
                _walk(a.path)
            else:
                p = client.download_artifacts(run_id, a.path, str(dest))
                downloaded.append(p)

    _walk()
    if not downloaded:
        print(f"  (no artifacts found for {model_name} — plots may not be logged to MLflow)")
    return downloaded


def build_table(all_metrics: dict, metric_keys: list, out_path: Path):
    rows = []
    for model, metrics in all_metrics.items():
        row = {"model": model}
        for k in metric_keys:
            row[k] = metrics.get(k, float("nan"))
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    df.to_csv(out_path)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="wf-diss")
    parser.add_argument("--tracking-uri", default="http://127.0.0.1:5000")
    parser.add_argument("--out", default="results/")
    args = parser.parse_args()

    import mlflow
    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    final_runs = get_final_runs(client, args.experiment)
    all_metrics = {}
    for model_name, run in final_runs.items():
        print(f"Exporting {model_name} (run_id={run.info.run_id})")
        all_metrics[model_name] = export_metrics(run, model_name, out_dir)
        export_artifacts(client, run.info.run_id, model_name, out_dir)

    with open(out_dir / "all_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    build_table(all_metrics, RQ1_METRIC_KEYS, out_dir / "rq1_optimism_gap_table.csv")
    build_table(all_metrics, HEADLINE_METRIC_KEYS, out_dir / "headline_results_table.csv")
    build_table(all_metrics, PERCLASS_METRIC_KEYS, out_dir / "perclass_appendix_table.csv")
    build_table(all_metrics, TF_VALIDATION_METRIC_KEYS, out_dir / "tf_validation_table.csv")

    print(f"\nDone. CSVs + artifacts in {out_dir.resolve()}")
    print("Feed the CSVs into a small pandas -> LaTeX (df.to_latex / booktabs) step,")
    print("or paste values directly into the templates you already have.")


if __name__ == "__main__":
    main()