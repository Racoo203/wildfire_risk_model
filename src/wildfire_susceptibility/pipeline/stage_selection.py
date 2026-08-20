"""Stage: model selection.

Aggregates every (season, model) manifest + eval result into one
comparison table, applies config.modeling.selection_rule to pick a
winner per season, and runs a Kruskal-Wallis test across models'
spatial-CV fold scores per season to report whether the differences are
statistically significant — not just numerically different. Run twice,
once on PR-AUC-macro (the selection metric, see _select_best_per_season)
and once on QWK (meaningful in its own right since the risk classes are
ordinal and QWK penalizes distant misclassifications more than adjacent
ones, unlike PR-AUC-macro's unordered one-vs-rest averaging) — the two
can disagree, and that disagreement is itself informative rather than a
bug to reconcile.

    stage_selection(config, input_paths) -> {
        "comparison_table": Path,      # csv
        "selection_summary": Path,     # json
        "kruskal_results": Path,       # csv, PR-AUC-macro
        "kruskal_results_qwk": Path,   # csv, QWK
    }

`input_paths` must contain:
    "train": {"<season>": {"<model_name>": Path}}          # artifact dirs
    "evaluate": {"<season>": {"<model_name>": dict}}        # eval results
"""
import json
import math
from pathlib import Path
from typing import Dict

import pandas as pd
from scipy.stats import kruskal

from ..utils.logger import setup_logger


def stage_selection(config: dict, input_paths: dict) -> Dict[str, Path]:
    # logger = setup_logger(log_file=config["logging"]["log_path"], level=config["logging"]["level"])
    logger = setup_logger()
    train_artifacts = input_paths["train"]
    eval_results = input_paths["evaluate"]
    selection_rule = config["modeling"].get("selection_rule", "best_pr_auc")

    rows = []
    manifests: Dict[str, Dict[str, dict]] = {}
    for season, models in train_artifacts.items():
        manifests[season] = {}
        for model_name, artifact_dir in models.items():
            manifest = json.loads((Path(artifact_dir) / "manifest.json").read_text())
            manifests[season][model_name] = manifest
            eval_result = eval_results.get(season, {}).get(model_name, {})
            gap = manifest.get("cv_optimism_gap") or {}
            rows.append({
                "season": season,
                "model": model_name,
                "cv_auc_standard": manifest["cv_auc_standard"],
                "cv_auc_spatial": manifest["cv_auc_spatial"],
                "cv_auc_optimism_gap": gap.get("auc"),
                "cv_f1_macro_standard": manifest.get("cv_f1_macro_standard"),
                "cv_f1_macro_spatial": manifest.get("cv_f1_macro_spatial"),
                "cv_f1_macro_optimism_gap": gap.get("f1_macro"),
                "cv_pr_auc_macro_standard": manifest.get("cv_pr_auc_macro_standard"),
                "cv_pr_auc_macro_spatial": manifest.get("cv_pr_auc_macro_spatial"),
                "cv_pr_auc_macro_optimism_gap": gap.get("pr_auc_macro"),
                "cv_qwk_standard": manifest.get("cv_qwk_standard"),
                "cv_qwk_spatial": manifest.get("cv_qwk_spatial"),
                "cv_qwk_optimism_gap": gap.get("qwk"),
                "val_auc": manifest["val_auc"],
                "val_f1": manifest["val_f1"],
                "val_pr_auc_macro": manifest.get("val_pr_auc_macro"),
                "val_qwk": manifest.get("val_qwk"),
                "tf_pct_medium_plus": (eval_result.get("time_forward_validation") or {}).get("pct_medium_plus"),
            })

    comparison_df = pd.DataFrame(rows).sort_values(["season", "cv_pr_auc_macro_standard"], ascending=[True, False])
    figures_dir = Path(config["base"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = figures_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    selection_summary = _select_best_per_season(manifests, selection_rule)
    selection_path = figures_dir / "selection_summary.json"
    selection_path.write_text(json.dumps(selection_summary, indent=2))

    kruskal_results = _kruskal_wallis_per_season(manifests, fold_field="cv_pr_auc_macro_spatial_folds")
    kruskal_path = figures_dir / "kruskal_wallis.csv"
    _kruskal_results_to_csv(kruskal_results).to_csv(kruskal_path, index=False)

    kruskal_results_qwk = _kruskal_wallis_per_season(manifests, fold_field="cv_qwk_spatial_folds")
    kruskal_qwk_path = figures_dir / "kruskal_wallis_qwk.csv"
    _kruskal_results_to_csv(kruskal_results_qwk).to_csv(kruskal_qwk_path, index=False)

    logger.info(f"[stage_selection] Comparison table -> {comparison_path.name}")
    logger.info(f"[stage_selection] Selection summary -> {selection_path.name}")
    logger.info(f"[stage_selection] Kruskal-Wallis results (PR-AUC-macro) -> {kruskal_path.name}")
    logger.info(f"[stage_selection] Kruskal-Wallis results (QWK) -> {kruskal_qwk_path.name}")

    return {
        "comparison_table": comparison_path,
        "selection_summary": selection_path,
        "kruskal_results": kruskal_path,
        "kruskal_results_qwk": kruskal_qwk_path,
    }


def _select_best_per_season(manifests: Dict[str, Dict[str, dict]], selection_rule: str) -> Dict[str, dict]:
    summary = {}
    for season, models in manifests.items():
        if selection_rule == "best_auc":
            best_model = max(models, key=lambda m: models[m]["val_auc"])
        elif selection_rule == "best_pr_auc":
            # PR-AUC-macro is Optuna's actual HPO objective (search.py) —
            # picked over plain AUC there specifically because AUC "stays
            # high under majority-class collapse" in a way PR-AUC doesn't.
            # This rule closes the gap where the searched-for metric and
            # the cross-model-selected-on metric used to disagree: winner
            # selection now uses the same held-out-validation PR-AUC-macro
            # (score_multiclass_fold via trainer.py::_fit_final_and_validate),
            # not a CV mean.
            best_model = max(models, key=lambda m: models[m]["val_pr_auc_macro"])
        elif selection_rule == "most_conservative":
            # smallest optimism gap (standard - spatial CV PR-AUC-macro)
            # among models within 1% of the best standard PR-AUC-macro —
            # favors generalizable models. Uses PR-AUC-macro rather than
            # AUC for the same majority-class-collapse-insensitivity reason
            # as best_pr_auc above, so this rule and best_pr_auc agree on
            # what "good" means.
            best_standard = max(m["cv_pr_auc_macro_standard"] for m in models.values())
            candidates = {
                name: m for name, m in models.items()
                if m["cv_pr_auc_macro_standard"] >= best_standard - 0.01
            }
            best_model = min(
                candidates,
                key=lambda m: candidates[m]["cv_pr_auc_macro_standard"] - candidates[m]["cv_pr_auc_macro_spatial"],
            )
        else:
            raise ValueError(f"Unknown selection_rule '{selection_rule}'")

        summary[season] = {"selected_model": best_model, "selection_rule": selection_rule, **models[best_model]}
    return summary


def _kruskal_wallis_per_season(manifests: Dict[str, Dict[str, dict]], fold_field: str) -> Dict[str, dict]:
    """
    Kruskal-Wallis H-test across models' spatial-CV fold scores, per
    season, for whichever per-fold metric list `fold_field` names on the
    manifest (currently "cv_pr_auc_macro_spatial_folds" or
    "cv_qwk_spatial_folds" — see stage_selection's two call sites).
    PR-AUC-macro is the metric best_pr_auc/most_conservative actually
    select on (see _select_best_per_season); QWK is run as a second,
    independent significance test because it's the metric that actually
    accounts for the risk classes being ordinal (penalizes distant
    misclassifications more than adjacent ones), which PR-AUC-macro's
    unordered one-vs-rest averaging does not — the two tests are allowed
    to disagree.

    ASSUMPTION: kruskal() treats each model's fold-score list as an
    independent sample — it does NOT require fold i in model A to
    correspond to fold i in model B (unlike a paired test). This holds
    regardless of fold alignment, so no assumption about matching fold
    splits across models is actually required here. What IS assumed is
    that within a single (season, model) pair, the named fold_field
    was produced by ModelTrainer._spatial_cv_check's FoldStrategy.make_folds
    call — i.e. every fold score in the list came from the same spatial
    block partition for that model's search space. Since all models for
    a given season currently share one FoldStrategy instance and one
    groups_train assignment (see stage_train.py), fold counts are in
    practice equal across models within a season, but Kruskal-Wallis
    doesn't require this — it tolerates unequal group sizes.

    The named fold_field can contain NaN entries (see
    cv/base.py::score_multiclass_fold) for folds whose validation split
    shared fewer than 2 classes with what the model could score — those
    are dropped here before scipy ever sees them, per fold, per model, so
    a model with at least one genuinely scorable fold still contributes
    its real (non-NaN) fold scores rather than being excluded outright or
    letting a single NaN poison scipy.stats.kruskal's whole result (its
    default nan_policy="propagate" would otherwise turn the statistic/
    p-value NaN for every model being compared, not just the affected
    one).
    """
    results = {}
    for season, models in manifests.items():
        groups = {}
        for name, m in models.items():
            folds = m.get(fold_field)
            if not folds:
                continue
            scored = [f for f in folds if not math.isnan(f)]
            if scored:
                groups[name] = scored

        if len(groups) < 2:
            results[season] = {"skipped": "fewer than 2 models with fold-level scores"}
            continue

        stat, p_value = kruskal(*groups.values())
        results[season] = {
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant_at_0.05": bool(p_value < 0.05),
            "models_compared": list(groups.keys()),
        }
    return results


def _kruskal_results_to_csv(kruskal_results: Dict[str, dict]) -> pd.DataFrame:
    """Flattens _kruskal_wallis_per_season's per-season dict into one row
    per season, matching the row-per-entity convention every other stat
    table in figures/ uses (e.g. temporal_eda/stationarity_summary.csv) —
    rather than the nested kruskal_wallis.json this replaces."""
    rows = []
    for season, result in kruskal_results.items():
        if "skipped" in result:
            rows.append({
                "season": season,
                "statistic": None,
                "p_value": None,
                "significant_at_0.05": None,
                "n_models_compared": None,
                "models_compared": None,
                "skip_reason": result["skipped"],
            })
        else:
            rows.append({
                "season": season,
                "statistic": result["statistic"],
                "p_value": result["p_value"],
                "significant_at_0.05": result["significant_at_0.05"],
                "n_models_compared": len(result["models_compared"]),
                "models_compared": ";".join(result["models_compared"]),
                "skip_reason": None,
            })
    return pd.DataFrame(rows, columns=[
        "season", "statistic", "p_value", "significant_at_0.05",
        "n_models_compared", "models_compared", "skip_reason",
    ])