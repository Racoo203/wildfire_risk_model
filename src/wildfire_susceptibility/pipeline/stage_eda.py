"""Stage: exploratory data analysis.

Reproduces eda.ipynb's figures as a pipeline stage: NaN coverage
(pre-imputation), VIF/Spearman correlation (post-imputation), class
balance before/after label cleaning, and one terrain-backdrop factor
map per predictor feature. All figures land in figures/manifest.json
via viz/.

Also writes a full-dataset Spearman significance export (every column,
not just the modeling feature subset the vif/spearman heatmap covers --
see viz/spearman_significance.py) as a flat CSV alongside those figures.
It's a diagnostic table, not a rendered figure, so it does not go through
figures/manifest.json (matching vif.csv/correlation_spearman.csv's
existing convention).

Factor maps reuse viz.render_all_factor_maps -- the same terrain-backdrop
renderer stage_evaluate.py uses for the susceptibility map -- so feature
rasters share its CRS/resolution/extent/domain-masking behaviour and
figure conventions exactly (every seasonal/climate feature raster is
already reprojected/aligned onto ref_path's grid at build time by
VarBuilder._to_reference, so no extra alignment work is needed here).
Static features (elevation, slope, aspect, d_roads, d_rivers,
d_buildings, landuse_class) don't vary by season, so they're rendered
once into a season-less "static" folder rather than duplicated per
season. Seasonal features (climate vars, diurnal_range, ndvi, d_fires)
use their *test*-split raster, matching what prepare_full_domain /
stage_evaluate's full-domain susceptibility raster was actually built
from -- so the two are spatially comparable side by side. landuse_class
is rendered as a single discrete-legend raster (one class per
human_fclasses entry, not one raster per one-hot dummy).

    stage_eda(config, input_paths) -> {
        "<season>": {
            "vif": Path, "spearman": Path, "class_balance": Path, "nan_coverage": Path,
            "spearman_significance_full": Path,
            "factor_maps": {"<feature_name>": Path, ...},        # seasonal features only
            "static_factor_maps": {"<feature_name>": Path, ...}, # same dict every season
        },
        ...
    }

`input_paths` must contain:
    "raw": {"<season>": {"train": Path, "test": Path}}     # stage_integration output
    "clean": {"<season>": {"train": Path, "test": Path}}   # stage_preprocessing output

`input_paths` may also contain (factor maps are skipped, with a warning,
if these aren't supplied or the DEM is missing on disk):
    "static": {"ref_path": Path, "elevation": Path, "boundary": Path, ...}  # stage_static output
    "seasonal": {"<season>": {"train": {...}, "test": {...}}}              # stage_seasonal output
    "labels": {"<season>": {"train": {...}, "test": {...}}}                # stage_labels output
"""
from pathlib import Path
from typing import Dict

import pandas as pd

from .. import viz
from ..utils.logger import setup_logger

# Paths in stage_static's output that aren't feature rasters.
_NON_FEATURE_STATIC_KEYS = ("ref_path", "boundary")


def _landuse_class_labels(config: dict) -> Dict[int, str]:
    """0 = no human-activity land use; 1..N = 1-indexed position in
    human_fclasses, matching features/proximity.py::_write_landuse_class's
    encoding exactly."""
    fclasses = config["data_sources"]["proximity"]["human_fclasses"]
    labels = {0: "No human activity"}
    labels.update({i + 1: name.replace("_", " ").title() for i, name in enumerate(fclasses)})
    return labels


def _render_static_factor_maps(config: dict, static_paths: dict, dem_path, figures_dir, logger) -> dict:
    feature_paths = {k: v for k, v in static_paths.items() if k not in _NON_FEATURE_STATIC_KEYS}
    if not feature_paths:
        return {}
    class_labels_by_feature = (
        {"landuse_class": _landuse_class_labels(config)} if "landuse_class" in feature_paths else {}
    )
    rendered = viz.render_all_factor_maps(
        feature_paths, dem_path, figures_dir, season=None,
        class_labels_by_feature=class_labels_by_feature,
    )
    logger.info(f"[stage_eda] Rendered {len(rendered)} static factor map(s): {sorted(rendered)}")
    return rendered


def _render_seasonal_factor_maps(season: str, seasonal_paths: dict, labels_paths: dict, dem_path, figures_dir, logger) -> dict:
    feature_paths = dict(seasonal_paths.get(season, {}).get("test", {}))
    d_fires_path = labels_paths.get(season, {}).get("test", {}).get("d_fires")
    if d_fires_path is not None:
        feature_paths["d_fires"] = d_fires_path
    if not feature_paths:
        return {}
    rendered = viz.render_all_factor_maps(feature_paths, dem_path, figures_dir, season=season)
    logger.info(f"[stage_eda] [{season}] Rendered {len(rendered)} seasonal factor map(s): {sorted(rendered)}")
    return rendered


def stage_eda(config: dict, input_paths: dict) -> Dict[str, dict]:
    """
    NOTE: class_balance compares train split only (raw vs. cleaned) --
    label cleaning (LabelCleaner.clean_flat) only ever runs on train per
    the documented no-test-leakage principle (see modeling/dataset_prep.py
    docstring), so a train-vs-clean comparison is the only one that's
    meaningful; test labels are never mutated and would show 0% change.
    """
    # logger = setup_logger(log_file=config["logging"]["log_path"], level=config["logging"]["level"])
    logger = setup_logger()
    figures_dir = config["base"]["figures_dir"]

    raw = input_paths["raw"]
    clean = input_paths["clean"]
    static_paths = input_paths.get("static", {})
    seasonal_paths = input_paths.get("seasonal", {})
    labels_paths = input_paths.get("labels", {})

    dem_path = static_paths.get("elevation")
    dem_path = Path(dem_path) if dem_path else None
    static_factor_maps: Dict[str, object] = {}
    if dem_path is None or not dem_path.exists():
        logger.warning(
            f"[stage_eda] No DEM found (static.elevation={dem_path}); skipping feature raster maps."
        )
    else:
        static_factor_maps = _render_static_factor_maps(config, static_paths, dem_path, figures_dir, logger)

    out: Dict[str, dict] = {}
    for season in raw:
        logger.info(f"[stage_eda] [{season}] Generating EDA figures...")
        df_raw = pd.read_csv(raw[season]["train"])
        df_clean = pd.read_csv(clean[season]["train"])

        feature_cols = [c for c in df_clean.columns if c not in ("label", "_x", "_y", "tas", "tasmin")]

        nan_arrays = {c: df_raw[c].to_numpy() for c in df_raw.columns if c != "label"}
        nan_path = viz.plot_nan_coverage(nan_arrays, figures_dir, season=season)

        vif_paths = viz.plot_vif_correlation(df_clean, feature_cols, figures_dir, season=season)

        spearman_sig = viz.compute_spearman_significance_export(df_clean, figures_dir, season=season)
        logger.info(
            f"[stage_eda] [{season}] Spearman significance export: "
            f"{spearman_sig['n_cols']} columns, {spearman_sig['n_rows']} pairs -> {spearman_sig['out_path']}"
        )

        # Class balance: raw label column pre-cleaning vs clean label column
        # post-cleaning. plot_class_balance's internal counting is shape-
        # agnostic, so flat 1-D arrays work the same as the old raster inputs.
        class_balance_path = viz.plot_class_balance(
            df_raw["label"].to_numpy(), df_clean["label"].to_numpy(), figures_dir, season=season
        )

        factor_maps: Dict[str, object] = {}
        if dem_path is not None and dem_path.exists():
            factor_maps = _render_seasonal_factor_maps(
                season, seasonal_paths, labels_paths, dem_path, figures_dir, logger
            )

        out[season] = {
            "nan_coverage": nan_path,
            "vif": vif_paths["vif"],
            "spearman": vif_paths["spearman"],
            "class_balance": class_balance_path,
            "spearman_significance_full": spearman_sig["out_path"],
            "factor_maps": factor_maps,
            "static_factor_maps": static_factor_maps,
        }

    return out
