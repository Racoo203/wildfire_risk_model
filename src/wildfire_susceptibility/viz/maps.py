"""Factor maps and susceptibility maps — all rendered via terrain_overlay
and logged to figures/manifest.json (Section 9)."""

from pathlib import Path
from typing import Dict, Optional

import geopandas as gpd

from .terrain_overlay import save_terrain_map
from .manifest import append_to_manifest
from ..modeling.class_labels import class_names_for

_FACTOR_CMAPS = {
    "elevation": "terrain",
    "slope": "YlOrRd",
    "aspect": "twilight",
    "ndvi": "YlGn",
    "tas": "RdYlBu_r",
    "tasmax": "Reds",
    "tasmin": "Blues",
    "diurnal_range": "Oranges",
    "rainfall": "Blues",
    "hurs": "Greens",
    "sfcWind": "Purples",
    "d_roads": "Blues_r",
    "d_rivers": "Blues_r",
    "d_fires": "Reds_r",
    "d_buildings": "Blues_r",
    "landuse_class": "tab10",
}

# Units shown alongside each factor's colorbar/legend label — units.md-free
# feature list, so this is the single place they're documented.
_FACTOR_UNITS = {
    "elevation": "m",
    "slope": "°",
    "aspect": "°",
    "d_roads": "km",
    "d_rivers": "km",
    "d_buildings": "km",
    "d_fires": "km",
    "tas": "°C",
    "tasmax": "°C",
    "tasmin": "°C",
    "diurnal_range": "°C",
    "rainfall": "mm",
    "hurs": "%",
    "sfcWind": "m/s",
}

_SUSCEPTIBILITY_CMAP = "RdYlGn_r"  # low (green) -> very high (red)


def render_factor_map(
    factor_name: str,
    data_path: Path,
    dem_path: Path,
    figures_dir: Path,
    season: Optional[str] = None,
    out_subdir: str = "factors",
    class_labels: Optional[Dict[int, str]] = None,
) -> Path:
    """Render one feature as a terrain-backdrop map. `class_labels` (e.g.
    {0: "No human activity", 1: "Residential", ...}) switches to a discrete,
    legend-based rendering for genuinely categorical rasters (landuse_class)
    instead of a continuous colorbar."""
    figures_dir = Path(figures_dir)
    cmap = _FACTOR_CMAPS.get(factor_name, "viridis")
    out_path = figures_dir / (season or "static") / out_subdir / f"{factor_name}.png"

    display_name = factor_name.replace("_", " ").title()
    unit = _FACTOR_UNITS.get(factor_name)
    colorbar_label = f"{display_name} ({unit})" if unit else display_name

    save_terrain_map(
        data_path, dem_path, out_path,
        cmap=cmap,
        title=display_name + (f" — {season}" if season else ""),
        colorbar_label=colorbar_label,
        discrete_legend=class_labels,
    )

    append_to_manifest(
        figures_dir, out_path,
        category="factor_map",
        generated_by="viz.maps.render_factor_map",
        season=season,
        params={"factor": factor_name, "cmap": cmap, "categorical": class_labels is not None},
    )
    return out_path


def render_susceptibility_map(
    labels_path: Path,
    dem_path: Path,
    figures_dir: Path,
    *,
    n_classes: int,
    season: Optional[str] = None,
    model_name: Optional[str] = None,
    fire_points_gdf: Optional[gpd.GeoDataFrame] = None,
) -> Path:
    figures_dir = Path(figures_dir)
    fname = f"{model_name}.png" if model_name else "default.png"
    out_path = figures_dir / (season or "static") / "susceptibility" / fname

    title = "Wildfire Susceptibility"
    if season:
        title += f" — {season}"
    if model_name:
        title += f" ({model_name})"

    # n_classes is a property of the model's fixed output space
    # (configs/labels.yaml's labels.n_classes), not of what happened to be
    # predicted in this particular raster — a raster missing a class (e.g.
    # zero "High" pixels this season) must still render with the full,
    # correctly-labeled legend rather than collapsing to fewer classes or
    # raising in class_names_for().
    class_names = class_names_for(n_classes)
    colorbar_label = f"Class (0={class_names[0]} .. {n_classes - 1}={class_names[-1]})"

    save_terrain_map(
        labels_path, dem_path, out_path,
        cmap=_SUSCEPTIBILITY_CMAP,
        title=title,
        colorbar_label=colorbar_label,
        vmin=0,
        vmax=n_classes - 1,
        points_gdf=fire_points_gdf,
        points_label="Test-period fires" if fire_points_gdf is not None else None,
    )

    append_to_manifest(
        figures_dir, out_path,
        category="susceptibility_map",
        generated_by="viz.maps.render_susceptibility_map",
        season=season,
        params={
            "model": model_name,
            "fire_points_overlaid": fire_points_gdf is not None,
        },
    )
    return out_path

def render_all_factor_maps(
    feature_paths: Dict[str, Path],
    dem_path: Path,
    figures_dir: Path,
    season: Optional[str] = None,
    class_labels_by_feature: Optional[Dict[str, Dict[int, str]]] = None,
) -> Dict[str, Path]:
    """Render every feature in `feature_paths` (as returned by the feature
    builders / WildfirePreprocessor) as a terrain-backdrop map.
    `class_labels_by_feature` (e.g. {"landuse_class": {0: "No human
    activity", ...}}) routes the named feature(s) through the discrete-
    legend rendering in render_factor_map instead of a continuous
    colorbar."""
    class_labels_by_feature = class_labels_by_feature or {}
    return {
        name: render_factor_map(
            name, path, dem_path, figures_dir, season=season,
            class_labels=class_labels_by_feature.get(name),
        )
        for name, path in feature_paths.items()
    }