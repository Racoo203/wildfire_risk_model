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
    "rainfall": "Blues",
    "hurs": "Greens",
    "sfcWind": "Purples",
    "d_roads": "Blues_r",
    "d_rivers": "Blues_r",
    "d_activity": "Blues_r",
    "d_fires": "Reds_r",
    "d_buildings": "Blues_r",
    "land_use": "tab10",
}

_SUSCEPTIBILITY_CMAP = "RdYlGn_r"  # low (green) -> very high (red)


def render_factor_map(
    factor_name: str,
    data_path: Path,
    dem_path: Path,
    figures_dir: Path,
    season: Optional[str] = None,
    out_subdir: str = "factors",
) -> Path:
    figures_dir = Path(figures_dir)
    cmap = _FACTOR_CMAPS.get(factor_name, "viridis")
    out_path = figures_dir / (season or "static") / out_subdir / f"{factor_name}.png"

    save_terrain_map(
        data_path, dem_path, out_path,
        cmap=cmap,
        title=factor_name.replace("_", " ").title() + (f" — {season}" if season else ""),
        colorbar_label=factor_name,
    )

    append_to_manifest(
        figures_dir, out_path,
        category="factor_map",
        generated_by="viz.maps.render_factor_map",
        season=season,
        params={"factor": factor_name, "cmap": cmap},
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
    include_d_fires_as_feature: Optional[bool] = None,
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
            "include_d_fires_as_feature": include_d_fires_as_feature,
            "fire_points_overlaid": fire_points_gdf is not None,
        },
    )
    return out_path

def render_all_factor_maps(
    feature_paths: Dict[str, Path],
    dem_path: Path,
    figures_dir: Path,
    season: Optional[str] = None,
) -> Dict[str, Path]:
    """Render every feature in `feature_paths` (as returned by the feature
    builders / WildfirePreprocessor) as a terrain-backdrop map."""
    return {
        name: render_factor_map(name, path, dem_path, figures_dir, season=season)
        for name, path in feature_paths.items()
    }