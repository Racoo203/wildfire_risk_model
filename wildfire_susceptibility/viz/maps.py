"""Factor maps and susceptibility maps — all rendered via terrain_overlay
and logged to figures/manifest.json (Section 9)."""

from pathlib import Path
from typing import Dict, Optional

from .terrain_overlay import save_terrain_map
from .manifest import append_to_manifest

_FACTOR_CMAPS = {
    "elevation": "terrain",
    "slope": "YlOrRd",
    "aspect": "twilight",
    "ndvi": "YlGn",
    "d_roads": "Blues_r",
    "d_rivers": "Blues_r",
    "d_activity": "Blues_r",
    "d_fires": "Reds_r",
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
    """Render one factor raster (e.g. elevation, ndvi, d_fires) with terrain backdrop."""
    figures_dir = Path(figures_dir)
    cmap = _FACTOR_CMAPS.get(factor_name, "viridis")
    fname = f"{factor_name}_{season}.png" if season else f"{factor_name}.png"
    out_path = figures_dir / out_subdir / fname

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
    season: Optional[str] = None,
    model_name: Optional[str] = None,
    include_d_fires_as_feature: Optional[bool] = None,
) -> Path:
    """Render a 4-class susceptibility raster (Low/Medium/High/Very High)."""
    figures_dir = Path(figures_dir)
    parts = ["susceptibility"]
    if season:
        parts.append(season)
    if model_name:
        parts.append(model_name)
    fname = "_".join(parts) + ".png"
    out_path = figures_dir / "susceptibility" / fname

    title = "Wildfire Susceptibility"
    if season:
        title += f" — {season}"
    if model_name:
        title += f" ({model_name})"

    save_terrain_map(
        labels_path, dem_path, out_path,
        cmap=_SUSCEPTIBILITY_CMAP,
        title=title,
        colorbar_label="Class (0=Low .. 3=Very High)",
    )

    append_to_manifest(
        figures_dir, out_path,
        category="susceptibility_map",
        generated_by="viz.maps.render_susceptibility_map",
        season=season,
        params={"model": model_name, "include_d_fires_as_feature": include_d_fires_as_feature},
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