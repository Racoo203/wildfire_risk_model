"""Shared pytest fixtures: small synthetic rasters/vectors built in-memory,
so tests run fast and deterministically without the multi-GB bronze layer."""

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import Point, Polygon

CRS = "EPSG:27700"
CELL_SIZE = 30.0
GRID_SIZE = 10  # 10x10 synthetic grid

@pytest.fixture
def reference_transform():
    """Affine transform for a 10x10, 30m-cell grid with origin at (0, 100_000)."""
    return from_origin(0, 100_000, CELL_SIZE, CELL_SIZE)


@pytest.fixture
def synthetic_dem(tmp_path, reference_transform):
    """A small synthetic DEM raster with a smooth elevation gradient (no NaNs)."""
    path = tmp_path / "dem.tif"
    rng = np.random.default_rng(42)
    elevation = (
        50 + 5 * np.arange(GRID_SIZE)[:, None] + rng.normal(0, 0.5, (GRID_SIZE, GRID_SIZE))
    ).astype("float32")

    meta = {
        "driver": "GTiff",
        "height": GRID_SIZE,
        "width": GRID_SIZE,
        "count": 1,
        "dtype": "float32",
        "crs": CRS,
        "transform": reference_transform,
        "nodata": np.nan,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(elevation[np.newaxis, :, :])
    return path


@pytest.fixture
def synthetic_reference_raster(tmp_path, reference_transform):
    """A reference raster acting as a land mask: all valid except one NaN corner."""
    path = tmp_path / "reference.tif"
    data = np.ones((GRID_SIZE, GRID_SIZE), dtype="float32")
    data[0, 0] = np.nan  # simulate a non-land / out-of-boundary pixel

    meta = {
        "driver": "GTiff",
        "height": GRID_SIZE,
        "width": GRID_SIZE,
        "count": 1,
        "dtype": "float32",
        "crs": CRS,
        "transform": reference_transform,
        "nodata": np.nan,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data[np.newaxis, :, :])
    return path


@pytest.fixture
def synthetic_boundary(tmp_path):
    """A simple square boundary polygon covering most of the synthetic grid."""
    path = tmp_path / "boundary.shp"
    poly = Polygon([(0, 100_000 - GRID_SIZE * CELL_SIZE), (GRID_SIZE * CELL_SIZE, 100_000 - GRID_SIZE * CELL_SIZE),
                     (GRID_SIZE * CELL_SIZE, 100_000), (0, 100_000)])
    gdf = gpd.GeoDataFrame({"CTYUA24NM": ["Essex"]}, geometry=[poly], crs=CRS)
    gdf.to_file(path)
    return path


@pytest.fixture
def synthetic_fire_points(tmp_path):
    """A handful of fire point locations scattered across the synthetic grid,
    with year/month attributes spanning a train/val/test-style split."""
    rng = np.random.default_rng(7)
    n = 12
    xs = rng.uniform(0, GRID_SIZE * CELL_SIZE, n)
    ys = rng.uniform(100_000 - GRID_SIZE * CELL_SIZE, 100_000, n)
    years = rng.integers(2009, 2025, n)
    months = rng.integers(1, 13, n)

    gdf = gpd.GeoDataFrame(
        {"year": years, "month": months},
        geometry=[Point(x, y) for x, y in zip(xs, ys)],
        crs=CRS,
    )
    return gdf


@pytest.fixture
def minimal_config(tmp_path):
    """A minimal config dict sufficient to instantiate VarBuilder subclasses in tests."""
    return {
        "base": {
            "boundary_shapefile": str(tmp_path / "boundary.shp"),
            "output_dir": str(tmp_path / "output"),
            "figures_dir": str(tmp_path / "figures"),
        },
        "processing": {
            "crs": CRS,
            "force_recompute": False,
            "training_years": [2009, 2022],
            "test_years": [2022, 2026],
        },
        "logging": {
            "level": "INFO",
            "log_path": str(tmp_path / "logs" / "pipeline.log"),
        },
    }

@pytest.fixture
def minimal_modeling_config(tmp_path):
    """A config dict with every ModelingConfig field the schema defaults —
    built by dumping WildfireConfig's own defaults, not hand-typed, so it
    can never drift from what ConfigLoader actually produces."""
    from wildfire_susceptibility.config.schema import WildfireConfig
    cfg = WildfireConfig().model_dump(mode="python")
    cfg["base"]["figures_dir"] = str(tmp_path / "figures")
    cfg["base"]["output_dir"] = str(tmp_path / "layers")
    return cfg

@pytest.fixture
def fast_modeling_config(minimal_modeling_config):
    """minimal_modeling_config with fast-test overrides layered on top."""
    cfg = minimal_modeling_config
    cfg["modeling"].update({
        "optuna_n_trials": 1,
        "cv_folds": 2,
        "use_smote": False,
    })
    return cfg