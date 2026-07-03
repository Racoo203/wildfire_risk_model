import json
import geopandas as gpd
import pandas as pd
from pathlib import Path
from typing import Union, Any, Dict
import logging

logger = logging.getLogger(__name__)

def read_geojson(path: Union[str, Path]) -> Dict[str, Any]:
    """Read GeoJSON file into dict"""
    with open(path) as f:
        return json.load(f)

def write_geojson(path: Union[str, Path]) -> None:
    """Write dict as GeoJSON file."""
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)  
    with open(path, "w") as f:
        json.dump(data, f, indent = 2)
    logger.info(f"Written {path}")

def read_csv(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read CSV file with error handling"""
    try:
        df = pd.read_csv(path, **kwargs)
        logger.info(f"Read {path}: {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        raise

def write_csv(df: pd.DataFrame, path: Union[str, Path], index: bool = False, **kwargs) -> None:
    """Write DataFrame to CSV with error handling."""
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)  

    try:
        df.to_csv(path, index = index, **kwargs)
        logger.info(f"Written {path}: {len(df):,} rows")
    except Exception as e:
        logger.error(f"Failed to write {path}: {e}")
        raise

def read_shapefile(path: Union[str, Path], **kwargs) -> gpd.GeoDataFrame:
    """Read shapefile with error handling."""
    try:
        gdf = gpd.read_file(path, **kwargs)
        logger.info(f"Read {path}: {len(gdf)} features, CRS= {gdf.crs}")
        return gdf
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        raise

def write_shapefile(
    gdf: gpd.GeoDataFrame, 
    path: Union[str, Path], 
    driver: str = "ESRI Shapefile", 
    **kwargs
) -> None:
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)

    try:
        gdf.to_file(path, driver = driver, **kwargs)
        logger.info(f"Written {path}: {len(df):,} {len(gdf)} features")
    except Exception as e:
        logger.error(f"Failed to write {path}: {e}")
        raise
