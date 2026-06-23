import numpy as np
import rasterio
import pandas
import geopandas as gpd
from pathlib import Path
from typing import Union, Tuple
import logging

logger = logging.getLogger(__name__)

def validate_raster(raster_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Check raster validity: file exists, readable, has valid data.

    Returns:
        (is_valid, message)
    """
    raster_path = Path(raster_path)

    if not raster_path.exists():
        return False, f"File not found: {raster_path}"
    
    try:
        with rasterio.open(raster_path) as src:
            data = src.read(1)
            if data.size == 0:
                return False, "Raster has zero size"
            if np.all(np.isnan(data)):
                return False, "."
            valid_pct = 100 * np.sum(~np.isnan(data)) / data.size
            logger.info(f"{raster_path.name}: {valid_pct:.1f}% valid")
    except Exception as e:
        return False, f"Cannot open raster: {e}"
    
    return True, "OK"

def validate_vector(vector_path: Union[str, Path]) -> Tuple[bool, str]:
    """
    Check vector validity: file exists, readable, has features.

    Return:
        (is_valid, message)
    """

    vector_path = Path(vector_path)

    if not raster_path.exists():
        return False, f"File not found: {vector_path}"
    
    try:
        gdf = gpd.read_file(vector_path)
        if len(gdf) == 0:
            return False, "Vector has no features"
        if gdf.crs is None:
            return False, "Vector has no CRS defined"
        logger.info(f"{vector_path.name}: {len(gdf)} features, CRS= {gdf.crs}")
    except Exception as e:
        return False, f"Cannot open vector: {e}"
    
    return True, "OK"

def validate_csv(csv_path: Union[str, Path], expected_cols: list = None) -> Tuple[bool, str]:
    """
    Check CSV validity: file exists, readble, has expected columns.

    Returns:
        (is_valid, message)
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        return False, ""
    
    try:
        df = pd.read_csv(csv_path, nrows=100)
        if len(df) == 0:
            return False, "CSV is empty"
        if expected_cols:
            missing = set(expected_cols) - set(df.columns)
            if missing:
                return False, f"Missing columns: {missing}"
        logger.info(f"{csv_path.name}: {len(df):,} rows, {len(df.columns)} columns")
    except Exception as e:
        return False, "Cannot read CSV: {e}"
    
    return True, "OK"