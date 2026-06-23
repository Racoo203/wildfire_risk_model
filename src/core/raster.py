import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.mask import mask
from pathlib import Path
from typing import Union, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class RasterManager:
    """Handles raster I/O, resampling, and alignment."""

    @staticmethod
    def read(
        path: Union[str, Path], 
        band: int = 1
    ) -> Tuple[np.ndarray, dict]:
        """
        Read a single band from a raster.

        Returns: 
            data: numpy array
            metadata: dict with crs, bounds, transforms, nodata, etc.
        """

        path = Path(path)

        with rasterio.open(path) as src:
            data = src.read(band).astype("float32")
            meta = src.meta.copy()

        return data, meta

    @staticmethod
    def write(
        data: np.ndarray, 
        path: Union[str, Path], 
        meta: dict, 
        compress: str = "deflate",
    ) -> None:
        """
        Write a single-band raster to disk.
        """

        path = Path(path)
        path.parent.mkdir(parents = True, exist_ok = True)

        meta.update({
            "dtype": data.dtype,
            "height": data.shape[0],
            "width": data.shape[1],
            "count": 1,
            "compress": compress,
            "nodata": np.nan
        })

        with rasterio.open(path, "w", **meta) as dst:
            dst.write(data[np.newaxis, :, :])

        logger.info(f"Written {path}")

    @staticmethod
    def reproject_and_resample(
        src_path: Union[str, Path], 
        dst_path: Union[str, Path],
        target_crs: str = "ESPG:27700",
        target_res: float = 30.0,
        resampling_method: Resampling = Resampling.bilinear,
    ) -> None:
        """
        Reproject raster to target CRS and resample to target resolution.
        """
        src_path = Path(src_path)
        dst_path = Path(dst_path)
        dst_path.parent.mkdir(parents = True, exist_ok = True)

        with rasterio.open(src_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, 
                src.height, *src.bounds, resolution = target_res
            )
            
            meta = src.meta.copy()
            meta.update({
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height,
                "dtype": "float32"
            })

            with rasterio.open(dst_path, "w", **meta) as dst:
                reproject(
                    source = rasterio.band(src, 1),
                    destination = rasterio.band(dst, 1),
                    src_transform = src.transform,
                    src_crs = src.crs,
                    dst_transform = transform,
                    dst_crs = target_crs,
                    resampling = resampling_method,
                )
    
        logger.info(f"Reprojected {src_path} to {dst_path}")

    @staticmethod
    def align_to_reference(
        src_path: Union[str, Path], 
        ref_path: Union[str, Path],
        dst_path: Union[str, Path],
    ):
        """
        Resample and reproject src to exactly match ref grid.
        """
        src_path = Path(src_path)
        dst_path = Path(dst_path)
        dst_path.parent.mkdir(parents = True, exist_ok = True)
        
        with rasterio.open(ref_path) as ref:
            ref_meta = ref.meta.copy()
            ref_transform = ref.transform
            ref_crs = ref.crs
            ref_width = ref.width
            ref_height = ref.height

        with rasterio.open(src_path) as src:
            data = np.empty((1, ref_height, ref_width), dtype = "float32")
            reproject(
                source = rasterio.band(src, 1),
                destination = data,
                src_transform = src.transform,
                src_crs = src.crs,
                dst_transform = ref_transform,
                dst_crs = ref_crs,
                resampling = Resampling.bilinear,
            )
        
        ref_meta.update({"dtype": "float32", "nodata": np.nan})

        with rasterio.open(dst_path, "w", **ref_meta) as dst:
            dst.write(data)
        
        logger.info(f"Aligned {src_path} to reference, to {dst_path}")

    @staticmethod
    def clip_to_boundary(
        raster_path: Union[str, Path],
        boundary_gdf,
        out_path: Union[str, Path]
    ):
        """
        Clip raster to vector boundary.
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents = True, exist_ok = True)

        geoms = [geom.__geo_interface__ for geom in boundary_gdf.geometry]
        
        with rasterio.open(raster_path) as src:
            out_image, out_transform = mask(src, geoms, crop = True, nodata = np.nan)
            meta = src.meta.copy()
            meta.update({
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "dtype": "float32",
                "nodata": np.nan,
            })

            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(out_image.astype("float32"))

        logger.info(f"Clipped {raster_path} to {out_path}")
        

    @staticmethod
    def stack_to_dataframe(
        raster_paths: dict,
        reference_raster: Union[str, Path],
        output_csv: Union[str, Path],
    ) -> "pd.DataFrame":
        """
        Stack multiple rasters into a pandas DataFrame.
        One row per valid pixel.
        """

        import pandas as pd

        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents = True, exist_ok = True)

        # Use reference to define valid pixel mask
        with rasterio.open(reference_raster) as ref:
            mask_data = ref.read(1)
        valid_mask = ~np.isnan(mask_data)

        data = {}

        for name, path in raster_paths.items():
            with rasterio.open(path) as src:
                arr = src.read(1)
            data[name] = arr[valid_mask].astype("float32")
            logger.info(f"Stacked {name}: {data[name].shape[0]:,} valid pixels")

        df = pd.DataFrame(data)
        df.to_csv(output_csv, index = False)
        logger.info(f"Dataset saved to {output_csv}")

        return df