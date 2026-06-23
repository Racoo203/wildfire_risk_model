import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rasterio.mask import mask
from pathlib import Path
import yaml

def plot_raster(tif_path, title, config_path = './config/config.yaml', cmap="viridis", label="", figsize=(7,5), out_name=None):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load the boundary shapefile first
    essex = gpd.read_file(config["base"]["boundary_shapefile"])

    with rasterio.open(Path(config["base"]["output_dir"]) / tif_path) as src:
        # Mask the raster using the geometry of the shapefile
        # crop=True trims the array to the extent of the shapefile
        out_image, out_transform = mask(src, essex.geometry, crop=True)
        data = out_image[0].astype("float32")
        nodata = src.nodata
        
        # Calculate new bounds based on the cropped transform
        height, width = data.shape
        left, bottom, right, top = rasterio.transform.array_bounds(height, width, out_transform)

    # Handle NoData values so they appear transparent
    if nodata is not None:
        if np.isnan(nodata):
            data[np.isnan(data)] = np.nan # Redundant but safe
        else:
            data[data == nodata] = np.nan

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        data,
        extent=[left, right, bottom, top],
        cmap=cmap,
        interpolation="nearest",
        origin="upper"
    )

    # Overlay boundary
    essex.boundary.plot(ax=ax, color="black", linewidth=1.0)

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label(label, fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Easting (m)", fontsize=8)
    ax.set_ylabel("Northing (m)", fontsize=8)

    plt.tight_layout()
    fname = out_name or (title.lower().replace(" ", "_")) + ".png"
    fig.savefig(Path(config["base"]["figures_dir"]) / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {Path(config["base"]["figures_dir"]) / fname}")