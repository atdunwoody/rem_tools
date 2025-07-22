import rasterio
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

def smooth_raster(
    input_raster: str,
    output_raster: str,
    method: str = "gaussian",
    sigma: float = 1.0,
    window: int = 3
) -> None:
    """
    Apply a focal‐filter to blur sharp breaks.
    
    Parameters
    ----------
    input_raster : path to your blocky raster (e.g. GeoTIFF or raster‐in‐GeoPackage)
    output_raster : path to write the smoothed result
    method : 'gaussian' or 'mean'
    sigma : standard deviation for Gaussian (if method == 'gaussian')
    window : neighbourhood size for mean filter (if method == 'mean')
    """
    print(f"Applying {method} smoothing to {input_raster}...")
    print(f"Sigma: {sigma}, Window size: {window}")
    with rasterio.open(input_raster) as src:
        arr = src.read(1, masked=True)
        profile = src.profile

    if method == "gaussian":
        # gaussian_filter requires a filled array
        filled = arr.filled(arr.fill_value)
        smooth = gaussian_filter(filled, sigma=sigma)
    else:
        # uniform_filter will average over a square window
        filled = arr.filled(arr.fill_value)
        smooth = uniform_filter(filled, size=window)

    # preserve mask (if you need nodata)
    out = np.where(arr.mask, profile["nodata"], smooth)

    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(out.astype(profile["dtype"]), 1)


smooth_raster(
    input_raster=r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\REM\Voronoi Method\REM_bathy-WSE.tif",
    output_raster=r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\REM\Voronoi Method\REM_bathy-WSE_smoothed.tif",
    method="gaussian",
    sigma=1.0
)