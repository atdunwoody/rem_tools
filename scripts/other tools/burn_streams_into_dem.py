from pathlib import Path
from typing import Union
import geopandas as gpd
import rasterio
import rasterio.features
import numpy as np

def burn_streams_into_dem(
    streams_vector: Union[str, Path],
    dem_path: Union[str, Path],
    out_path: Union[str, Path],
    *,
    stream_layer: str | None = None,
    burn_depth: float = 1.0,
) -> str:
    """
    Burn a stream network into a DEM by lowering stream-adjacent cells.

    Parameters
    ----------
    streams_vector : str | Path
        Path to the input vector (GeoPackage/Shapefile) of streams.
    dem_path : str | Path
        Path to the input DEM raster.
    out_path : str | Path
        Path to save the burned DEM raster.
    stream_layer : str, optional
        Layer name (required if streams_vector is a GeoPackage).
    burn_depth : float, default=1.0
        Depth (in DEM units, typically meters) to lower stream cells.

    Returns
    -------
    str
        Path to the burned DEM.
    """
    streams_gdf = gpd.read_file(streams_vector, layer=stream_layer)

    # Open DEM
    with rasterio.open(dem_path) as src:
        dem = src.read(1, masked=True)
        meta = src.meta.copy()

        # Ensure CRS alignment
        if streams_gdf.crs != src.crs:
            streams_gdf = streams_gdf.to_crs(src.crs)

        # Rasterize streams onto DEM grid
        stream_raster = rasterio.features.rasterize(
            ((geom, 1) for geom in streams_gdf.geometry if geom is not None),
            out_shape=dem.shape,
            transform=src.transform,
            fill=0,
            dtype="uint8",
        )

        # Burn DEM (subtract burn_depth where stream pixels exist)
        burned_dem = dem.copy()
        burned_dem[stream_raster == 1] = burned_dem[stream_raster == 1] - burn_depth

    # Save burned DEM
    meta.update(dtype="float32", nodata=src.nodata)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(burned_dem.astype("float32"), 1)

    return str(out_path)


streams_vector = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\Web App Processing\FLIR Data\streams_to_burn_in.shp"
dem_path = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\LiDAR\rasters_USGS10m\USGS 10m DEM Clip.tif"
out_path = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\Web App Processing\FLIR Data\burned_dem.tif"

burned_dem_path = burn_streams_into_dem(
    streams_vector=streams_vector,
    dem_path=dem_path,
    out_path=out_path,
    burn_depth=5.0
)
