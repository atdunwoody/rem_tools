from osgeo import gdal, ogr
import math
import numpy as np
import os
import glob
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling


def interpolate_water_surface(
    gpkg_path: str,
    out_path: str,
    field: str,
    pix_size: float,
    power: float,
    smoothing: float):
    """
    Uses GDAL Grid (IDW) to interpolate point elevations to a TIFF.
    """
    ds = ogr.Open(gpkg_path)
    if ds is None:
        raise RuntimeError(f"Cannot open GeoPackage: {gpkg_path}")
    layer = ds.GetLayer(0)

    xmin, xmax, ymin, ymax = layer.GetExtent()
    x_res = math.ceil((xmax - xmin) / pix_size)
    y_res = math.ceil((ymax - ymin) / pix_size)

    grid_opts = gdal.GridOptions(
        format="GTiff",
        outputType=gdal.GDT_Float32,
        width=int(x_res),
        height=int(y_res),
        outputBounds=(xmin, ymin, xmax, ymax),
        zfield=field,
        algorithm=f"invdist:power={power}:smoothing={smoothing}"
    )

    gdal.Grid(
        destName=out_path,
        srcDS=gpkg_path,
        options=grid_opts
    )

    print(f"[✔] Interpolation complete: {os.path.basename(out_path)}")

def difference_rasters(raster_path1: str, raster_path2: str, output_path: str):
    """
    Compute the difference (raster1 - raster2) and write to a new raster.
    """
    with rasterio.open(raster_path1) as src1:
        arr1 = src1.read(1, masked=True)
        profile = src1.profile.copy()

    with rasterio.open(raster_path2) as src2:
        if (src2.width, src2.height) != (profile['width'], profile['height']) or src2.transform != profile['transform']:
            arr2 = src2.read(
                1,
                out_shape=(src2.count, profile['height'], profile['width']),
                resampling=Resampling.bilinear
            )[0]
        else:
            arr2 = src2.read(1, masked=True)

    diff = np.ma.subtract(arr1, arr2)

    profile.update(
        driver='GTiff',              
        dtype=diff.dtype,
        nodata=profile.get('nodata', 0)
    )

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(diff.filled(profile.get('nodata', 0)), 1)

    print(f"[✔] Difference raster written: {os.path.basename(output_path)}")


if __name__ == "__main__":
    # ─────────────── Configuration ────────────────────
    # gpkg_file = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\REM\IDW Sensitivity\min_elev_points_100m_BF.gpkg"
    # output_dir = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\REM\IDW Sensitivity"
    # ref_raster = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\REM\IDW Sensitivity\bathymetry_cluster_9.tif"


    gpkg_file = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM\2 meter\min_elev_points_100m_BF.gpkg"
    output_dir = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\REM\IDW Sensitivity Full"
    ref_raster = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\LiDAR\grmw_rasters\bathymetry.tif"
    elevation_field = "elevation"
    pixel_size     = 2.0

    # Sensitivity analysis parameters
    idw_powers     = [1.0, 2.0, 3.0]
    idw_smoothings = [0.5, 1.0, 5.0]

    # Create output subdirectories
    ws_dir   = os.path.join(output_dir, "ws_rasters")
    diff_dir = os.path.join(output_dir, "diff_rasters")
    os.makedirs(ws_dir, exist_ok=True)
    os.makedirs(diff_dir, exist_ok=True)

    # Loop through parameter combinations
    for power in idw_powers:
        for smoothing in idw_smoothings:
            print(f"Processing power={power}, smoothing={smoothing}")
            # Define file names
            ws_file   = os.path.join(ws_dir, f"ws_p{power}_s{smoothing}.tif")
            diff_file = os.path.join(diff_dir, f"diff_p{power}_s{smoothing}.tif")

            # Interpolate
            interpolate_water_surface(
                gpkg_path=gpkg_file,
                out_path=ws_file,
                field=elevation_field,
                pix_size=pixel_size,
                power=power,
                smoothing=smoothing
            )

            # # Compute difference to reference
            # difference_rasters(
            #     raster_path1=ref_raster,
            #     raster_path2=ws_file,
            #     output_path=diff_file
            # )

    print("Sensitivity analysis complete.")
