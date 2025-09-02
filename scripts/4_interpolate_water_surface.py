import math
import numpy as np
import os
import glob
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
import sys
from osgeo import gdal, ogr

import math
import sys
from osgeo import ogr, gdal

def interpolate_water_surface(
    gpkg_path: str,
    out_path: str,
    field: str,
    pix_size: float,
    power: float,
    smoothing: float,
    radius: float = None
) -> None:
    """
    Uses GDAL Grid (IDW) to interpolate point elevations to a TIFF,
    dropping any points where `field` is NULL.
    """
    print(f"Interpolating water surface from {gpkg_path} to {out_path}...")
    print(f"Using field '{field}', pixel size {pix_size}m, power {power}, "
          f"smoothing {smoothing}, radius {radius}")

    # open vector and get layer
    ds = ogr.Open(gpkg_path)
    if ds is None:
        raise RuntimeError(f"Cannot open GeoPackage: {gpkg_path}")
    layer = ds.GetLayer(0)

    # drop features with NULL in our field
    layer.SetAttributeFilter(f"{field} IS NOT NULL")

    # compute extent on filtered layer
    xmin, xmax, ymin, ymax = layer.GetExtent()
    x_res = math.ceil((xmax - xmin) / pix_size)
    y_res = math.ceil((ymax - ymin) / pix_size)

    # build the IDW algorithm string
    if radius is None:
        alg = f"invdist:power={power}:smoothing={smoothing}:nodata=0"
    else:
        alg = (
            f"invdist:power={power}:smoothing={smoothing}:"
            f"radius1={radius}:radius2={radius}:nodata=0"
        )

    # include a WHERE clause so GDAL only reads non-null features
    grid_opts = gdal.GridOptions(
        format="GTiff",
        outputType=gdal.GDT_Float32,
        width=int(x_res),
        height=int(y_res),
        outputBounds=(xmin, ymin, xmax, ymax),
        zfield=field,
        algorithm=alg,
        where=f"{field} IS NOT NULL"
    )

    # progress callback
    def progress(complete, message, _):
        pct = complete * 100
        sys.stdout.write(f"\r[Interpolation] {pct:6.2f}% {message}")
        sys.stdout.flush()
        return 1

    # run the grid
    gdal.Grid(
        destName=out_path,
        srcDS=gpkg_path,
        options=grid_opts,
        callback=progress
    )

    # newline after progress bar
    print()
    print(f"[✔] Interpolation complete. Output raster: {out_path}")


def merge_tifs(input_folder: str, output_path: str) -> None:
    """
    Merge all .tif files in input_folder into a single GeoTIFF at output_path.

    Parameters
    ----------
    input_folder : str
        Path to the folder containing input .tif files.
    output_path : str
        Path (including filename) where the merged GeoTIFF will be written.
    """
    print(f"Merging .tif files from \n{input_folder} \ninto \n{output_path}...")
    # Find all .tif files
    pattern = os.path.join(input_folder, "*.tif")
    tif_files = glob.glob(pattern)
    if not tif_files:
        raise FileNotFoundError(f"No .tif files found in {input_folder!r}")

    # FInd no data value from first file
    with rasterio.open(tif_files[0]) as src:
        no_data_value = src.nodata
        if no_data_value is None:
            no_data_value = 0
    
    # Open all rasters
    srcs = [rasterio.open(fp) for fp in tif_files]

    # Merge
    mosaic, out_transform = merge(srcs)

    # Use metadata of first source as base
    out_meta = srcs[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "nodata": no_data_value,
    })

    # Write merged raster
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Clean up
    for src in srcs:
        src.close()

def difference_rasters(raster_path1: str, raster_path2: str, output_path: str):
    """
    Compute the difference (raster1 - raster2) and write to a new raster.

    Parameters
    ----------
    raster_path1 : str
        File path to the first input raster (minuend).
    raster_path2 : str
        File path to the second input raster (subtrahend).
    output_path : str
        File path for the output difference raster.
    """
    print(f"Computing difference: \n{raster_path1} \n       - \n{raster_path2} \n       =\n{output_path}")
    # Open first raster and read data & profile
    with rasterio.open(raster_path1) as src1:
        arr1 = src1.read(1, masked=True)
        profile = src1.profile.copy()
    print(f"Raster shape: {arr1.shape}, dtype: {arr1.dtype}, nodata: {profile.get('nodata', None)}")
    # Open second raster, resample if needed, then read data
    with rasterio.open(raster_path2) as src2:
        # Ensure same shape & transform
        if (src2.width, src2.height) != (profile['width'], profile['height']) or src2.transform != profile['transform']:
            arr2 = src2.read(
                1,
                out_shape=(src2.count,
                           profile['height'],
                           profile['width']),
                resampling=Resampling.bilinear
            )[0]
        else:
            arr2 = src2.read(1, masked=True)
    print(f"Raster shapes: {arr1.shape} vs {arr2.shape}")
    # Compute difference, preserving mask
    diff = np.ma.subtract(arr1, arr2)
    print(f"Difference shape: {diff.shape}, dtype: {diff.dtype}")
    # override the driver before writing
    profile.update(
        driver='GTiff',              
        dtype=diff.dtype,
        nodata=profile.get('nodata', None)
    )



    # Write output
    with rasterio.open(output_path, 'w', **profile) as dst:
        print(f"Writing difference raster to {output_path}")
        dst.write(diff.filled(profile.get('nodata', 0)), 1)

if __name__ == "__main__":
    
    min_points_gpkg = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\REM\Voronoi Method\20250725\small_area_test\min_elev_points.gpkg"

    output_dir = os.path.dirname(min_points_gpkg)
    output_WS_raster = os.path.join(output_dir, "interpolated_WSE.tif")
    gpkg_dir = os.path.join(output_dir, "clusters")
    
    # ──────────────── Configuration ────────────────────
    # Name of the attribute field holding elevation values
    elevation_field = "elevation"  # or "BF_depth_Legg_m", "BF_depth_Beechie_m"
    #elevation_field = "BF_depth_Castro_m"
    #elevation_field = "BF_depth_Beechie_m"
    output_WS_raster = os.path.join(output_dir, "interpolated_WSE.tif")
    # Raster pixel size (in the same units as your GeoPackage CRS)
    pixel_size     = 2.0
    # IDW parameters
    idw_power      = 2.0   # power parameter (controls distance weighting) higher = more localized influence
    idw_smoothing  = 1.0   # smoothing parameter (reduces bull’s-eye effect) greater than 1 = more smoothing
    # Set to half the max valley width in the network
    idw_radius     = 750   # search radius for IDW interpolation

    # ────────────────────────────────────────────────────
    
        
    
    interpolate_water_surface(
        gpkg_path    = min_points_gpkg,
        out_path     = output_WS_raster,
        field        = elevation_field,
        pix_size     = pixel_size,
        power        = idw_power,
        smoothing    = idw_smoothing,
        radius       = idw_radius  
    )
    

    # REM_reference_raster = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\LiDAR\grmw_rasters\bare_earth\hdr.adf"
    # output_WS_raster = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\REM\Voronoi Method\combined corrected REM\min_points_interpolated_radius_WSE_corrected_ndv.tif"

    # diff_output = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\REM\Voronoi Method\combined corrected REM\REM_bathy-WSE_interpolated_radius_corrected.tif"
    # difference_rasters(REM_reference_raster, output_WS_raster, diff_output)
    
