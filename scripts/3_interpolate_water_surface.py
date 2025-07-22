from osgeo import gdal, ogr
import math
import numpy as np
import os
import glob
import rasterio
from rasterio.merge import merge
from rasterio.enums import Resampling
import sys

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
    showing live progress in the terminal.
    """
    ds = ogr.Open(gpkg_path)
    if ds is None:
        raise RuntimeError(f"Cannot open GeoPackage: {gpkg_path}")
    layer = ds.GetLayer(0)

    xmin, xmax, ymin, ymax = layer.GetExtent()
    x_res = math.ceil((xmax - xmin) / pix_size)
    y_res = math.ceil((ymax - ymin) / pix_size)
    
    # Build GridOptions as before
    if radius is None:
        alg = f"invdist:power={power}:smoothing={smoothing}:nodata=0"
    else:
        alg = f"invdist:power={power}:smoothing={smoothing}:radius1={radius}:radius2={radius}:nodata=0"

    grid_opts = gdal.GridOptions(
        format="GTiff",
        outputType=gdal.GDT_Float32,
        width=int(x_res),
        height=int(y_res),
        outputBounds=(xmin, ymin, xmax, ymax),
        zfield=field,
        algorithm=alg
    )

    # Custom progress callback
    def progress(complete, message, _):
        """
        complete: float in [0.0, 1.0]
        message: optional string from GDAL (often empty)
        Return 1 to continue, 0 to abort.
        """
        pct = complete * 100
        # \r returns cursor to start of line; end='' prevents newline
        sys.stdout.write(f"\r[Interpolation] {pct:6.2f}% {message}")
        sys.stdout.flush()
        return 1

    # Run the interpolation with our callback
    gdal.Grid(
        destName=out_path,
        srcDS=gpkg_path,
        options=grid_opts,
        callback=progress
    )

    # Finish with a newline so the prompt isn’t stuck on the same line
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

    # Compute difference, preserving mask
    diff = np.ma.subtract(arr1, arr2)

    # override the driver before writing
    profile.update(
        driver='GTiff',              
        dtype=diff.dtype,
        nodata=profile.get('nodata', None)
    )



    # Write output
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(diff.filled(profile.get('nodata', 0)), 1)

if __name__ == "__main__":
    
    min_points_gpkg = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\REM\Voronoi Method\low coverage manual\min_elev_points_10m.gpkg"
    output_dir = os.path.dirname(min_points_gpkg)
    gpkg_dir = os.path.join(output_dir, "clusters")
    
    # ──────────────── Configuration ────────────────────
    # Name of the attribute field holding elevation values
    elevation_field = "elevation"
    # Raster pixel size (in the same units as your GeoPackage CRS)
    pixel_size     = 2.0
    # IDW parameters
    idw_power      = 2.0   # power parameter (controls distance weighting) higher = more localized influence
    idw_smoothing  = 1.0   # smoothing parameter (reduces bull’s-eye effect) greater than 1 = more smoothing
    # Set to half the max valley width in the network
    idw_radius     = 500   # search radius for IDW interpolation

    # ────────────────────────────────────────────────────
    
        
    output_WS_raster = os.path.join(output_dir, "min_points_interpolated_radius_bathy.tif")
    interpolate_water_surface(
        gpkg_path    = min_points_gpkg,
        out_path     = output_WS_raster,
        field        = elevation_field,
        pix_size     = pixel_size,
        power        = idw_power,
        smoothing    = idw_smoothing,
        radius       = idw_radius  
    )
    
    # os.makedirs(gpkg_dir, exist_ok=True)
    # # Create a new gpkg for each unique cluster_id
    # import geopandas as gpd
    # min_points_gdf = gpd.read_file(min_points_gpkg)
    # clusters = min_points_gdf["cluster_id"].unique()
    # for cluster in clusters:    
    #     cluster_gdf = min_points_gdf[min_points_gdf["cluster_id"] == cluster]
    #     cluster_gpkg = os.path.join(gpkg_dir, f"cluster_{cluster}.gpkg")
    #     cluster_gdf.to_file(cluster_gpkg, driver="GPKG")
    #     print(f"[✔] Created GeoPackage for cluster {cluster}: {cluster_gpkg}")
    
    # gpkg_list = [f for f in os.listdir(gpkg_dir) if f.endswith('.gpkg')]
    # # Interpolate point elevations
    # for gpkg_file in gpkg_list:
    #     output_WS_raster = os.path.join(gpkg_dir, f"{gpkg_file[:-5]}.tif")
    #     gpkg_file = os.path.join(gpkg_dir, gpkg_file)

    #     interpolate_water_surface(
    #         gpkg_path    = gpkg_file,
    #         out_path     = output_WS_raster,
    #         field        = elevation_field,
    #         pix_size     = pixel_size,
    #         power        = idw_power,
    #         smoothing    = idw_smoothing    
    #     )

    # interpolated_min_raster = os.path.join(output_dir, "interpolated_bathy.tif")
    # merge_tifs(gpkg_dir, interpolated_min_raster)
    
    # diff_output = os.path.join(output_dir, "REM_bathy-WSE_interpolated.tif")
    # difference_rasters(REM_reference_raster, output_WS_raster, diff_output)
    
