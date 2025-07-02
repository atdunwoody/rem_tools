from rasterio.enums import Resampling
import rasterio
from rasterio.merge import merge
import numpy as np
import math
from osgeo import gdal, ogr
import geopandas as gpd
import fiona
import os
import math
import glob
from rasterio.enums import Resampling
import rasterio
from rasterio.merge import merge
import numpy as np
import math
from osgeo import gdal
import geopandas as gpd
import fiona
import os
import glob
import logging

def interpolate_stream_network(
    gpkg_path: str,
    cluster_field: str,
    value_field: str,
    out_dir: str,
    pixel_size: float = 1.0,
    power: float = 2.0,
    smoothing: float = 1.0
):
    """
    For each unique cluster_id in the input GeoPackage of points,
    interpolate an IDW raster of the `value_field` over that cluster's extent.

    Parameters
    ----------
    gpkg_path : str
        Path to input point GeoPackage.
    cluster_field : str
        Name of the field holding cluster IDs.
    value_field : str
        Name of the numeric field to interpolate.
    out_dir : str
        Directory where per‐cluster rasters will be written.
    pixel_size : float
        Cell size for interpolation (same units as CRS).
    power : float
        IDW power parameter.
    smoothing : float
        IDW smoothing parameter.
    """
    # ensure output directory exists
    os.makedirs(out_dir, exist_ok=True)

    # read all points
    pts = gpd.read_file(gpkg_path)
    total_clusters = pts[cluster_field].nunique()
    for cl_id in pts[cluster_field].unique():
        print(f"Processing cluster {cl_id} of {total_clusters}")
        cluster_gpkg = os.path.join(out_dir, f"cluster_{cl_id}.gpkg")
        # Save cluster subset to a temporary GeoPackage
        pts[pts[cluster_field] == cl_id].to_file(cluster_gpkg, driver='GPKG')
        
        out_path = os.path.join(out_dir, f"interpolated_{cl_id}.tif")
        ds = ogr.Open(cluster_gpkg)
        if ds is None:
            raise RuntimeError(f"Cannot open GeoPackage: {cluster_gpkg}")
        layer = ds.GetLayer(0)

        xmin, xmax, ymin, ymax = layer.GetExtent()
        x_res = math.ceil((xmax - xmin) / pixel_size)
        y_res = math.ceil((ymax - ymin) / pixel_size)

        grid_opts = gdal.GridOptions(
            format="GTiff",
            outputType=gdal.GDT_Float32,
            width=int(x_res),
            height=int(y_res),
            outputBounds=(xmin, ymin, xmax, ymax),
            zfield=value_field,
            algorithm=f"invdist:power={power}:smoothing={smoothing}"
        )

        gdal.Grid(
            destName=out_path,
            srcDS=gpkg_path,
            options=grid_opts
        )

        print(f"[✔] Interpolation complete. Output raster: {out_path}")



def join_tifs_in_folder(input_folder: str, output_tif_path: str) -> None:
    import numpy as np
    tif_paths = glob.glob(os.path.join(input_folder, '*.tif'))
    if not tif_paths:
        raise FileNotFoundError(f"No .tif files found in {input_folder!r}")
    invalid_tifs = []
    for tif in tif_paths:
        # Check if the raster is valid by looking at min/max values
        with rasterio.open(tif) as src:
            arr = src.read(1, masked=True)
            # Check if min or max are negative or NaN
            if np.isnan(arr.min()) or np.isnan(arr.max()) or arr.min() < 0 or arr.max() < 0:
                print(f"Warning: Raster {tif} has invalid values (min: {arr.min()}, max: {arr.max()}). Skipping.")
                invalid_tifs.append(tif)
                continue
    for invalid_tif in invalid_tifs:
        os.remove(invalid_tif)
        tif_paths.remove(invalid_tif)


    # Open all datasets
    src_files_to_mosaic = []
    for fp in tif_paths:
        src = rasterio.open(fp)
        src_files_to_mosaic.append(src)

    # Perform merge
    mosaic, out_transform = merge(src_files_to_mosaic)

    # Copy metadata and update
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "crs": src_files_to_mosaic[0].crs
    })

    # Write the mosaic raster to disk
    with rasterio.open(output_tif_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    # Close all inputs
    for src in src_files_to_mosaic:
        src.close()

if __name__ == "__main__":
    
    # ──────────────── Configuration ────────────────────
    # Path to your input GeoPackage of points (must have an 'elevation' field)
    min_elev_points_gpkg     = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM\min_elev_points_100m_clustered.gpkg"
    # Desired output raster path (raw interpolation)
    interpolated_min_raster_dir  = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM\interpolated_HAWS"
    
    # Name of the attribute field holding elevation values
    elevation_field = "elevation"
    # Raster pixel size (in the same units as your GeoPackage CRS)
    pixel_size     = 10.0
    # IDW parameters
    idw_power      = 2.0   # power parameter (controls distance weighting)
    idw_smoothing  = 1.0   # smoothing parameter (reduces bull’s-eye effect)

    # ────────────────────Optional Masking ────────────────────
    # If you have a mask GeoPackage, set its path here; otherwise leave as None
    mask_gpkg                = None
    # Output path for the clipped interpolation
    clipped_min_raster       = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM\test_REM\interpolated_HAWS_clipped.tif"
    
    # ──────────────────── DEM Difference ────────────────────
    original_dem = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\LiDAR\grmw_rasters\bare_earth\hdr.adf"
    diff_output = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM\test_REM\REM_HAWS.tif"
    # ────────────────────────────────────────────────────
        
    # 1) Interpolate point elevations
    interpolate_stream_network(
        gpkg_path=min_elev_points_gpkg,
        cluster_field="cluster_id",  # Adjust this if your GPKG has a different cluster field
        value_field=elevation_field,
        out_dir=interpolated_min_raster_dir,
        pixel_size=pixel_size,
        power=idw_power,
        smoothing=idw_smoothing
    )

    join_tifs_in_folder(
        input_folder=interpolated_min_raster_dir,
        output_tif_path=os.path.join(interpolated_min_raster_dir, "interpolated_HAWS.tif")
    )
    
    # # 2) Optionally clip the interpolated raster by your mask
    # if mask_gpkg:
    #     clip_raster_by_mask(
    #         raster_path=interpolated_min_raster,
    #         mask_gpkg=mask_gpkg,
    #         out_path=clipped_min_raster
    #     )
    #     raster_for_diff = clipped_min_raster
    # else:
    #     raster_for_diff = interpolated_min_raster

    # # 3) Compute difference between original DEM and (clipped) interpolation
    # raster_difference(
    #     raster_path1=original_dem,
    #     raster_path2=raster_for_diff,
    #     output_path=diff_output
    # )



def clip_raster_by_mask(
    raster_path: str,
    mask_gpkg: str,
    out_path: str,
    mask_layer: str = None):
    """
    Clips a raster by the polygon(s) in a GeoPackage mask.
    """
    if mask_gpkg is None:
        raise RuntimeError("No mask GeoPackage provided.")
    warp_opts = gdal.WarpOptions(
        format="GTiff",
        cutlineDSName=mask_gpkg,
        cutlineLayerName=mask_layer,
        cropToCutline=True,
        dstNodata=np.nan
    )
    gdal.Warp(
        destNameOrDestDS=out_path,
        srcDSOrSrcDSTab=raster_path,
        options=warp_opts
    )
    print(f"[✔] Raster clipped by mask: {out_path}")

def raster_difference(raster_path1: str, raster_path2: str, output_path: str):
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

    # Update profile: set dtype and keep nodata
    profile.update(
        dtype=diff.dtype,
        nodata=profile.get('nodata', None)
    )

    # Write output
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(diff.filled(profile.get('nodata', 0)), 1)
