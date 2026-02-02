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
    print(f"Using field '{field}', pixel size {pix_size}, power {power}, "
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

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

def difference_rasters(raster_path1: str, raster_path2: str, output_path: str,
                       resampling: Resampling = Resampling.bilinear,
                       out_dtype=np.float32,
                       nodata_out=None):
    """
    Compute difference raster1 - raster2 on raster1's grid and write GeoTIFF.
    """

    print(f"Computing difference:\n{raster_path1}\n  -\n{raster_path2}\n  =\n{output_path}")

    with rasterio.open(raster_path1) as src1:
        arr1 = src1.read(1, masked=True)
        profile = src1.profile.copy()
        dst_crs = src1.crs
        dst_transform = src1.transform
        dst_height = src1.height
        dst_width = src1.width

        nodata1 = src1.nodata

    with rasterio.open(raster_path2) as src2:
        # Prepare destination array for raster2 reprojected onto raster1 grid
        arr2 = np.empty((dst_height, dst_width), dtype=out_dtype)

        # pick nodata for src2 if present; needed to avoid smearing invalids
        nodata2 = src2.nodata

        reproject(
            source=rasterio.band(src2, 1),
            destination=arr2,
            src_transform=src2.transform,
            src_crs=src2.crs,
            src_nodata=nodata2,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=nodata2,
            resampling=resampling,
        )

    # Build masks: combine raster1 mask with raster2 nodata areas
    mask2 = np.zeros_like(arr2, dtype=bool)
    if nodata2 is not None:
        mask2 = np.isclose(arr2, nodata2)

    # Ensure we have a masked array for arr2
    arr2m = np.ma.array(arr2, mask=mask2)

    diff = (arr1.astype(out_dtype) - arr2m.astype(out_dtype))

    # Decide output nodata
    if nodata_out is None:
        # prefer raster1 nodata if defined, otherwise choose a sane float nodata
        nodata_out = nodata1 if nodata1 is not None else -9999.0

    # Update output profile
    profile.update(
        driver="GTiff",
        count=1,
        dtype=np.dtype(out_dtype).name,
        nodata=nodata_out,
        compress="LZW",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(diff.filled(nodata_out).astype(out_dtype), 1)

    print(f"Done. Output nodata={nodata_out}, dtype={out_dtype}, shape={diff.shape}")

if __name__ == "__main__":
    
    # min_points_gpkg = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\DEMs\UKFP\min_elev_points.gpkg"
    dem = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\REMs\Topography\2016_USDA_DEM.tif"
    min_points_gpkg = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\REMs\GGL\median_elev_points.gpkg"
    output_dir = os.path.dirname(min_points_gpkg)

    # ──────────────── Configuration ────────────────────
    # Name of the attribute field holding elevation values
    elevation_field = "elevation"  # or "BF_depth_Legg_m", "BF_depth_Beechie_m"
    # Raster pixel size (in the same units as your GeoPackage CRS)
    pixel_size     = 1
    # IDW parameters
    idw_power      = 2   # default 2. power parameter (controls distance weighting) higher = more localized influence
    idw_smoothing  = 1   # default 1. smoothing parameter (reduces bull’s-eye effect) greater than 1 = more smoothing
    # Set to half the max valley width in the network
    idw_radius     = 200   # search radius for IDW interpolation

    output_WS_raster = os.path.join(output_dir, f"interpolated_GGL_{pixel_size}m.tif")
    output_HAWS_raster = os.path.join(output_dir, f"GGL_REM_{pixel_size}m.tif")

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

    difference_rasters(
        raster_path1 = dem,
        raster_path2 = output_WS_raster,
        output_path  = output_HAWS_raster
    )
    
    # elevation_field = "BF_depth_Legg_m"  # or "BF_depth_Legg_m", "BF_depth_Beechie_m"
    # output_WS_raster = os.path.join(output_dir, "BF_depth_Legg_m.tif")
    # interpolate_water_surface(
    #     gpkg_path    = min_points_gpkg,
    #     out_path     = output_WS_raster,
    #     field        = elevation_field,
    #     pix_size     = pixel_size,
    #     power        = idw_power,
    #     smoothing    = idw_smoothing,
    #     radius       = idw_radius  
    # )