import math
import os
import re
import sys
from typing import Optional

import fiona
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from osgeo import gdal, ogr


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def sanitize_name(value, prefix: str = "stream") -> str:
    """
    Convert a layer name or stream_id to a filesystem-safe name.
    """
    if value is None:
        value = "unknown"

    name = str(value).strip()
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = name.strip("_")

    if not name:
        name = "unknown"

    if name[0].isdigit():
        name = f"{prefix}_{name}"

    return name


def get_gpkg_layers(gpkg_path: str) -> list[str]:
    """
    Return all layer names in a GeoPackage.
    """
    if not os.path.exists(gpkg_path):
        raise FileNotFoundError(f"GeoPackage not found: {gpkg_path}")

    layers = list(fiona.listlayers(gpkg_path))

    if not layers:
        raise ValueError(f"No layers found in GeoPackage: {gpkg_path}")

    return layers


# -----------------------------------------------------------------------------
# Interpolation
# -----------------------------------------------------------------------------

def interpolate_water_surface(
    gpkg_path: str,
    out_path: str,
    field: str,
    pix_size: float,
    power: float,
    smoothing: float,
    layer_name: str,
    radius: Optional[float] = None,
) -> bool:
    """
    Uses GDAL Grid IDW to interpolate point elevations from one GeoPackage layer
    to a water-surface raster.

    Returns True if interpolation was completed, False if the layer has no valid
    input points for the selected field.
    """
    print(f"\nInterpolating water surface")
    print(f"  Input GPKG: {gpkg_path}")
    print(f"  Layer:      {layer_name}")
    print(f"  Output:     {out_path}")
    print(f"  Field:      {field}")
    print(f"  Pixel size: {pix_size}")
    print(f"  Power:      {power}")
    print(f"  Smoothing:  {smoothing}")
    print(f"  Radius:     {radius}")

    ds = ogr.Open(gpkg_path)
    if ds is None:
        raise RuntimeError(f"Cannot open GeoPackage: {gpkg_path}")

    layer = ds.GetLayerByName(layer_name)
    if layer is None:
        raise ValueError(f"Layer '{layer_name}' not found in {gpkg_path}")

    layer.SetAttributeFilter(f"{field} IS NOT NULL")
    valid_count = layer.GetFeatureCount()

    if valid_count == 0:
        print(f"Skipping layer '{layer_name}': no non-null values in field '{field}'.")
        return False

    xmin, xmax, ymin, ymax = layer.GetExtent()

    if xmin == xmax or ymin == ymax:
        print(f"Skipping layer '{layer_name}': invalid point extent.")
        return False

    x_res = max(math.ceil((xmax - xmin) / pix_size), 1)
    y_res = max(math.ceil((ymax - ymin) / pix_size), 1)

    if radius is None:
        alg = f"invdist:power={power}:smoothing={smoothing}:nodata=0"
    else:
        alg = (
            f"invdist:power={power}:smoothing={smoothing}:"
            f"radius1={radius}:radius2={radius}:nodata=0"
        )

    grid_opts = gdal.GridOptions(
        format="GTiff",
        outputType=gdal.GDT_Float32,
        width=int(x_res),
        height=int(y_res),
        outputBounds=(xmin, ymin, xmax, ymax),
        zfield=field,
        algorithm=alg,
        where=f"{field} IS NOT NULL",
        layers=[layer_name],
        noData=0,
    )

    def progress(complete, message, _):
        pct = complete * 100
        sys.stdout.write(f"\r[Interpolation] {pct:6.2f}% {message}")
        sys.stdout.flush()
        return 1

    result = gdal.Grid(
        destName=out_path,
        srcDS=gpkg_path,
        options=grid_opts,
        callback=progress,
    )

    print()

    if result is None:
        raise RuntimeError(f"GDAL Grid failed for layer '{layer_name}'.")

    result = None
    ds = None

    print(f"[✔] Interpolation complete: {out_path}")
    return True


# -----------------------------------------------------------------------------
# Raster differencing
# -----------------------------------------------------------------------------

def difference_rasters(
    raster_path1: str,
    raster_path2: str,
    output_path: str,
    resampling: Resampling = Resampling.bilinear,
    out_dtype=np.float32,
    nodata_out: Optional[float] = None,
):
    """
    Compute raster_path1 - raster_path2 on raster_path1's grid and write GeoTIFF.

    For REM work:
      raster_path1 = DEM
      raster_path2 = interpolated water surface
      output       = DEM - water surface
    """
    print(f"\nComputing difference:")
    print(f"  {raster_path1}")
    print(f"  minus")
    print(f"  {raster_path2}")
    print(f"  equals")
    print(f"  {output_path}")

    with rasterio.open(raster_path1) as src1:
        arr1 = src1.read(1, masked=True)
        profile = src1.profile.copy()

        dst_crs = src1.crs
        dst_transform = src1.transform
        dst_height = src1.height
        dst_width = src1.width

        nodata1 = src1.nodata

    with rasterio.open(raster_path2) as src2:
        arr2 = np.empty((dst_height, dst_width), dtype=out_dtype)
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

    mask2 = np.zeros_like(arr2, dtype=bool)

    if nodata2 is not None:
        mask2 = np.isclose(arr2, nodata2)

    arr2m = np.ma.array(arr2, mask=mask2)
    diff = arr1.astype(out_dtype) - arr2m.astype(out_dtype)

    if nodata_out is None:
        nodata_out = nodata1 if nodata1 is not None else -9999.0

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

    print(f"[✔] REM complete: {output_path}")
    print(f"    nodata={nodata_out}, dtype={out_dtype}, shape={diff.shape}")


# -----------------------------------------------------------------------------
# Average overlapping REMs
# -----------------------------------------------------------------------------

def average_rem_rasters(
    rem_paths: list[str],
    output_path: str,
    nodata_out: float = -9999.0,
) -> str:
    """
    Create a merged REM by averaging valid pixels across input REMs.

    Assumes all REMs are on the same grid. This is true for this workflow because
    each REM is differenced onto the DEM grid.

    Where only one REM has data, that value is retained.
    Where two or more REMs overlap, the output is the arithmetic mean.
    Where no REM has data, output is nodata.
    """
    if not rem_paths:
        raise ValueError("No REM rasters provided for averaging.")

    print(f"\nCreating merged average REM:")
    print(f"  Output: {output_path}")
    print(f"  Input REM count: {len(rem_paths)}")

    with rasterio.open(rem_paths[0]) as ref:
        profile = ref.profile.copy()
        height = ref.height
        width = ref.width
        transform = ref.transform
        crs = ref.crs

    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=nodata_out,
        compress="LZW",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    # Confirm all rasters are aligned before averaging.
    for path in rem_paths:
        with rasterio.open(path) as src:
            if (
                src.height != height
                or src.width != width
                or src.transform != transform
                or src.crs != crs
            ):
                raise ValueError(
                    "REM rasters are not aligned. This averaging function expects "
                    f"matching grid geometry. Problem raster: {path}"
                )

    with rasterio.open(output_path, "w", **profile) as dst:
        with rasterio.open(rem_paths[0]) as ref:
            windows = list(ref.block_windows(1))

        for _, window in windows:
            sum_arr = np.zeros((window.height, window.width), dtype=np.float64)
            count_arr = np.zeros((window.height, window.width), dtype=np.uint16)

            for path in rem_paths:
                with rasterio.open(path) as src:
                    arr = src.read(1, window=window).astype(np.float32)
                    nodata = src.nodata

                    valid = np.isfinite(arr)

                    if nodata is not None:
                        valid &= ~np.isclose(arr, nodata)

                    # Optional REM-specific guard:
                    # If you want to exclude negative REM values, uncomment this.
                    # valid &= arr >= 0

                    sum_arr[valid] += arr[valid]
                    count_arr[valid] += 1

            out = np.full((window.height, window.width), nodata_out, dtype=np.float32)
            valid_out = count_arr > 0
            out[valid_out] = (sum_arr[valid_out] / count_arr[valid_out]).astype(np.float32)

            dst.write(out, 1, window=window)

    print(f"[✔] Merged average REM complete: {output_path}")
    return output_path


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def create_rems_from_multilayer_min_points(
    dem: str,
    min_points_gpkg: str,
    output_dir: str,
    elevation_field: str = "elevation",
    pixel_size: float = 3,
    idw_power: float = 2,
    idw_smoothing: float = 1,
    idw_radius: Optional[float] = 1000,
    overwrite: bool = True,
) -> dict:
    """
    Create one REM per min-points layer, then create a merged average REM.

    Returns a dictionary with:
      - water_surface_rasters
      - rem_rasters
      - merged_rem
    """
    os.makedirs(output_dir, exist_ok=True)

    ws_dir = os.path.join(output_dir, "water_surface_by_stream_id")
    rem_dir = os.path.join(output_dir, "rem_by_stream_id")

    os.makedirs(ws_dir, exist_ok=True)
    os.makedirs(rem_dir, exist_ok=True)

    layers = get_gpkg_layers(min_points_gpkg)

    print(f"Found {len(layers)} min-point layers:")
    for layer in layers:
        print(f"  - {layer}")

    water_surface_rasters = []
    rem_rasters = []

    for layer_name in layers:
        safe_name = sanitize_name(layer_name)

        output_ws_raster = os.path.join(
            ws_dir,
            f"interpolated_WS_{safe_name}_{pixel_size}ft.tif",
        )

        output_rem_raster = os.path.join(
            rem_dir,
            f"REM_{safe_name}_{pixel_size}ft.tif",
        )

        if overwrite:
            for path in [output_ws_raster, output_rem_raster]:
                if os.path.exists(path):
                    os.remove(path)

        completed = interpolate_water_surface(
            gpkg_path=min_points_gpkg,
            out_path=output_ws_raster,
            field=elevation_field,
            pix_size=pixel_size,
            power=idw_power,
            smoothing=idw_smoothing,
            layer_name=layer_name,
            radius=idw_radius,
        )

        if not completed:
            continue

        difference_rasters(
            raster_path1=dem,
            raster_path2=output_ws_raster,
            output_path=output_rem_raster,
        )

        water_surface_rasters.append(output_ws_raster)
        rem_rasters.append(output_rem_raster)

    if not rem_rasters:
        raise RuntimeError("No REM rasters were created. Check input layers and elevation field.")

    merged_rem = os.path.join(
        output_dir,
        f"merged_average_REM_{pixel_size}ft.tif",
    )

    if overwrite and os.path.exists(merged_rem):
        os.remove(merged_rem)

    average_rem_rasters(
        rem_paths=rem_rasters,
        output_path=merged_rem,
        nodata_out=-9999.0,
    )

    return {
        "water_surface_rasters": water_surface_rasters,
        "rem_rasters": rem_rasters,
        "merged_rem": merged_rem,
    }


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    dem = (
        r"C:\L\Lichen\Lichen - Documents\Projects\20260003_Owens-Snipe Assessment "
        r"(UCSWCD)\07_GIS\0_Data_In\Public\LiDAR\USGS3ft_proj_2020-2021.tif"
    )

    min_points_gpkg = (
        r"C:\L\Lichen\Lichen - Documents\Projects\20260003_Owens-Snipe Assessment "
        r"(UCSWCD)\07_GIS\1_Analysis\Stream Network Analysis\REM"
        r"\min_elev_points_by_stream_id.gpkg"
    )

    output_dir = r"C:\L\Lichen\Lichen - Documents\Projects\20260003_Owens-Snipe Assessment (UCSWCD)\07_GIS\1_Analysis\Stream Network Analysis\REM\REM Tiles"

    # ──────────────── Configuration ────────────────────

    elevation_field = "elevation"

    # Pixel size in the same units as your GeoPackage / DEM CRS.
    pixel_size = 3

    # IDW parameters
    idw_power = 2
    idw_smoothing = 1

    # Search radius for IDW interpolation.
    # Set to approximately half the max valley width if that is your intended limit.
    idw_radius = 350

    # ────────────────────────────────────────────────────

    results = create_rems_from_multilayer_min_points(
        dem=dem,
        min_points_gpkg=min_points_gpkg,
        output_dir=output_dir,
        elevation_field=elevation_field,
        pixel_size=pixel_size,
        idw_power=idw_power,
        idw_smoothing=idw_smoothing,
        idw_radius=idw_radius,
        overwrite=True,
    )

    print("\nOutputs:")
    print(f"  Merged REM: {results['merged_rem']}")
    print(f"  Individual REM count: {len(results['rem_rasters'])}")