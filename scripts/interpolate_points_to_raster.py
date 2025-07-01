from osgeo import gdal, ogr
import math
import numpy as np

# ──────────────── Configuration ────────────────────
# Path to your input GeoPackage of points (must have an 'elevation' field)
min_elev_points_gpkg     = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20250006_Wallowa R Remeander (AP)\07_GIS\Data\REM\min_elev_points_400ft.gpkg"
# Desired output raster path
interpolated_min_raster  = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20250006_Wallowa R Remeander (AP)\07_GIS\Data\REM\interpolated_HAWS.tif"
# Name of the attribute field holding elevation values
elevation_field = "elevation"
# Raster pixel size (in the same units as your GeoPackage CRS)
pixel_size     = 3.0
# IDW parameters
idw_power      = 2.0   # power parameter (controls distance weighting)
idw_smoothing  = 1.0   # smoothing parameter (reduces bull’s-eye effect)

original_dem = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20250006_Wallowa R Remeander (AP)\07_GIS\Data\AP_WallowaSunriseDEM\WallowaSunriseDEM\GRMW_unclipped_1ft_DEM.tif"
diff_output = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20250006_Wallowa R Remeander (AP)\07_GIS\Data\REM\REM_HAWS.tif"

# ────────────────────────────────────────────────────

def interpolate_points_to_raster(
    gpkg_path: str,
    out_path: str,
    field: str,
    pix_size: float,
    power: float,
    smoothing: float
):
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

    print(f"[✔] Interpolation complete. Output raster: {out_path}")


def difference_rasters(
    raster1_path: str,
    raster2_path: str,
    out_path: str
):
    """
    Computes pixel-wise difference (raster1 - raster2) and saves as a new raster.
    """
    ds1 = gdal.Open(raster1_path)
    ds2 = gdal.Open(raster2_path)
    if ds1 is None or ds2 is None:
        raise RuntimeError(f"Cannot open input rasters: {raster1_path}, {raster2_path}")

    if (ds1.RasterXSize != ds2.RasterXSize or
        ds1.RasterYSize != ds2.RasterYSize or
        ds1.GetGeoTransform() != ds2.GetGeoTransform() or
        ds1.GetProjection() != ds2.GetProjection()):
        raise RuntimeError("Input rasters must have identical size, geotransform, and projection.")

    band1 = ds1.GetRasterBand(1).ReadAsArray().astype(np.float32)
    band2 = ds2.GetRasterBand(1).ReadAsArray().astype(np.float32)
    diff = band1 - band2

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        out_path,
        ds1.RasterXSize,
        ds1.RasterYSize,
        1,
        gdal.GDT_Float32
    )

    out_ds.SetGeoTransform(ds1.GetGeoTransform())
    out_ds.SetProjection(ds1.GetProjection())

    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(diff)
    out_band.SetNoDataValue(np.nan)
    out_band.FlushCache()
    out_ds = None

    print(f"[✔] Difference raster created: {out_path}")


if __name__ == "__main__":
    # Interpolate point elevations
    interpolate_points_to_raster(
        gpkg_path    = min_elev_points_gpkg,
        out_path     = interpolated_min_raster,
        field        = elevation_field,
        pix_size     = pixel_size,
        power        = idw_power,
        smoothing    = idw_smoothing
    )

    # difference_rasters(original_dem, interpolated_min_raster, diff_output)
