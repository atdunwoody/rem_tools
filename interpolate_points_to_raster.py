"""
Interpolate point elevations from a GeoPackage into a raster (IDW).

Configuration:
  - Edit the variables in the “Configuration” block below.
  - Run in VS Code (or any Python IDE)—no command-line arguments required.
"""

from osgeo import gdal, ogr
import math

# ──────────────── Configuration ────────────────────
# Path to your input GeoPackage of points (must have an 'elevation' field)
input_gpkg     = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20250006_Wallowa R Remeander (AP)\07_GIS\Data\REM\min_elev_points_400ft.gpkg"
# Desired output raster path
output_raster  = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20250006_Wallowa R Remeander (AP)\07_GIS\Data\REM\interpolated_HAWS.tif"
# Name of the attribute field holding elevation values
elevation_field = "elevation"
# Raster pixel size (in the same units as your GeoPackage CRS)
pixel_size     = 3.0
# IDW parameters
idw_power      = 2.0   # power parameter (controls distance weighting)
idw_smoothing  = 1.0   # smoothing parameter (reduces bull’s-eye effect)
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
    # 1) Open the vector layer
    ds = ogr.Open(gpkg_path)
    if ds is None:
        raise RuntimeError(f"Cannot open GeoPackage: {gpkg_path}")
    layer = ds.GetLayer(0)

    # 2) Compute the point layer extent: (xmin, xmax, ymin, ymax)
    xmin, xmax, ymin, ymax = layer.GetExtent()

    # 3) Compute raster dimensions (ceil to ensure full coverage)
    x_res = math.ceil((xmax - xmin) / pix_size)
    y_res = math.ceil((ymax - ymin) / pix_size)

    # 4) Configure GDAL GridOptions for IDW
    grid_opts = gdal.GridOptions(
        format="GTiff",
        outputType=gdal.GDT_Float32,
        width=int(x_res),
        height=int(y_res),
        outputBounds=(xmin, ymin, xmax, ymax),
        zfield=field,
        algorithm=f"invdist:power={power}:smoothing={smoothing}"
    )

    # 5) Run the interpolation
    gdal.Grid(
        destName=out_path,
        srcDS=gpkg_path,
        options=grid_opts
    )

    print(f"[✔] Interpolation complete. Output raster: {out_path}")


if __name__ == "__main__":
    interpolate_points_to_raster(
        gpkg_path    = input_gpkg,
        out_path     = output_raster,
        field        = elevation_field,
        pix_size     = pixel_size,
        power        = idw_power,
        smoothing    = idw_smoothing
    )
