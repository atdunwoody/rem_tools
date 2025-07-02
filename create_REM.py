"""
Scripts to create a Relative Elevation Model (REM) using only a DEM as an input.
Steps:

1. Get streams from whitebox tools. These are thresholded to a flow accumulation (e.g. basin areas in projected units).
    A "thinned" stream is also produced where the number of nodes is reduced
2. Create transects from the thinned streams. These transects are perpendicular to the stream and spaced at a specified interval.
3. Get elevation values along the transects from the DEM. The options are to pick the minimum value along the transect (e.g. water surface) or the median value (e.g. valley bottom)
   The width of these transects should be set to the approximate width of the valley bottom.
   Points are created along each transect line at the minimum value as well as both ends of the transect.
4. Interpolate the elevation values to create a continuous surface.
"""

from scripts.get_streams import get_streams
from scripts.create_transects import create_transects
from scripts.get_elevations_along_transect import extract_min_points
from scripts.interpolate_points_to_raster import interpolate_points_to_raster, difference_rasters
import os

overwrite = False # Set to True if you want to overwrite existing files

dem = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\LiDAR\rasters_USGS10m\USGS 10m DEM Clip.tif"
output_dir = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM"

# stream_mask_gpkg = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM\Streams\streams_mask.gpkg"
stream_mask_gpkg = None # If you don't have a mask, set this to None
base_DEM_for_REM_diff = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\LiDAR\grmw_rasters\bare_earth\hdr.adf"

threshold = 10000 # Flow accum threshold in units of DEM (e.g. 10000 m²)

# 2. Transect params
transect_spacing = 100 # Spacing between transects in meters
transect_length = 100 # Length of transects in meters, adjust according to sinuosity and valley bottom width

# 4. Interpolation params
pixel_size = 1.0 # Pixel size for the output raster in meters
idw_power = 2.0 # Inverse distance weighting power
idw_smoothing = 1.0 # Smoothing factor for IDW interpolation

# 1. Get streams from DEM
streams_output_dir = os.path.join(output_dir, f"Streams_{threshold/1000}k")
streams_gpkg = os.path.join(streams_output_dir, f"streams_{threshold/1000}k.gpkg")
if not os.path.exists(streams_gpkg) or overwrite:
    streams_gpkg = get_streams(dem, output_dir, threshold=threshold, overwrite=False, breach_depressions=True, thin_n=10)

# 2. Create transects from streams
streams_gpkg = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM\Streams\streams_100k_clipped_to_LiDAR.gpkg"

transects_gpkg = os.path.join(output_dir, f"transects_{transect_spacing}m.gpkg")
if not os.path.exists(transects_gpkg) or overwrite:
    transect_gpkg = create_transects(streams_gpkg, transects_gpkg, spacing=transect_spacing, transect_length=transect_length)

# 3. Get elevation values along transects
min_elev_points_gpkg = os.path.join(output_dir, f"min_elev_points_{transect_spacing}m.gpkg")
if not os.path.exists(min_elev_points_gpkg) or overwrite:
    # Extract minimum elevation points along transects
    # This will create points at the minimum elevation along each transect line
    # and at both ends of the transect.
    min_elev_points_gpkg = extract_min_points(transect_gpkg, dem, min_elev_points_gpkg)

# 4. Interpolate points to raster
interpolated_min_raster = os.path.join(output_dir, "interpolated_HAWS.tif")
if not os.path.exists(interpolated_min_raster) or overwrite:
    interpolate_points_to_raster(min_elev_points_gpkg, interpolated_min_raster, field = "elevation", pix_size=pixel_size, power=idw_power, smoothing=idw_smoothing)

# 5. Compute difference between original DEM and interpolated raster

diff_output = os.path.join(output_dir, "REM_HAWS.tif")
raster_difference(base_DEM_for_REM_diff, interpolatd_raster_clipped, diff_output)
