import argparse
import sys
import numpy as np
import geopandas as gpd
import fiona
import rasterio
from shapely.geometry import LineString


def segment_and_compute_slope(input_gpkg: str,
                               dem_path: str,
                               output_gpkg: str,
                               interval: float,
                               layer: str = None) -> None:
    """
    Segments a centerline by a specified interval and computes slope
    for each segment by fitting a linear model to all DEM samples along it.

    Parameters
    ----------
    input_gpkg : str
        Path to input GeoPackage containing the centerline.
    dem_path : str
        Path to input DEM raster.
    output_gpkg : str
        Path to output GeoPackage to write segmented centerline with slope.
    interval : float
        Interval (in same units as CRS) to segment the line.
    layer : str, optional
        Name of the layer in the input GeoPackage. If not provided, the first layer is used.
    """
    # Determine layer
    if layer is None:
        layers = fiona.listlayers(input_gpkg)
        if not layers:
            raise ValueError(f"No layers found in {input_gpkg}")
        layer = layers[0]

    # Read centerline
    gdf = gpd.read_file(input_gpkg, layer=layer)
    if gdf.empty:
        raise ValueError("Input centerline layer contains no features.")

    segments = []
    records = []

    # Segment each feature
    for _, row in gdf.iterrows():
        line = row.geometry
        total_length = line.length
        # generate segment break distances
        distances = np.arange(0, total_length, interval)
        if len(distances) == 0 or distances[-1] < total_length:
            distances = np.append(distances, total_length)
        points = [line.interpolate(d) for d in distances]

        # build segment geometries and copy attributes
        for start_pt, end_pt in zip(points[:-1], points[1:]):
            seg = LineString([start_pt, end_pt])
            segments.append(seg)
            records.append(row.drop(labels='geometry'))

    # Create GeoDataFrame of segments
    seg_gdf = gpd.GeoDataFrame(records, geometry=segments, crs=gdf.crs)

    # Compute slope from best-fit line of all DEM samples per segment
    slopes = []
    with rasterio.open(dem_path) as src:
        # sampling resolution: smallest pixel dimension
        xres, yres = src.res
        sample_interval = min(abs(xres), abs(yres))

        for geom in seg_gdf.geometry:
            length = geom.length
            if length <= 0:
                slopes.append(np.nan)
                continue

            # distances along segment at DEM resolution
            dists = np.arange(0, length, sample_interval)
            if len(dists) == 0 or dists[-1] < length:
                dists = np.append(dists, length)

            pts = [geom.interpolate(d) for d in dists]
            coords = [(pt.x, pt.y) for pt in pts]

            # sample elevations
            elevs = [val[0] for val in src.sample(coords)]
            dists_arr = np.array(dists)
            elevs_arr = np.array(elevs, dtype=float)

            # filter out nodata
            valid = ~np.isnan(elevs_arr)
            if valid.sum() < 2:
                slopes.append(np.nan)
                continue

            # fit linear regression: elev = m * dist + b
            m = np.polyfit(dists_arr[valid], elevs_arr[valid], 1)[0]
            slopes.append(abs(m))

    seg_gdf['slope'] = slopes

    # Write output
    output_layer = f"{layer}_segmented"
    seg_gdf.to_file(output_gpkg, driver='GPKG', layer=output_layer)
    print(f"Segmented centerline with slope written to '{output_layer}' in {output_gpkg}")



# Use CLI args if provided, else defaults
if __name__ == '__main__':
    # Default parameters for VSCode debugging
    default_input = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Wallowa\AP_WallowaSunrise_Terrain\Streams\streams_100k.gpkg"
    default_dem = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Wallowa\AP_WallowaSunrise_Terrain\output_USGS10m_EPSG6559.tif"
    default_output = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Wallowa\AP_WallowaSunrise_Terrain\Streams\streams_100k_with_slope.gpkg"
    default_interval = 50.0

    segment_and_compute_slope(
        input_gpkg=default_input,
        dem_path=default_dem,
        output_gpkg=default_output,
        interval=default_interval,
    )