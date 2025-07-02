import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge
import math
from rasterstats import zonal_stats
import rasterio
from rasterstats import zonal_stats

import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import linemerge

def create_transects(input_gpkg, output_gpkg, spacing, transect_length=100):
    
    print(f"Creating transects from {input_gpkg} with spacing {spacing}m and transect length {transect_length}m...")
    # Read all centerline features
    gdf = gpd.read_file(input_gpkg)

    transects = []
    for idx, row in gdf.iterrows():
        geom = row.geometry

        # If MultiLineString, merge and pick the longest branch
        if isinstance(geom, MultiLineString):
            merged = linemerge(geom)
            if isinstance(merged, MultiLineString):
                merged = max(merged.geoms, key=lambda l: l.length)
            centerline = merged
        else:
            centerline = geom

        total_length = centerline.length
        distance = 0.0

        # Step along this centerline
        while distance <= total_length:
            pt = centerline.interpolate(distance)
            nx, ny = get_normal(centerline, distance)

            # Build transect centered on pt
            half = transect_length / 2.0
            p1 = Point(pt.x - half * nx, pt.y - half * ny)
            p2 = Point(pt.x + half * nx, pt.y + half * ny)
            transect = LineString([p1, p2])

            transects.append({
                "geometry": transect,
                "station": format_station(distance),
                "centerline_id": idx,
                "bank_depth_m": row.get("bank_depth_m", None),
                "bank_width_m": row.get("bank_width_m", None),
            })

            distance += spacing

    # Write out all transects
    transect_gdf = gpd.GeoDataFrame(transects, crs=gdf.crs)
    
    transect_gdf.to_file(output_gpkg, driver="GPKG")
    return output_gpkg

def get_normal(line, distance, delta=0.01):
    """Compute unit normal vector at given distance along the line."""
    d1 = max(distance - delta, 0)
    d2 = min(distance + delta, line.length)
    p1 = line.interpolate(d1)
    p2 = line.interpolate(d2)
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    length = math.hypot(dx, dy)
    if length == 0:
        return (0, 0)
    nx = -dy / length
    ny = dx / length
    return (nx, ny)

def format_station(distance):
    """Format station string #+##."""
    station_int = int(round(distance))
    plus = station_int % 100
    main = station_int // 100
    return f"{main}+{plus:02d}"

if __name__ == "__main__":
    
    # add_BF_dims(
    # r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM\test_REM\streams_100k_clip.gpkg",
    # r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM\PRISM_annual_clipped_EPSG_26911.tif",
    # r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM\test_REM\streams_100k_clip_BF.gpkg"
    # )
    streams_gpkg = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM\Streams\streams_100k_clipped_to_LiDAR.gpkg"
    output_transects_gpkg = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM\transects_200m.gpkg"

    create_transects(streams_gpkg,
                 output_transects_gpkg,
                 spacing=100,
                 transect_length=200,
                 )
