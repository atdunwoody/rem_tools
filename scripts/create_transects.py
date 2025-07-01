import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge
import math

def create_transects(input_gpkg, output_gpkg, spacing, transect_length=100):
    # Read input centerline
    gdf = gpd.read_file(input_gpkg)
    
    # Merge multilines if necessary
    if len(gdf) > 1 or isinstance(gdf.geometry.iloc[0], MultiLineString):
        merged = linemerge(gdf.geometry.unary_union)
        if isinstance(merged, MultiLineString):
            merged = max(merged, key=lambda l: l.length)
        centerline = merged
    else:
        centerline = gdf.geometry.iloc[0]

    # Create transects
    transects = []
    distance = 0
    total_length = centerline.length

    while distance <= total_length:
        point = centerline.interpolate(distance)
        normal = get_normal(centerline, distance)

        # Create transect line (centered on point)
        dx = (transect_length / 2.0) * normal[0]
        dy = (transect_length / 2.0) * normal[1]
        p1 = Point(point.x - dx, point.y - dy)
        p2 = Point(point.x + dx, point.y + dy)
        transect_line = LineString([p1, p2])
        
        # Compute station label
        station_label = format_station(distance)
        transects.append({"geometry": transect_line, "station": station_label})
        
        distance += spacing

    transect_gdf = gpd.GeoDataFrame(transects, crs=gdf.crs)
    transect_gdf.to_file(output_gpkg, driver="GPKG")

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


create_transects(r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20250006_Wallowa R Remeander (AP)\07_GIS\Data\REM\CenterlineValley.gpkg",
                 r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20250006_Wallowa R Remeander (AP)\07_GIS\Data\REM\TransectsValley_400ft.gpkg", 
                 spacing=400, 
                 transect_length=1500)
