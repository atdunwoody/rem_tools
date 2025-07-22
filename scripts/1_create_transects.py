import geopandas as gpd
import math
import os
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import linemerge


def get_normal(line, distance, delta=0.01):
    """Compute unit normal vector at given distance along the line. Return None if geometry is degenerate."""
    d1 = max(distance - delta, 0.0)
    d2 = min(distance + delta, line.length)
    p1 = line.interpolate(d1)
    p2 = line.interpolate(d2)

    # Skip if interpolation failed
    if p1.is_empty or p2.is_empty:
        return None

    dx = p2.x - p1.x
    dy = p2.y - p1.y
    length = math.hypot(dx, dy)
    if length == 0:
        return None
    # normal: (-dy, dx) / length
    return (-dy / length, dx / length)


def format_station(distance):
    """Format station string as 'main+plus' with two-digit plus."""
    station_int = int(round(distance))
    plus = station_int % 100
    main = station_int // 100
    return f"{main}+{plus:02d}"


def create_transects(input_gpkg, output_gpkg, spacing, transect_length=100):
    print(f"Creating transects from {input_gpkg} with spacing {spacing}m and transect length {transect_length}m...")
    gdf = gpd.read_file(input_gpkg)

    # Repair geometries: make valid or buffer(0)
    try:
        from shapely.validation import make_valid
        gdf["geometry"] = gdf.geometry.apply(make_valid)
    except ImportError:
        gdf["geometry"] = gdf.geometry.buffer(0)

    # Drop empty or zero-length features
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.length > 0]

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
            # skip if interpolation failed
            if pt.is_empty:
                distance += spacing
                continue

            normal = get_normal(centerline, distance)
            # skip degenerate normals
            if normal is None:
                distance += spacing
                continue

            nx, ny = normal
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
    print(f"[✔] Created transects GeoPackage: {output_gpkg}")
    return output_gpkg


if __name__ == "__main__":
    streams_gpkg = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\REM\Voronoi Method\low coverage manual\manual_centerline_fixed.gpkg"
    spacing = 10  # meters
    transect_length = 500  # meters
    output_transects_gpkg = os.path.join(
        os.path.dirname(streams_gpkg),
        f"transects_{spacing}m_{transect_length}m.gpkg"
    )

    create_transects(
        streams_gpkg,
        output_transects_gpkg,
        spacing=spacing,
        transect_length=transect_length
    )
