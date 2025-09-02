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
                "DA_km2": row.get("DA_km2", None),
                "BF_width_Legg_m": row.get("BF_width_Legg_m", None),
                "BF_depth_Legg_m": row.get("BF_depth_Legg_m", None),
                "BF_width_Castro_m": row.get("BF_width_Castro_m", None),
                "BF_depth_Castro_m": row.get("BF_depth_Castro_m", None),
                "BF_width_Beechie_m": row.get("BF_width_Beechie_m", None),
            })

            distance += spacing

    # Write out all transects
    transect_gdf = gpd.GeoDataFrame(transects, crs=gdf.crs)
    transect_gdf.to_file(output_gpkg, driver="GPKG")
    print(f"[✔] Created transects GeoPackage: {output_gpkg}")
    return output_gpkg


import geopandas as gpd
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import linemerge
from shapely.validation import make_valid

def create_bendy_transects(input_gpkg, output_gpkg, spacing, transect_length=100):
    """
    Creates transects from centerlines in input_gpkg, bending transects slightly 
    to avoid intersecting previously created transects. Processes centerlines 
    from largest to smallest drainage area (DA_km2). Skips any transect that cannot be
    de‑conflicted.
    """
    # read & repair
    gdf = gpd.read_file(input_gpkg)
    try:
        gdf["geometry"] = gdf.geometry.apply(make_valid)
    except ImportError:
        gdf["geometry"] = gdf.geometry.buffer(0)

    # drop empties & zero‑length\    
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.length > 0]

    # sort by descending drainage area
    gdf = gdf.sort_values("DA_km2", ascending=False)

    existing = []   # list of LineString or multi‑vertex transects
    transects = []

    for idx, row in gdf.iterrows():
        geom = row.geometry
        # collapse MultiLineString to its longest branch
        if isinstance(geom, MultiLineString):
            merged = linemerge(geom)
            if isinstance(merged, MultiLineString):
                merged = max(merged.geoms, key=lambda l: l.length)
            centerline = merged
        else:
            centerline = geom

        total_length = centerline.length
        dist = 0.0

        while dist <= total_length:
            pt = centerline.interpolate(dist)
            if pt.is_empty:
                dist += spacing
                continue

            normal = get_normal(centerline, dist)
            if normal is None:
                dist += spacing
                continue
            nx, ny = normal

            half = transect_length / 2.0
            p1 = Point(pt.x - half * nx, pt.y - half * ny)
            p2 = Point(pt.x + half * nx, pt.y + half * ny)
            straight = LineString([p1, p2])

            # does it hit any existing transect?
            if not any(straight.intersects(e) for e in existing):
                chosen = straight
            else:
                # try a single-bend: offset the midpoint by transect_length/4
                bend_off = transect_length / 4.0
                chosen = None
                for sign in (1, -1):
                    mid = Point(pt.x + sign * bend_off * nx,
                                pt.y + sign * bend_off * ny)
                    bend_line = LineString([p1, mid, p2])
                    if not any(bend_line.intersects(e) for e in existing):
                        chosen = bend_line
                        break
                if chosen is None:
                    dist += spacing
                    continue  # skip this station

            # accept & record
            existing.append(chosen)
            transects.append({
                "geometry": chosen,
                "station": format_station(dist),
                "centerline_id": idx,
                "DA_km2": row.get("DA_km2"),
                "BF_width_Legg_m": row.get("BF_width_Legg_m"),
                "BF_depth_Legg_m": row.get("BF_depth_Legg_m"),
                "BF_width_Castro_m": row.get("BF_width_Castro_m"),
                "BF_depth_Castro_m": row.get("BF_depth_Castro_m"),
                "BF_width_Beechie_m": row.get("BF_width_Beechie_m"),
            })

            dist += spacing

    # write out
    out_gdf = gpd.GeoDataFrame(transects, crs=gdf.crs)
    out_gdf.to_file(output_gpkg, driver="GPKG")
    print(f"[✔] Created bendy transects GeoPackage: {output_gpkg}")
    return output_gpkg


if __name__ == "__main__":
    streams_gpkg = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Wallowa\AP_WallowaSunrise_Terrain\Streams\streams_100k.gpkg"
    spacing = 100  # meters
    transect_length = 1  # meters
    output_transects_gpkg = os.path.join(
        os.path.dirname(streams_gpkg),
        f"transects_bendy_{spacing}m_{transect_length}m.gpkg"
    )

    create_bendy_transects(
        streams_gpkg,
        output_transects_gpkg,
        transect_length=transect_length,
        spacing=spacing,
    )
