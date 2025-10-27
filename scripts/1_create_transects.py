import math
import os
from typing import Iterable, Optional, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString, base
from shapely.ops import linemerge
from shapely.validation import make_valid
from pyproj import CRS

# ---------------------------
# Helpers
# ---------------------------

def _ensure_make_valid(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Make geometries valid. Falls back to buffer(0) where needed."""
    try:
        gdf["geometry"] = gdf.geometry.apply(make_valid)
    except Exception:
        gdf["geometry"] = gdf.geometry.buffer(0)
    # Drop empties and zero-length lines
    gdf = gdf[~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.length > 0]
    return gdf


def _longest_linestring(geom: base.BaseGeometry) -> LineString:
    """Collapse to a single LineString, selecting longest branch if MultiLineString."""
    if isinstance(geom, LineString):
        return geom
    if isinstance(geom, MultiLineString):
        merged = linemerge(geom)
        if isinstance(merged, MultiLineString):
            return max(merged.geoms, key=lambda l: l.length)
        return merged
    # If geometry collection or other, try extracting lines
    try:
        parts: Iterable[LineString] = [g for g in geom.geoms if isinstance(g, LineString)]
        if not parts:
            raise ValueError("Geometry does not contain a LineString.")
        return max(parts, key=lambda l: l.length)
    except Exception as e:
        raise ValueError(f"Unsupported geometry type for centerline: {geom.geom_type}") from e


def _format_station(distance: float) -> str:
    """Format station as main+plus with two digits."""
    station_int = int(round(distance))
    plus = station_int % 100
    main = station_int // 100
    return f"{main}+{plus:02d}"


def _smooth_normal(line: LineString, distance: float, window: float) -> Optional[Tuple[float, float]]:
    """
    Smoothed unit normal vector at 'distance' along 'line' using forward/back averaging over 'window'.
    Returns (nx, ny) or None if degenerate.
    """
    L = line.length
    if L == 0:
        return None
    # Clamp sample points
    d0 = max(0.0, distance - window)
    d1 = min(L, distance + window)
    # Interpolate
    p = line.interpolate(distance)
    pb = line.interpolate(d0)
    pf = line.interpolate(d1)

    if p.is_empty or pb.is_empty or pf.is_empty:
        return None

    # Back and forward vectors relative to p
    dx_b, dy_b = (p.x - pb.x), (p.y - pb.y)
    dx_f, dy_f = (pf.x - p.x), (pf.y - p.y)

    # Average direction
    dx_avg = (dx_b + dx_f) / 2.0
    dy_avg = (dy_b + dy_f) / 2.0
    len_dir = math.hypot(dx_avg, dy_avg)
    if len_dir == 0:
        return None

    # Perpendicular (rotate +90 deg): (-dy, dx), then normalize
    nx, ny = (-dy_avg / len_dir, dx_avg / len_dir)
    return (nx, ny)


def _needs_projection(crs: Optional[CRS]) -> bool:
    """True if CRS is geographic (degrees) or missing."""
    if crs is None:
        return True
    try:
        c = CRS.from_user_input(crs)
        return c.is_geographic
    except Exception:
        return True


def _guess_local_utm_crs(gdf: gpd.GeoDataFrame) -> CRS:
    """Choose a reasonable UTM based on dataset centroid (WGS84)."""
    # Reproject a lightweight centroid to WGS84
    centroid_wgs84 = gdf.to_crs(4326).unary_union.centroid
    lon, lat = centroid_wgs84.x, centroid_wgs84.y
    zone = int((lon + 180) // 6) + 1
    is_northern = lat >= 0
    epsg = 32600 + zone if is_northern else 32700 + zone
    return CRS.from_epsg(epsg)


def _project_for_linear_ops(
    gdf: gpd.GeoDataFrame,
) -> Tuple[gpd.GeoDataFrame, Optional[CRS]]:
    """
    If source CRS is geographic/unknown, project to a guessed UTM for linear units (meters).
    Returns (projected_gdf, back_crs) where back_crs is the original CRS (or None if none).
    """
    src_crs = gdf.crs
    if _needs_projection(src_crs):
        utm = _guess_local_utm_crs(gdf)
        return gdf.set_crs(src_crs, allow_override=True).to_crs(utm), src_crs
    return gdf, None


def _to_source_crs(gdf: gpd.GeoDataFrame, back_crs: Optional[CRS]) -> gpd.GeoDataFrame:
    """Project back to source CRS if one was provided; otherwise return as-is."""
    if back_crs is not None:
        return gdf.to_crs(back_crs)
    return gdf


# ---------------------------
# Main function
# ---------------------------

def create_bendy_transects_smooth(
    input_gpkg: str,
    output_gpkg: str,
    input_layer: Optional[str] = None,
    output_layer: Optional[str] = None,
    spacing: float = 100.0,
    transect_length: float = 1000.0,
    window: float = 200.0,
) -> str:
    """
    Create de-conflicted transects that use a *smoothed* perpendicular orientation,
    derived from forward/back vectors over a 'window' distance (meters).
    - Preserves source CRS in output, projecting internally if the source is geographic.
    - Ensures 'DA_km2' exists on the input centerlines (created with NaN if absent).
    - Processes centerlines by descending 'DA_km2' to prioritize larger rivers.

    Parameters
    ----------
    input_gpkg : str
        Path to input GeoPackage containing centerlines (LineString/MultiLineString).
    output_gpkg : str
        Path to output GeoPackage to write transects.
    input_layer : Optional[str]
        Name of the input layer in the GeoPackage. If None, default layer is used.
    output_layer : Optional[str]
        Name of the output layer to create. If None, uses 'transects_bendy_smooth'.
    spacing : float
        Spacing between transects along the centerline (meters).
    transect_length : float
        Total length of each transect (meters).
    window : float
        Smoothing window for direction estimation (meters). Larger = smoother orientation.

    Returns
    -------
    str
        The output GeoPackage path.
    """
    # ---- Read & validate
    gdf = gpd.read_file(input_gpkg, layer=input_layer)
    gdf = _ensure_make_valid(gdf)

    # Ensure DA_km2 exists (if missing, create with NaN)
    if "DA_km2" not in gdf.columns:
        gdf["DA_km2"] = np.nan

    # Internal projection for linear units (meters)
    gdf_proj, back_crs = _project_for_linear_ops(gdf)

    # Sort by descending drainage area (NaNs sort last)
    gdf_proj = gdf_proj.sort_values("DA_km2", ascending=False)

    existing: list[LineString] = []  # accepted transects (for de-conflict tests)
    rows_out = []

    for idx, row in gdf_proj.iterrows():
        center_geom = _longest_linestring(row.geometry)
        L = center_geom.length
        if L <= 0:
            continue

        d = 0.0
        half = transect_length / 2.0

        while d <= L:
            # Base point
            p = center_geom.interpolate(d)
            if p.is_empty:
                d += spacing
                continue

            # Smoothed normal
            n = _smooth_normal(center_geom, d, window=window)
            if n is None:
                d += spacing
                continue
            nx, ny = n

            # Endpoints of the straight candidate
            p1 = Point(p.x - half * nx, p.y - half * ny)
            p2 = Point(p.x + half * nx, p.y + half * ny)
            straight = LineString([p1, p2])

            # De-conflict: accept straight if no intersections
            if not any(straight.intersects(e) for e in existing):
                chosen = straight
            else:
                # Try single-bend (offset mid along the normal by quarter-length)
                bend_off = transect_length / 4.0
                chosen = None
                for sign in (1, -1):
                    mid = Point(p.x + sign * bend_off * nx, p.y + sign * bend_off * ny)
                    bend_line = LineString([p1, mid, p2])
                    if not any(bend_line.intersects(e) for e in existing):
                        chosen = bend_line
                        break
                if chosen is None:
                    # Could not de-conflict this station; skip
                    d += spacing
                    continue

            # Accept & record
            existing.append(chosen)
            rows_out.append(
                {
                    "geometry": chosen,
                    "station": _format_station(d),
                    "centerline_id": idx,
                    "DA_km2": row.get("DA_km2"),
                    "BF_width_Legg_m": row.get("BF_width_Legg_m"),
                    "BF_depth_Legg_m": row.get("BF_depth_Legg_m"),
                    "BF_width_Castro_m": row.get("BF_width_Castro_m"),
                    "BF_depth_Castro_m": row.get("BF_depth_Castro_m"),
                    "BF_width_Beechie_m": row.get("BF_width_Beechie_m"),
                }
            )
            d += spacing

    # Build GeoDataFrame in projected space, then convert back to source CRS
    out_gdf = gpd.GeoDataFrame(rows_out, crs=gdf_proj.crs)
    out_gdf = _to_source_crs(out_gdf, back_crs)

    # Write
    out_layer = output_layer or "transects_bendy_smooth"
    out_gdf.to_file(output_gpkg, layer=out_layer, driver="GPKG")
    print(
        f"[✔] Created bendy transects (smoothed normals) to {output_gpkg} layer={out_layer} "
        f"(spacing={spacing} m, length={transect_length} m, window={window} m)."
    )
    return output_gpkg



if __name__ == "__main__":
    streams_gpkg = r"C:\L\OneDrive - Lichen\Documents\Projects\SF Toutle Brownell\REM\loch_trouble\streams_100k.gpkg"
    input_layer = None  # e.g., "centerline"; set if your GPKG has multiple layers
    spacing = 30.0
    transect_length = 200.0
    window = 200.0

    out_path = os.path.join(
        os.path.dirname(streams_gpkg),
        f"transects_bendy_smooth_{spacing}m_{transect_length}m.gpkg",
    )
    create_bendy_transects_smooth(
        input_gpkg=streams_gpkg,
        output_gpkg=out_path,
        input_layer=input_layer,
        output_layer="transects_bendy_smooth",
        spacing=spacing,
        transect_length=transect_length,
        window=window,
    )
