import math
import os
from decimal import Decimal, getcontext
from typing import Iterable, Optional, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, LineString, MultiLineString, base
from shapely.ops import linemerge, unary_union
from shapely.validation import make_valid
from pyproj import CRS

# ---------------------------
# Helpers
# ---------------------------

M_TO_FT = 3.280839895  # international feet per meter
FT_PER_MILE = 5280.0


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


def _format_station_ft(distance_ft: float) -> str:
    """
    Format stationing in feet as 00+00 where:
      - main = hundreds of feet
      - plus = remaining feet (0-99)
    """
    station_int = int(round(distance_ft))
    plus = station_int % 100
    main = station_int // 100
    main_str = str(main).zfill(2)
    return f"{main_str}+{plus:02d}"


def _smooth_tangent(line: LineString, distance: float, window: float) -> Optional[Tuple[float, float]]:
    """
    Smoothed unit tangent vector at 'distance' along 'line' using forward/back averaging over 'window'.
    Returns (tx, ty) or None if degenerate.
    """
    L = line.length
    if L == 0:
        return None
    d0 = max(0.0, distance - window)
    d1 = min(L, distance + window)

    p = line.interpolate(distance)
    pb = line.interpolate(d0)
    pf = line.interpolate(d1)

    if p.is_empty or pb.is_empty or pf.is_empty:
        return None

    dx_b, dy_b = (p.x - pb.x), (p.y - pb.y)
    dx_f, dy_f = (pf.x - p.x), (pf.y - p.y)

    dx_avg = (dx_b + dx_f) / 2.0
    dy_avg = (dy_b + dy_f) / 2.0
    len_dir = math.hypot(dx_avg, dy_avg)
    if len_dir == 0:
        return None

    return (dx_avg / len_dir, dy_avg / len_dir)


def _azimuth_from_tangent(tx: float, ty: float) -> float:
    """
    Convert unit tangent (tx, ty) to azimuth degrees:
      - 0° = North, 90° = East, increases clockwise.
    """
    return (math.degrees(math.atan2(tx, ty)) + 360.0) % 360.0


def _qgis_rotation_from_azimuth(azimuth_deg: float, *, keep_upright: bool = True) -> float:
    """
    QGIS label rotation (Horizontal oriented text):
      - 0° points East and increases clockwise.

    Convert azimuth (0=N,90=E) -> QGIS rotation (0=E,90=S,180=W,270=N):
      rotation = (azimuth - 90) mod 360

    If keep_upright=True, flips by 180° for angles that would render upside-down.
    """
    rot = (azimuth_deg - 90.0) % 360.0
    if keep_upright and (90.0 < rot < 270.0):
        rot = (rot + 180.0) % 360.0
    return rot


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
    centroid_wgs84 = gdf.to_crs(4326).unary_union.centroid
    lon, lat = centroid_wgs84.x, centroid_wgs84.y
    zone = int((lon + 180) // 6) + 1
    is_northern = lat >= 0
    epsg = 32600 + zone if is_northern else 32700 + zone
    return CRS.from_epsg(epsg)


def _project_for_linear_ops(gdf: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, Optional[CRS]]:
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


def _crs_linear_unit_to_feet_per_unit(crs: CRS) -> Optional[float]:
    """Returns feet per CRS unit if CRS is projected and linear; else None."""
    try:
        if crs.is_geographic:
            return None
        unit_name = (crs.axis_info[0].unit_name or "").lower()
        if "metre" in unit_name or "meter" in unit_name:
            return M_TO_FT
        if "foot" in unit_name or "feet" in unit_name:
            return 1.0
        if unit_name in ("m",):
            return M_TO_FT
        if unit_name in ("ft", "us-ft", "ftus"):
            return 1.0
        return None
    except Exception:
        return None


def _feet_to_crs_units(distance_ft: float, crs: CRS) -> float:
    """Convert a distance in feet to CRS units."""
    ft_per_unit = _crs_linear_unit_to_feet_per_unit(crs)
    if ft_per_unit is None:
        # Fallback: assume meters
        return distance_ft / M_TO_FT
    return distance_ft / ft_per_unit


def _endpoint_points(line: LineString) -> Tuple[Point, Point]:
    coords = list(line.coords)
    return Point(coords[0]), Point(coords[-1])


def _pick_terminal_endpoint_of_max_DA(
    gdf_lines: gpd.GeoDataFrame,
    *,
    da_field: str,
    touch_tol_units: float,
) -> Tuple[int, Point]:
    """
    Returns (row_index_of_max_DA_feature, terminal_endpoint_point)
    where terminal endpoint = endpoint that no other line is within touch_tol_units of.
    """
    if da_field not in gdf_lines.columns:
        raise ValueError(f"Required field '{da_field}' not found in input layer.")

    da = gdf_lines[da_field].to_numpy()
    if len(da) == 0:
        raise ValueError("No line features found.")
    if not np.isfinite(da).any():
        raise ValueError(f"Field '{da_field}' has no finite values to choose a max from.")

    max_idx = int(np.nanargmax(da))
    max_geom = _longest_linestring(gdf_lines.geometry.iloc[max_idx])

    a, b = _endpoint_points(max_geom)

    # Build union of "all other lines"
    others = gdf_lines.drop(gdf_lines.index[max_idx]).geometry
    if len(others) == 0:
        return max_idx, b

    other_union = unary_union(list(others))

    def is_terminal(pt: Point) -> bool:
        return pt.distance(other_union) > touch_tol_units

    a_term = is_terminal(a)
    b_term = is_terminal(b)

    if a_term and not b_term:
        return max_idx, a
    if b_term and not a_term:
        return max_idx, b
    if a_term and b_term:
        return max_idx, b

    raise ValueError(
        "Could not find a terminal endpoint on the max-DA line: both endpoints are touched by other linework.\n"
        "This usually means the max-DA feature is not at the network outlet, or the network has a loop/branching at both ends."
    )


def _dissolve_centerline_containing_feature(
    gdf_proj: gpd.GeoDataFrame,
    *,
    must_intersect_geom: LineString,
    touch_tol_units: float,
) -> LineString:
    """
    Dissolve/merge all linework into (ideally) one LineString.
    If the result is multipart, choose the merged LineString that contains/intersects the provided geometry.
    """
    u = unary_union(list(gdf_proj.geometry))
    merged = linemerge(u)

    def candidates_from_geom(geom) -> list[LineString]:
        if isinstance(geom, LineString):
            return [geom]
        if isinstance(geom, MultiLineString):
            return list(geom.geoms)
        out = []
        if hasattr(geom, "geoms"):
            for gg in geom.geoms:
                if isinstance(gg, LineString):
                    out.append(gg)
                elif isinstance(gg, MultiLineString):
                    out.extend(list(gg.geoms))
        return out

    parts = candidates_from_geom(merged)
    if not parts:
        raise ValueError("Dissolve/merge did not produce any LineStrings.")

    hits = []
    for ln in parts:
        if ln.distance(must_intersect_geom) <= touch_tol_units:
            hits.append(ln)

    if hits:
        return max(hits, key=lambda l: l.length)

    return max(parts, key=lambda l: l.length)


def _orient_line_to_start_at_point(line: LineString, start_pt: Point) -> LineString:
    """
    Ensure line direction starts near start_pt (by comparing to endpoints).
    Reverses coordinates if needed.
    """
    a, b = _endpoint_points(line)
    if start_pt.distance(a) <= start_pt.distance(b):
        return line
    return LineString(list(line.coords)[::-1])


# ---------------------------
# Main function
# ---------------------------

def create_stationing_points_smooth(
    input_gpkg: str,
    output_gpkg: str,
    input_layer: Optional[str] = None,
    output_layer: Optional[str] = None,
    spacing_ft: float = 100.0,     # FEET INPUT
    window_ft: float = 200.0,      # FEET INPUT
    *,
    start_station_mi: float = 0.0, # label offset, in miles
    da_field: str = "DA_km2",
    endpoint_touch_tol_ft: float = 1.0,  # how close counts as "touching" at the outlet node
    debug: bool = True,
    debug_every_n: int = 25,
) -> str:
    """
    Create stationing *points* along a dissolved centerline at exact multiples of spacing_ft.

    Outputs point geometry at each station, plus fields:
      - stream_dir_deg : azimuth of stream tangent (0=N, 90=E, clockwise)
      - rotation       : label rotation for QGIS (0=E, clockwise) adjusted to keep upright
                         (use as data-defined label rotation in QGIS with Horizontal text)
    """
    getcontext().prec = 28

    if spacing_ft <= 0:
        raise ValueError("spacing_ft must be > 0.")
    if window_ft < 0:
        raise ValueError("window_ft must be >= 0.")
    if start_station_mi < 0:
        raise ValueError("start_station_mi must be >= 0 (miles).")

    start_station_offset_ft = float(start_station_mi) * FT_PER_MILE

    # ---- Read & validate
    gdf = gpd.read_file(input_gpkg, layer=input_layer)
    gdf = _ensure_make_valid(gdf)

    # Internal projection for linear units
    gdf_proj, back_crs = _project_for_linear_ops(gdf)
    crs_work = CRS.from_user_input(gdf_proj.crs)

    ft_per_unit = _crs_linear_unit_to_feet_per_unit(crs_work)
    if ft_per_unit is None:
        ft_per_unit = M_TO_FT

    # Tolerances in CRS units
    touch_tol_units = _feet_to_crs_units(endpoint_touch_tol_ft, crs_work)

    # ---- Find terminal start point on max-DA feature
    max_idx, start_pt = _pick_terminal_endpoint_of_max_DA(
        gdf_proj,
        da_field=da_field,
        touch_tol_units=touch_tol_units,
    )
    max_da_geom = _longest_linestring(gdf_proj.geometry.iloc[max_idx])

    # ---- Dissolve centerline, but keep the component containing max-DA feature
    centerline = _dissolve_centerline_containing_feature(
        gdf_proj,
        must_intersect_geom=max_da_geom,
        touch_tol_units=touch_tol_units,
    )
    if centerline.length <= 0:
        raise ValueError("Dissolved centerline has zero length.")

    # ---- Orient centerline so stationing begins at the chosen terminal endpoint
    centerline = _orient_line_to_start_at_point(centerline, start_pt)

    # ---- Convert feet inputs -> CRS units (for geometry work)
    window = _feet_to_crs_units(window_ft, crs_work)

    # ---- Build exact station list in FEET (multiples of spacing_ft) for GEOMETRY
    L_units = centerline.length
    L_ft = float(L_units * ft_per_unit)

    spacing_dec = Decimal(str(spacing_ft))
    L_ft_dec = Decimal(str(L_ft))

    n_steps = int((L_ft_dec / spacing_dec).to_integral_value(rounding="ROUND_FLOOR"))
    station_ft_vals_dec = [spacing_dec * Decimal(k) for k in range(n_steps + 1)]

    rows_out = []
    skips = {"past_end": 0, "empty_p": 0, "no_tangent": 0}

    for i, st_ft_dec in enumerate(station_ft_vals_dec):
        # ---- Distance along line for GEOMETRY (unchanged)
        st_ft_geom = float(st_ft_dec)  # 0, spacing, 2*spacing, ...
        d_units = float(st_ft_dec / Decimal(str(ft_per_unit)))  # CRS units

        if d_units < 0.0:
            continue
        if d_units > L_units:
            skips["past_end"] += 1
            continue

        p = centerline.interpolate(d_units)
        if p.is_empty:
            skips["empty_p"] += 1
            continue

        t = _smooth_tangent(centerline, d_units, window=window)
        if t is None:
            skips["no_tangent"] += 1
            continue
        tx, ty = t

        stream_dir = _azimuth_from_tangent(tx, ty)               # 0=N, clockwise
        rotation = _qgis_rotation_from_azimuth(stream_dir, keep_upright=True)  # 0=E, clockwise

        # ---- LABELS / ATTRIBUTES (shifted)
        st_ft_label = st_ft_geom + start_station_offset_ft
        river_mi_label = st_ft_label / FT_PER_MILE
        station_m_label = st_ft_label / M_TO_FT  # numeric meters for the labeled station

        rows_out.append(
            {
                "geometry": Point(p.x, p.y),               # <-- POINTS (not transect lines)
                "station_ft": _format_station_ft(st_ft_label),
                "river_mi": river_mi_label,
                "station_ft_val": st_ft_label,
                "station_m": station_m_label,
                "station_ft_geom": st_ft_geom,             # unshifted along-line feet (QC)
                "station_mi_geom": st_ft_geom / FT_PER_MILE,
                "stream_dir_deg": stream_dir,              # <-- NEW
                "rotation": rotation,                      # <-- NEW (QGIS label rotation)
            }
        )

        if debug and (i % max(1, debug_every_n) == 0):
            print(
                f"[DEBUG] i={i}/{len(station_ft_vals_dec)-1} "
                f"geom_ft={st_ft_geom:.1f} label_ft={st_ft_label:.1f} "
                f"dir={stream_dir:.1f} rot={rotation:.1f} "
                f"created={len(rows_out)} skips={skips}"
            )

    out_gdf = gpd.GeoDataFrame(rows_out, crs=gdf_proj.crs)
    out_gdf = _to_source_crs(out_gdf, back_crs)

    out_layer = output_layer or "stationing_points"
    out_gdf.to_file(output_gpkg, layer=out_layer, driver="GPKG")

    print(
        f"[✔] Created stationing points to {output_gpkg} layer={out_layer}\n"
        f"    spacing_ft={spacing_ft} ft (exact multiples), window_ft={window_ft} ft\n"
        f"    start_station_mi={start_station_mi} mi (label offset = {start_station_offset_ft:.1f} ft)\n"
        f"    da_field={da_field}, endpoint_touch_tol_ft={endpoint_touch_tol_ft} ft\n"
        f"    Working CRS: {crs_work.to_string()}\n"
        f"    Dissolved centerline length: {L_units:.3f} (CRS units) = {L_ft:.3f} ft\n"
        f"    Stationing begins at terminal endpoint of max-{da_field} feature (geometry).\n"
        f"    Output points created: {len(rows_out)} / attempted: {len(station_ft_vals_dec)}\n"
        f"    Skip counts: {skips}\n\n"
        f"    Fields:\n"
        f"      stream_dir_deg = azimuth (0=N, 90=E, clockwise)\n"
        f"      rotation       = QGIS label rotation (0=E, clockwise; kept upright)\n"
    )
    return output_gpkg


if __name__ == "__main__":
    streams_gpkg = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\REMs\centerline.gpkg"
    input_layer = None

    # FEET inputs:
    spacing_ft = 2640.0  # 0.5 mile
    window_ft = 200.0

    start_station = 7.5  # miles (label offset)

    out_path = os.path.join(
        os.path.dirname(streams_gpkg),
        f"stationing_points_{int(spacing_ft)}ft.gpkg",
    )

    create_stationing_points_smooth(
        input_gpkg=streams_gpkg,
        output_gpkg=out_path,
        input_layer=input_layer,
        output_layer="stationing_points",
        spacing_ft=spacing_ft,
        window_ft=window_ft,
        start_station_mi=start_station,
        da_field="DA_km2",
        endpoint_touch_tol_ft=1.0,
        debug=True,
        debug_every_n=10,
    )
