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

def _sanitize_layer_name(value, prefix: str = "stream") -> str:
    """
    Convert a stream_id value to a safe GeoPackage layer name.
    """
    import re

    if value is None or (isinstance(value, float) and np.isnan(value)):
        value = "unknown"

    name = str(value).strip()
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = name.strip("_")

    if not name:
        name = "unknown"

    # Avoid layer names that begin with a number.
    if name[0].isdigit():
        name = f"{prefix}_{name}"

    return name

def create_transects(
    input_gpkg: str,
    output_gpkg: str,
    DA_field: str = "DA_km2",
    stream_id_field: str = "stream_id",
    input_layer: Optional[str] = None,
    spacing: float = 100.0,
    window: float = 200.0,
    trans_power=1 / 3,
    trans_multiplier=100.0,
    overwrite: bool = True,
) -> str:
    """
    Create de-conflicted transects from stream centerlines and write one
    output GeoPackage layer per unique stream_id.

    Notes
    -----
    - Preserves source CRS in output, projecting internally if source CRS is geographic.
    - Requires `DA_field` and `stream_id_field` in the input stream layer.
    - Processes centerlines by descending drainage area to prioritize larger rivers.
    - Keeps transect de-confliction separate within each stream_id/output layer.
    - Transects from different stream_id layers are allowed to intersect.
    - If multiple input features share the same stream_id, their transects are written
      to the same output layer and are de-conflicted against each other.
    """
    output_dir = os.path.dirname(output_gpkg)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if overwrite and os.path.exists(output_gpkg):
        os.remove(output_gpkg)

    gdf = gpd.read_file(input_gpkg, layer=input_layer)
    gdf = _ensure_make_valid(gdf)

    if DA_field not in gdf.columns:
        raise ValueError(
            f"DA_field '{DA_field}' not found in input data columns: "
            f"{gdf.columns.tolist()}"
        )

    if stream_id_field not in gdf.columns:
        raise ValueError(
            f"stream_id_field '{stream_id_field}' not found in input data columns: "
            f"{gdf.columns.tolist()}"
        )

    gdf_proj, back_crs = _project_for_linear_ops(gdf)
    gdf_proj = gdf_proj.sort_values(DA_field, ascending=False)

    # Accumulate transects by sanitized stream_id layer name.
    transects_by_layer: dict[str, list[dict]] = {}

    # Track existing transects separately by stream_id/output layer.
    # This is the key change: de-confliction is no longer global.
    existing_by_layer: dict[str, list[LineString]] = {}

    # Track original stream_id values for reporting.
    stream_ids_by_layer: dict[str, object] = {}

    total_features_processed = 0
    total_transects_created = 0

    for idx, row in gdf_proj.iterrows():
        stream_id = row.get(stream_id_field)
        out_layer = _sanitize_layer_name(stream_id)

        # Only check intersections against transects already created
        # for this same stream_id/output layer.
        existing = existing_by_layer.setdefault(out_layer, [])

        center_geom = _longest_linestring(row.geometry)
        L = center_geom.length
        if L <= 0:
            continue

        da_value = row.get(DA_field, np.nan)

        try:
            da_value = float(da_value)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(da_value) or da_value <= 0:
            continue

        transect_length = (da_value ** trans_power) * trans_multiplier
        half = transect_length / 2.0

        rows_out = []

        d = 0.0
        while d <= L:
            p = center_geom.interpolate(d)
            if p.is_empty:
                d += spacing
                continue

            n = _smooth_normal(center_geom, d, window=window)
            if n is None:
                d += spacing
                continue

            nx, ny = n

            p1 = Point(p.x - half * nx, p.y - half * ny)
            p2 = Point(p.x + half * nx, p.y + half * ny)
            straight = LineString([p1, p2])

            if not any(straight.intersects(e) for e in existing):
                chosen = straight
            else:
                bend_off = transect_length / 4.0
                chosen = None

                for sign in (1, -1):
                    mid = Point(p.x + sign * bend_off * nx, p.y + sign * bend_off * ny)
                    bend_line = LineString([p1, mid, p2])

                    if not any(bend_line.intersects(e) for e in existing):
                        chosen = bend_line
                        break

                if chosen is None:
                    d += spacing
                    continue

            existing.append(chosen)

            rows_out.append(
                {
                    "geometry": chosen,
                    "station": _format_station(d),
                    "centerline_id": idx,
                    stream_id_field: stream_id,
                    DA_field: da_value,
                    "transect_length_m": transect_length,
                    "BF_width_Legg_m": row.get("BF_width_Legg_m"),
                    "BF_depth_Legg_m": row.get("BF_depth_Legg_m"),
                    "BF_width_Castro_m": row.get("BF_width_Castro_m"),
                    "BF_depth_Castro_m": row.get("BF_depth_Castro_m"),
                    "BF_width_Beechie_m": row.get("BF_width_Beechie_m"),
                }
            )

            d += spacing

        if not rows_out:
            continue

        transects_by_layer.setdefault(out_layer, []).extend(rows_out)
        stream_ids_by_layer[out_layer] = stream_id

        total_features_processed += 1
        total_transects_created += len(rows_out)

        print(
            f"[✔] Created {len(rows_out)} transects for feature {idx}, "
            f"stream_id={stream_id}."
        )

    layers_written = 0
    total_transects_written = 0

    for out_layer, rows in transects_by_layer.items():
        if not rows:
            continue

        out_gdf = gpd.GeoDataFrame(rows, crs=gdf_proj.crs)
        out_gdf = _to_source_crs(out_gdf, back_crs)

        out_gdf.to_file(output_gpkg, layer=out_layer, driver="GPKG")

        layers_written += 1
        total_transects_written += len(out_gdf)

        print(
            f"[✔] Wrote layer '{out_layer}' for stream_id="
            f"{stream_ids_by_layer.get(out_layer)} with {len(out_gdf)} transects."
        )

    print(
        f"[✔] Finished writing {layers_written} layers and "
        f"{total_transects_written} transects to {output_gpkg}."
    )

    print(
        f"[i] Processed {total_features_processed} input features with valid transects."
    )

    return output_gpkg



if __name__ == "__main__":
    streams_gpkg = r"C:\L\Lichen\Lichen - Documents\Projects\20260003_Owens-Snipe Assessment (UCSWCD)\07_GIS\1_Analysis\Stream Network Analysis\Streams\streams_1km2_group_ids.gpkg"

    input_layer = None
    spacing = 100  # meters, regardless of source CRS units
    window = 1000.0

    out_path = r"C:\L\Lichen\Lichen - Documents\Projects\20260003_Owens-Snipe Assessment (UCSWCD)\07_GIS\1_Analysis\Stream Network Analysis\REM\transects.gpkg"

    create_transects(
        input_gpkg=streams_gpkg,
        output_gpkg=out_path,
        DA_field="DA_sqmi",
        stream_id_field="stream_id",
        input_layer=input_layer,
        spacing=spacing,
        window=window,
        trans_power=0.52,
        trans_multiplier=100.0,
        overwrite=True,
    )