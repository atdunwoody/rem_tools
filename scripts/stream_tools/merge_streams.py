#!/usr/bin/env python
"""
merge_streams.py

Snap two stream centerlines at their closest points (within a search radius,
default 100 m) and merge/dissolve into a single centerline geometry.

Inputs are hard-coded to the two paths provided by the user.

Key implementation notes:
- Each input file is dissolved (unary_union) into a single linework geometry.
- Shapely 2.x: linemerge() cannot be called on a single LineString, so we only
  linemerge MultiLineString/collections.
- Snapping is done by moving ONE vertex on B (the vertex nearest the closest-point
  location) to the closest-point location on A. This keeps the edit local and avoids
  shifting an entire line.
- Output is a single-feature GeoPackage.

Assumptions:
- Both inputs share the same projected CRS and units are meters. If CRS units are
  not meters, change SEARCH_RADIUS_M accordingly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import geopandas as gpd
from shapely.geometry import GeometryCollection, LineString, MultiLineString, Point
from shapely.ops import linemerge, nearest_points, unary_union

# -----------------------------------------------------------------------------
# User paths (requested)
# -----------------------------------------------------------------------------
IN_A = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\DEMs\Trout-Bear Cr\Full length analysis\lower_stream_centerline.gpkg"
IN_B = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\DEMs\Trout-Bear Cr\TB_centerline.gpkg"

SEARCH_RADIUS_M = 100.0

OUT_GPKG = os.path.join(os.path.dirname(IN_A), "centerlines_snapped_merged.gpkg")
OUT_LAYER = "centerline_merged"


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
def _to_merged_line(geom: object) -> LineString | MultiLineString:
    """
    Convert dissolved geometry into LineString/MultiLineString, avoiding Shapely 2.x
    linemerge(LineString) errors.
    """
    if geom is None:
        raise ValueError("Geometry is None.")
    if getattr(geom, "is_empty", False):
        raise ValueError("Empty geometry after dissolve.")

    gt = getattr(geom, "geom_type", None)
    if gt == "LineString":
        return geom  # type: ignore[return-value]
    if gt == "MultiLineString":
        return linemerge(geom)  # type: ignore[arg-type]
    if gt == "GeometryCollection":
        gc: GeometryCollection = geom  # type: ignore[assignment]
        lines = [g for g in gc.geoms if g.geom_type in ("LineString", "MultiLineString")]
        if not lines:
            raise TypeError("GeometryCollection contains no line geometries.")
        return linemerge(unary_union(lines))

    raise TypeError(f"Unexpected geometry type after dissolve: {gt}")


def _read_dissolved_line(gpkg_path: str) -> Tuple[gpd.GeoDataFrame, LineString | MultiLineString]:
    gdf = gpd.read_file(gpkg_path)
    if gdf.empty:
        raise ValueError(f"No features found: {gpkg_path}")

    dissolved = unary_union(gdf.geometry.values)
    merged = _to_merged_line(dissolved)

    if merged.geom_type not in ("LineString", "MultiLineString"):
        raise TypeError(f"Expected LineString/MultiLineString; got {merged.geom_type} from {gpkg_path}")

    return gdf, merged


def _edit_nearest_vertex(
    line: LineString | MultiLineString,
    target_point: Point,
) -> LineString | MultiLineString:
    """
    Move the single vertex (across the whole geometry) that is nearest to target_point
    to exactly target_point.

    For MultiLineString, edits the nearest vertex among all parts.
    """
    tx, ty = target_point.x, target_point.y

    def _best_vertex_in_ls(ls: LineString) -> Tuple[float, int]:
        coords = list(ls.coords)
        # return (min_distance, index)
        best_d = float("inf")
        best_i = -1
        for i, (x, y) in enumerate(coords):
            d = (x - tx) ** 2 + (y - ty) ** 2  # squared distance (faster)
            if d < best_d:
                best_d = d
                best_i = i
        return best_d, best_i

    if line.geom_type == "LineString":
        coords = list(line.coords)
        if len(coords) < 2:
            raise ValueError("LineString has <2 vertices; cannot edit safely.")
        _, i = _best_vertex_in_ls(line)
        coords[i] = (tx, ty)
        return LineString(coords)

    parts = list(line.geoms)
    if not parts:
        raise ValueError("MultiLineString has no parts.")

    # find best (part_index, vertex_index)
    best = (float("inf"), -1, -1)
    for pi, part in enumerate(parts):
        d, vi = _best_vertex_in_ls(part)
        if d < best[0]:
            best = (d, pi, vi)

    _, pi, vi = best
    coords = list(parts[pi].coords)
    if len(coords) < 2:
        raise ValueError("A part of MultiLineString has <2 vertices; cannot edit safely.")
    coords[vi] = (tx, ty)
    parts[pi] = LineString(coords)
    return MultiLineString(parts)


def dissolve_to_single_line(geom: LineString | MultiLineString) -> LineString | MultiLineString:
    merged = linemerge(unary_union(geom))
    if merged.geom_type not in ("LineString", "MultiLineString"):
        raise TypeError(f"Unexpected geometry type after dissolve/merge: {merged.geom_type}")
    return merged


# -----------------------------------------------------------------------------
# Snapping (closest points)
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class SnapResult:
    snapped_a: LineString | MultiLineString
    snapped_b: LineString | MultiLineString
    snapped_distance: float
    a_snap_point: Point
    b_snap_point: Point


def snap_at_closest_points(
    a: LineString | MultiLineString,
    b: LineString | MultiLineString,
    search_radius: float,
) -> SnapResult:
    """
    Compute closest points between geometries A and B, then snap B locally to A by
    moving B's nearest vertex to the closest-point location on A.

    This guarantees a shared coordinate at the snapped location (subject to vertex
    insertion granularity on A, see note below).

    Note: We do not modify A. If you need both geometries to share an explicit vertex
    at the closest-point location, you can optionally densify or split A at that point.
    In most workflows, moving B to A's closest point is sufficient to make them touch.
    """
    pa, pb = nearest_points(a, b)  # pa on A, pb on B
    dist = pa.distance(pb)

    if dist > search_radius:
        raise ValueError(
            f"Closest points are {dist:.3f} CRS units apart, exceeding search radius "
            f"{search_radius:.3f}. Check CRS/units or increase SEARCH_RADIUS_M."
        )

    # Move the nearest vertex on B to pa (closest point on A)
    snapped_b = _edit_nearest_vertex(b, pa)

    return SnapResult(
        snapped_a=a,
        snapped_b=snapped_b,
        snapped_distance=float(dist),
        a_snap_point=pa,
        b_snap_point=pb,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    gdf_a, geom_a = _read_dissolved_line(IN_A)
    gdf_b, geom_b = _read_dissolved_line(IN_B)

    crs_a = gdf_a.crs
    crs_b = gdf_b.crs
    if crs_a is None or crs_b is None:
        raise ValueError("One or both inputs have no CRS defined. Define CRS before running.")
    if crs_a != crs_b:
        raise ValueError(f"CRS mismatch:\n  A: {crs_a}\n  B: {crs_b}\nReproject one input so they match.")

    snap = snap_at_closest_points(geom_a, geom_b, SEARCH_RADIUS_M)

    merged = dissolve_to_single_line(unary_union([snap.snapped_a, snap.snapped_b]))

    out_gdf = gpd.GeoDataFrame(
        {
            "src_a": [os.path.basename(IN_A)],
            "src_b": [os.path.basename(IN_B)],
            "snap_dist": [snap.snapped_distance],
            "snap_ax": [snap.a_snap_point.x],
            "snap_ay": [snap.a_snap_point.y],
            "snap_bx": [snap.b_snap_point.x],
            "snap_by": [snap.b_snap_point.y],
        },
        geometry=[merged],
        crs=crs_a,
    )

    if os.path.exists(OUT_GPKG):
        try:
            os.remove(OUT_GPKG)
        except PermissionError as e:
            raise PermissionError(f"Cannot overwrite output (file is open?): {OUT_GPKG}") from e

    out_gdf.to_file(OUT_GPKG, layer=OUT_LAYER, driver="GPKG")

    print("Wrote:")
    print(f"  {OUT_GPKG}")
    print(f"  layer: {OUT_LAYER}")
    print("Snapping summary (closest points):")
    print(f"  search radius: {SEARCH_RADIUS_M}")
    print(f"  closest-point distance (CRS units): {snap.snapped_distance:.3f}")
    print(f"  A snap point: ({snap.a_snap_point.x:.3f}, {snap.a_snap_point.y:.3f})")
    print(f"  B closest point (pre-snap): ({snap.b_snap_point.x:.3f}, {snap.b_snap_point.y:.3f})")


if __name__ == "__main__":
    main()