"""
Extract elevation-derived points for transect lines.

Methods:
- "min": minimum elevation point (optionally endpoints)
- "median": median elevation point (optionally endpoints)
- "interval": points at a specified spacing along each line (optionally endpoints)

For interval points, DEM is sampled at each point location and stored in "elevation".
"""

import argparse
import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import Point, LineString, MultiLineString
from shapely.ops import linemerge
from sklearn.cluster import KMeans
import os


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _as_single_line(geom):
    """
    Return a LineString suitable for .coords and .interpolate().
    - If geom is LineString: return it
    - If geom is MultiLineString:
        * try linemerge -> may produce LineString
        * if still multipart, pick the longest LineString component
    Returns None if cannot produce a usable line.
    """
    if geom is None or geom.is_empty:
        return None

    gtype = geom.geom_type
    if gtype == "LineString":
        return geom

    if gtype == "MultiLineString":
        merged = linemerge(geom)
        if merged.geom_type == "LineString":
            return merged
        elif merged.geom_type == "MultiLineString":
            parts = list(merged.geoms)
            if not parts:
                return None
            return max(parts, key=lambda g: g.length)

    return None


def _safe_endpoints(line: LineString):
    """
    Get (start_point, end_point) from a LineString.
    """
    coords = list(line.coords)
    if len(coords) < 2:
        return None, None
    return Point(coords[0]), Point(coords[-1])


def _sample_dem_values(src, coords, no_data_value=None):
    """
    Sample DEM and return a masked array if nodata is defined.
    """
    values = np.array([val[0] for val in src.sample(coords)], dtype=float)

    # Mask nodata if defined and numeric
    if no_data_value is not None:
        try:
            if np.isnan(no_data_value):
                pass
            else:
                values = np.ma.masked_equal(values, no_data_value)
        except TypeError:
            # nodata may be non-numeric in unusual cases
            pass

    return values


def _distances_at_interval(length: float, interval: float, include_end: bool = True):
    """
    Return distances [0, interval, 2*interval, ...] up to line length.
    Includes endpoint exactly if include_end=True and not already present.
    """
    if interval <= 0:
        raise ValueError("point_interval must be > 0")

    if length <= 0:
        return np.array([0.0], dtype=float)

    dists = np.arange(0.0, length + 1e-12, interval, dtype=float)

    # Ensure exact endpoint if requested and not already present
    if include_end and (len(dists) == 0 or not np.isclose(dists[-1], length)):
        dists = np.append(dists, length)

    # Guard against tiny floating overshoot
    dists = np.clip(dists, 0.0, length)
    return dists


def _base_attrs(row, idx, include_station=True):
    """
    Common attribute payload copied from input row if fields exist.
    """
    attrs = {
        "centerline_id": row.get("centerline_id", idx),
        "DA_km2": row.get("DA_km2", None),
        "BF_width_Legg_m": row.get("BF_width_Legg_m", None),
        "BF_depth_Legg_m": row.get("BF_depth_Legg_m", None),
        "BF_width_Castro_m": row.get("BF_width_Castro_m", None),
        "BF_depth_Castro_m": row.get("BF_depth_Castro_m", None),
        "BF_width_Beechie_m": row.get("BF_width_Beechie_m", None),
    }
    if include_station:
        attrs["station"] = row.get("station", idx)
    return attrs


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def extract_elevations_along_transect(
    transect_gpkg: str,
    dem_path: str,
    output_gpkg: str,
    flank_min_points: bool = False,
    method: str = "min",
    point_interval: float = None,   # NEW: used by method="interval"
    include_endpoints: bool = True, # NEW: used by method="interval"
):
    if method == "min":
        return extract_min_points(
            transect_gpkg=transect_gpkg,
            dem_path=dem_path,
            output_gpkg=output_gpkg,
            flank_min_points=flank_min_points,
        )
    elif method == "median":
        return extract_median_points(
            transect_gpkg=transect_gpkg,
            dem_path=dem_path,
            output_gpkg=output_gpkg,
            layer_name="median_elev_points",
            flank_points=flank_min_points,
        )
    elif method == "interval":
        if point_interval is None:
            raise ValueError("point_interval must be provided when method='interval'")
        return extract_interval_points(
            transect_gpkg=transect_gpkg,
            dem_path=dem_path,
            output_gpkg=output_gpkg,
            point_interval=point_interval,
            include_endpoints=include_endpoints,
            layer_name="interval_elev_points",
        )
    else:
        raise ValueError(f"Unknown method: {method}\nValid options are 'min', 'median', or 'interval'.")


def extract_min_points(
    transect_gpkg: str,
    dem_path: str,
    output_gpkg: str,
    flank_min_points: bool = False,
):
    print("Extracting minimum elevation points from transects...")
    gdf_lines = gpd.read_file(transect_gpkg)
    crs = gdf_lines.crs

    points = []

    with rasterio.open(dem_path) as src:
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)
        sample_dist = min(res_x, res_y)
        no_data_value = src.nodata

        print(f"Creating points for this many lines: {len(gdf_lines)}")

        for idx, row in gdf_lines.iterrows():
            raw_geom = row.geometry
            line = _as_single_line(raw_geom)

            if line is None or line.is_empty:
                print(f"Skipping row {idx}: empty/invalid or unsupported geometry ({getattr(raw_geom, 'geom_type', None)})")
                continue

            length = line.length
            if length <= 0:
                print(f"Skipping row {idx}: zero-length line")
                continue

            # Sample along the line at raster-resolution spacing
            n_samples = max(int(length / sample_dist) + 1, 2)
            distances = np.linspace(0, length, n_samples)
            sample_pts = [line.interpolate(d) for d in distances]
            coords = [(pt.x, pt.y) for pt in sample_pts]

            values = _sample_dem_values(src, coords, no_data_value=no_data_value)

            if np.ma.is_masked(values) and values.mask.all():
                print(f"Skipping row {idx}: all sampled DEM values are nodata")
                continue

            min_idx = int(values.argmin())
            min_val = float(values[min_idx])
            min_pt = Point(coords[min_idx])

            elev = min_val if min_val > 0 else None

            if flank_min_points:
                start_pt, end_pt = _safe_endpoints(line)
                if start_pt is None or end_pt is None:
                    print(f"Skipping row {idx}: could not determine endpoints")
                    continue

                for geom, pt_type, dist_along in [
                    (min_pt, "min", float(distances[min_idx])),
                    (start_pt, "start", 0.0),
                    (end_pt, "end", float(length)),
                ]:
                    rec = {
                        "geometry": geom,
                        "elevation": elev,
                        "pt_type": pt_type,
                        "dist_along": dist_along,
                    }
                    rec.update(_base_attrs(row, idx, include_station=True))
                    points.append(rec)
            else:
                rec = {
                    "geometry": min_pt,
                    "elevation": elev,
                    "pt_type": "min",
                    "dist_along": float(distances[min_idx]),
                }
                rec.update(_base_attrs(row, idx, include_station=False))
                points.append(rec)

    gdf_pts = gpd.GeoDataFrame(points, crs=crs)

    if not gdf_pts.empty and "elevation" in gdf_pts.columns:
        gdf_pts = gdf_pts.dropna(subset=["elevation"])
        gdf_pts = gdf_pts[gdf_pts["elevation"] > 0]

    gdf_pts.to_file(output_gpkg, driver="GPKG")
    print(f"Written {len(gdf_pts)} points to '{output_gpkg}'")

    return output_gpkg


def extract_median_points(
    transect_gpkg: str,
    dem_path: str,
    output_gpkg: str,
    layer_name: str = "median_elev_points",
    flank_points: bool = True,
):
    """
    Extract median-elevation point (closest sampled point to median) and optionally endpoints.
    """
    gdf_lines = gpd.read_file(transect_gpkg)
    crs = gdf_lines.crs

    points = []

    with rasterio.open(dem_path) as src:
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)
        sample_dist = min(res_x, res_y)
        no_data_value = src.nodata

        for idx, row in gdf_lines.iterrows():
            raw_geom = row.geometry
            line = _as_single_line(raw_geom)

            if line is None or line.is_empty:
                print(f"Skipping row {idx}: empty/invalid or unsupported geometry ({getattr(raw_geom, 'geom_type', None)})")
                continue

            length = line.length
            if length <= 0:
                print(f"Skipping row {idx}: zero-length line")
                continue

            n_samples = max(int(length / sample_dist) + 1, 2)
            distances = np.linspace(0, length, n_samples)
            sample_pts = [line.interpolate(d) for d in distances]
            coords = [(pt.x, pt.y) for pt in sample_pts]

            values = _sample_dem_values(src, coords, no_data_value=no_data_value)

            if np.ma.is_masked(values) and values.mask.all():
                print(f"Skipping row {idx}: all sampled DEM values are nodata")
                continue

            med_val = float(np.ma.median(values))
            filled = np.ma.filled(values, np.nan)
            med_idx = int(np.nanargmin(np.abs(filled - med_val)))
            med_pt = Point(coords[med_idx])

            if flank_points:
                start_pt, end_pt = _safe_endpoints(line)
                if start_pt is None or end_pt is None:
                    print(f"Skipping row {idx}: could not determine endpoints")
                    continue

                for geom, pt_type, dist_along in [
                    (med_pt, "median", float(distances[med_idx])),
                    (start_pt, "start", 0.0),
                    (end_pt, "end", float(length)),
                ]:
                    points.append(
                        {
                            "geometry": geom,
                            "elevation": med_val,
                            "centerline_id": row.get("centerline_id", idx),
                            "station": row.get("station", idx),
                            "pt_type": pt_type,
                            "dist_along": dist_along,
                        }
                    )
            else:
                points.append(
                    {
                        "geometry": med_pt,
                        "elevation": med_val,
                        "centerline_id": row.get("centerline_id", idx),
                        "station": row.get("station", idx),
                        "pt_type": "median",
                        "dist_along": float(distances[med_idx]),
                    }
                )

    gdf_pts = gpd.GeoDataFrame(points, crs=crs)
    gdf_pts.to_file(output_gpkg, driver="GPKG", layer=layer_name)
    print(f"Written {len(gdf_pts)} points to '{output_gpkg}' layer='{layer_name}'")
    return output_gpkg


def extract_interval_points(
    transect_gpkg: str,
    dem_path: str,
    output_gpkg: str,
    point_interval: float,
    include_endpoints: bool = True,
    layer_name: str = "interval_elev_points",
):
    """
    Place points along each line at a specified interval and sample DEM elevation at each point.

    Parameters
    ----------
    point_interval : float
        Spacing in line CRS units (e.g., feet or meters).
    include_endpoints : bool
        If True, ensure start/end points are included exactly.
    """
    print(f"Extracting interval points every {point_interval} map units...")
    gdf_lines = gpd.read_file(transect_gpkg)
    crs = gdf_lines.crs

    points = []

    with rasterio.open(dem_path) as src:
        no_data_value = src.nodata
        print(f"Creating interval points for this many lines: {len(gdf_lines)}")

        for idx, row in gdf_lines.iterrows():
            raw_geom = row.geometry
            line = _as_single_line(raw_geom)

            if line is None or line.is_empty:
                print(f"Skipping row {idx}: empty/invalid or unsupported geometry ({getattr(raw_geom, 'geom_type', None)})")
                continue

            length = line.length
            if length <= 0:
                print(f"Skipping row {idx}: zero-length line")
                continue

            distances = _distances_at_interval(length, point_interval, include_end=include_endpoints)
            sample_pts = [line.interpolate(float(d)) for d in distances]
            coords = [(pt.x, pt.y) for pt in sample_pts]

            values = _sample_dem_values(src, coords, no_data_value=no_data_value)

            # Convert to regular array for per-point handling
            if np.ma.isMaskedArray(values):
                values_filled = np.ma.filled(values, np.nan)
            else:
                values_filled = values

            base = _base_attrs(row, idx, include_station=True)

            for i, (pt, d, val) in enumerate(zip(sample_pts, distances, values_filled)):
                # Optional positive-elevation screen to match your existing behavior
                elev = float(val) if np.isfinite(val) and val > 0 else None

                if i == 0:
                    pt_type = "start"
                elif i == len(sample_pts) - 1 and np.isclose(float(d), float(length)):
                    pt_type = "end"
                else:
                    pt_type = "interval"

                rec = {
                    "geometry": pt,
                    "elevation": elev,
                    "pt_type": pt_type,
                    "dist_along": float(d),
                    "line_length": float(length),
                    "pt_index": int(i),
                    "interval": float(point_interval),
                }
                rec.update(base)
                points.append(rec)

    gdf_pts = gpd.GeoDataFrame(points, crs=crs)

    # Match your current filtering behavior (drop null / non-positive)
    if not gdf_pts.empty and "elevation" in gdf_pts.columns:
        gdf_pts = gdf_pts.dropna(subset=["elevation"])
        gdf_pts = gdf_pts[gdf_pts["elevation"] > 0]

    gdf_pts.to_file(output_gpkg, driver="GPKG", layer=layer_name)
    print(f"Written {len(gdf_pts)} points to '{output_gpkg}' layer='{layer_name}'")
    return output_gpkg


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # default_transect_gpkg = r"C:\L\Lichen\Lichen - Documents\Projects\20250012_Woodard Cr Design (LCEP)\07_GIS\Data\Analysis\REM\Columbia\transects 600ft.gpkg"
    default_transect_gpkg = r"C:\L\Lichen\Lichen - Documents\Projects\20240005_Dry Creek Plan (YN)\07_GIS\Data_Out\HAWS\Confluence HAWS\transects.gpkg"
    default_dem_path = r"C:\L\Lichen\Lichen - Documents\Projects\20240005_Dry Creek Plan (YN)\07_GIS\Data_In\LiDAR 2015\2015_DTM.tif"
    default_output_gpkg = os.path.join(os.path.dirname(default_transect_gpkg), "min_elev_points_stratified.gpkg")

    extract_elevations_along_transect(
        transect_gpkg=default_transect_gpkg,
        dem_path=default_dem_path,
        output_gpkg=default_output_gpkg,
        method="interval",          # "min", "median", or "interval"
        flank_min_points=True,      # used by min/median only
        point_interval=300.0,        # <-- specify spacing here (CRS units)
        include_endpoints=True,     # used by interval only
    )