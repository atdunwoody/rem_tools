"""
Extract minimum-elevation points and endpoints for transect lines.

For each line in the input GeoPackage, samples the DEM at
intervals equal to the raster resolution, finds the minimum
elevation and its location, then creates three points:
  - The location of minimum elevation
  - The start vertex of the line
  - The end vertex of the line

All points carry the minimum elevation under the field "elevation".
Results are saved to a new layer in an output GeoPackage.
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
        # Try to merge connected parts into one line
        merged = linemerge(geom)
        if merged.geom_type == "LineString":
            return merged
        elif merged.geom_type == "MultiLineString":
            parts = list(merged.geoms)
            if not parts:
                return None
            return max(parts, key=lambda g: g.length)

    # Not supported
    return None


def _safe_endpoints(line: LineString):
    """
    Get (start_point, end_point) from a LineString.
    """
    coords = list(line.coords)
    if len(coords) < 2:
        return None, None
    return Point(coords[0]), Point(coords[-1])

# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def extract_elevations_along_transect(
    transect_gpkg: str,
    dem_path: str,
    output_gpkg: str,
    flank_min_points: bool = False,
    method: str = "min",
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
    else:
        raise ValueError(f"Unknown method: {method}\nValid options are 'min' or 'median'.")


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

            # Sample along the line
            n_samples = max(int(length / sample_dist) + 1, 2)
            distances = np.linspace(0, length, n_samples)
            sample_pts = [line.interpolate(d) for d in distances]
            coords = [(pt.x, pt.y) for pt in sample_pts]

            values = np.array([val[0] for val in src.sample(coords)], dtype=float)

            # Mask nodata if defined
            if no_data_value is not None and not np.isnan(no_data_value):
                values = np.ma.masked_equal(values, no_data_value)

            # If everything is masked, skip
            if np.ma.is_masked(values) and values.mask.all():
                print(f"Skipping row {idx}: all sampled DEM values are nodata")
                continue

            # Min (safe for masked arrays)
            min_idx = int(values.argmin())
            min_val = float(values[min_idx])
            min_pt = Point(coords[min_idx])

            elev = min_val if min_val > 0 else None

            if flank_min_points:
                start_pt, end_pt = _safe_endpoints(line)
                if start_pt is None or end_pt is None:
                    print(f"Skipping row {idx}: could not determine endpoints")
                    continue

                for geom in (min_pt, start_pt, end_pt):
                    points.append(
                        {
                            "geometry": geom,
                            "elevation": elev,
                            "centerline_id": row.get("centerline_id", idx),
                            "station": row.get("station", idx),
                            "DA_km2": row.get("DA_km2", None),
                            "BF_width_Legg_m": row.get("BF_width_Legg_m", None),
                            "BF_depth_Legg_m": row.get("BF_depth_Legg_m", None),
                            "BF_width_Castro_m": row.get("BF_width_Castro_m", None),
                            "BF_depth_Castro_m": row.get("BF_depth_Castro_m", None),
                            "BF_width_Beechie_m": row.get("BF_width_Beechie_m", None),
                        }
                    )
            else:
                points.append(
                    {
                        "geometry": min_pt,
                        "elevation": elev,
                        "centerline_id": row.get("centerline_id", idx),
                        "DA_km2": row.get("DA_km2", None),
                        "BF_width_Legg_m": row.get("BF_width_Legg_m", None),
                        "BF_depth_Legg_m": row.get("BF_depth_Legg_m", None),
                        "BF_width_Castro_m": row.get("BF_width_Castro_m", None),
                        "BF_depth_Castro_m": row.get("BF_depth_Castro_m", None),
                        "BF_width_Beechie_m": row.get("BF_width_Beechie_m", None),
                    }
                )

    gdf_pts = gpd.GeoDataFrame(points, crs=crs)

    # Drop null/negative elevations (guard if column missing / empty)
    if not gdf_pts.empty and "elevation" in gdf_pts.columns:
        gdf_pts = gdf_pts.dropna(subset=["elevation"])
        gdf_pts = gdf_pts[gdf_pts["elevation"] > 0]

    # gdf_pts = cluster_points(gdf_pts, n_clusters=11, new_field="cluster_id")
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

    - Handles MultiLineString safely (merge/longest-part)
    - Masks nodata (if defined)
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

            values = np.array([val[0] for val in src.sample(coords)], dtype=float)

            # Mask nodata if defined
            if no_data_value is not None and not np.isnan(no_data_value):
                values = np.ma.masked_equal(values, no_data_value)

            # If everything is masked, skip
            if np.ma.is_masked(values) and values.mask.all():
                print(f"Skipping row {idx}: all sampled DEM values are nodata")
                continue

            # Median of unmasked values
            med_val = float(np.ma.median(values))

            # Find sample point closest to that median (use filled array for abs diff)
            filled = np.ma.filled(values, np.nan)
            med_idx = int(np.nanargmin(np.abs(filled - med_val)))
            med_pt = Point(coords[med_idx])

            if flank_points:
                start_pt, end_pt = _safe_endpoints(line)
                if start_pt is None or end_pt is None:
                    print(f"Skipping row {idx}: could not determine endpoints")
                    continue

                for geom in (med_pt, start_pt, end_pt):
                    points.append(
                        {
                            "geometry": geom,
                            "elevation": med_val,
                            "centerline_id": row.get("centerline_id", idx),
                            "station": row.get("station", idx),
                        }
                    )
            else:
                points.append(
                    {
                        "geometry": med_pt,
                        "elevation": med_val,
                        "centerline_id": row.get("centerline_id", idx),
                        "station": row.get("station", idx),
                    }
                )

    gdf_pts = gpd.GeoDataFrame(points, crs=crs)
    gdf_pts.to_file(output_gpkg, driver="GPKG", layer=layer_name)
    print(f"Written {len(gdf_pts)} points to '{output_gpkg}' layer='{layer_name}'")
    return output_gpkg


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    default_transect_gpkg = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\REMs\GGL\transects clipped.gpkg"
    default_dem_path = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\REMs\Topography\2016_USDA_DEM.tif"
    default_output_gpkg = os.path.join(os.path.dirname(default_transect_gpkg), "median_elev_points.gpkg")

    extract_elevations_along_transect(
        transect_gpkg=default_transect_gpkg,
        dem_path=default_dem_path,
        output_gpkg=default_output_gpkg,
        method="median",          # "min" for HAWS or "median" for GGL
        flank_min_points=True,    # include endpoints too
    )
