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
from shapely.geometry import Point
import sys
from sklearn.cluster import KMeans
import os


def cluster_points(
    gdf: gpd.GeoDataFrame,
    n_clusters: int = 11,
    new_field: str = "cluster_id",):
    """
    Reads points from `input_gpkg_path`/`layer`, clusters them into `n_clusters` groups
    based on XY proximity (KMeans), writes to `output_gpkg_path` with a new integer field.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame containing point geometries.
    output_gpkg_path : str
        Path to output GeoPackage (will be created or overwritten).
    n_clusters : int
        Number of clusters to create.
    new_field : str
        Name of the new integer field to hold cluster IDs (1..n_clusters).
    """

    # 2. Reproject (if geographic, to a suitable projected CRS)
    if gdf.crs.is_geographic:
        # choose an appropriate local projection; here we use UTM zone of the centroid
        centroid = gdf.unary_union.centroid
        utm_crs = f"+proj=utm +zone={int((centroid.x + 180)//6)+1} +datum=WGS84 +units=m +no_defs"
        gdf = gdf.to_crs(utm_crs)

    # 3. Extract coordinates
    coords = np.vstack([gdf.geometry.x, gdf.geometry.y]).T

    # 4. Cluster
    model = KMeans(n_clusters=n_clusters, random_state=0)
    labels = model.fit_predict(coords)

    # 5. Assign labels 1..n_clusters
    gdf[new_field] = labels + 1

    return gdf

def extract_min_points(transect_gpkg: str,
                       dem_path: str,
                       output_gpkg: str,
                       flank_min_points: bool = False):
    # Read input transects
    print("Extracting minimum elevation points from transects...")
    gdf_lines = gpd.read_file(transect_gpkg)
    crs = gdf_lines.crs

    points = []

    with rasterio.open(dem_path) as src:
        # DEM pixel size
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)
        sample_dist = min(res_x, res_y)
        no_data_value = src.nodata
        print(f"Creating points for this many lines: {len(gdf_lines)}")

        for idx, row in gdf_lines.iterrows():
            line = row.geometry

            # Skip invalid or missing geometries
            if line is None or line.is_empty:
                print(f"Skipping row {idx}: empty or null geometry")
                continue

            if line.geom_type not in ("LineString", "MultiLineString"):
                print(f"Skipping row {idx}: geometry type {line.geom_type} not supported")
                continue

            length = line.length
            if length == 0:
                print(f"Skipping row {idx}: zero-length line")
                continue

            # Number of sample points along the line
            n_samples = max(int(length / sample_dist) + 1, 2)

            # Generate equidistant points along the line
            distances = np.linspace(0, length, n_samples)
            sample_pts = [line.interpolate(d) for d in distances]
            coords = [(pt.x, pt.y) for pt in sample_pts]

            # Sample the DEM
            values = np.array([val[0] for val in src.sample(coords)], dtype=float)
            # Mask no-data values
            values = np.ma.masked_equal(values, no_data_value)
            # Identify minimum
            min_idx = int(np.nanargmin(values))
            min_val = float(values[min_idx])
            min_pt = Point(coords[min_idx])

            # Decide whether to keep elevation
            elev = min_val if min_val > 0 else None

            if flank_min_points:
                # Endpoints
                start_pt = Point(line.coords[0])
                end_pt   = Point(line.coords[-1])

                # Append three points, elevation only if positive
                for geom in (min_pt, start_pt, end_pt):
                    points.append({
                        "geometry":       geom,
                        "elevation":      elev,
                        "centerline_id":  row.get("centerline_id", idx),
                        "station":        row.get("station", idx),
                        "DA_km2": row.get("DA_km2", None),
                        "BF_width_Legg_m": row.get("BF_width_Legg_m", None),
                        "BF_depth_Legg_m": row.get("BF_depth_Legg_m", None),
                        "BF_width_Castro_m": row.get("BF_width_Castro_m", None),
                        "BF_depth_Castro_m": row.get("BF_depth_Castro_m", None),
                        "BF_width_Beechie_m": row.get("BF_width_Beechie_m", None),
                    })
            else:
                # Append only the minimum point
                points.append({
                    "geometry":       min_pt,
                    "elevation":      elev,
                    "centerline_id":  row.get("centerline_id", idx),
                    "DA_km2": row.get("DA_km2", None),
                    "BF_width_Legg_m": row.get("BF_width_Legg_m", None),
                    "BF_depth_Legg_m": row.get("BF_depth_Legg_m", None),
                    "BF_width_Castro_m": row.get("BF_width_Castro_m", None),
                    "BF_depth_Castro_m": row.get("BF_depth_Castro_m", None),
                    "BF_width_Beechie_m": row.get("BF_width_Beechie_m", None),
                })

    # Build GeoDataFrame and write out
    gdf_pts = gpd.GeoDataFrame(points, crs=crs)
    
    # Drop any points with negative elevation
    gdf_pts = gdf_pts[gdf_pts["elevation"] > 0]
    #gdf_pts = cluster_points(gdf_pts, n_clusters=11, new_field="cluster_id")
    gdf_pts.to_file(output_gpkg, driver="GPKG")
    print(f"Written {len(gdf_pts)} points to '{output_gpkg}'")
    
    return output_gpkg

def extract_median_points(transect_gpkg: str,
                          dem_path: str,
                          output_gpkg: str,
                          layer_name: str = "median_elev_points"):
    """
    Extract median-elevation points and endpoints for transect lines.

    For each line in the input GeoPackage, samples the DEM at
    intervals equal to the raster resolution, finds the median
    elevation and its location, then creates three points:
      - The location of the median elevation
      - The start vertex of the line
      - The end vertex of the line

    All points carry the median elevation under the field "elevation".
    Results are saved to a new layer in an output GeoPackage.
    """
    # Read input transects
    gdf_lines = gpd.read_file(transect_gpkg)
    crs = gdf_lines.crs

    points = []

    with rasterio.open(dem_path) as src:
        # DEM pixel size
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)
        sample_dist = min(res_x, res_y)

        for idx, row in gdf_lines.iterrows():
            line = row.geometry
            length = line.length

            # Number of sample points along the line
            n_samples = max(int(length / sample_dist) + 1, 2)

            # Generate equidistant points along the line
            distances = np.linspace(0, length, n_samples)
            sample_pts = [line.interpolate(d) for d in distances]
            coords = [(pt.x, pt.y) for pt in sample_pts]

            # Sample the DEM
            values = np.array([val[0] for val in src.sample(coords)], dtype=float)

            # Compute median elevation
            med_val = float(np.nanmedian(values))

            # Find the sample point closest to that median
            med_idx = int(np.nanargmin(np.abs(values - med_val)))
            med_pt = Point(coords[med_idx])

            # Endpoints
            start_pt = Point(line.coords[0])
            end_pt   = Point(line.coords[-1])

            # Append three points (all carry the transect's median elevation)
            for geom in (med_pt, start_pt, end_pt):
                points.append({
                    "geometry": geom,
                    "elevation": med_val,
                    # optionally: "transect_id": row.get("id", idx)
                })

    # Build GeoDataFrame and write out
    gdf_pts = gpd.GeoDataFrame(points, crs=crs)
    gdf_pts.to_file(output_gpkg, driver="GPKG", layer=layer_name)
    print(f"Written {len(gdf_pts)} points to '{output_gpkg}' layer='{layer_name}'")
    return output_gpkg

if __name__ == '__main__':
    default_transect_gpkg = r"C:\L\OneDrive - Lichen\Documents\Projects\SF Toutle Brownell\REM\loch_trouble\transects_bendy_smooth_30.0m_200.0m.gpkg"
    default_dem_path = r"C:\L\OneDrive - Lichen\Documents\Projects\SF Toutle Brownell\REM\loch_trouble\2025_dem_3ft.tif"
    default_output_gpkg = os.path.join(os.path.dirname(default_transect_gpkg), "min_elev_points_loch_trouble_2025.gpkg")

    extract_min_points(
        transect_gpkg=default_transect_gpkg,
        dem_path=default_dem_path,
        output_gpkg=default_output_gpkg,
        flank_min_points=True
    )
