import geopandas as gpd
from shapely.ops import voronoi_diagram
import random
from shapely.geometry import Point
import os

import geopandas as gpd
from shapely.ops import voronoi_diagram, unary_union

def create_clipped_voronoi(
    points_gpkg_path: str,
    aoi_gpkg_path: str,
    output_gpkg_path: str,
) -> gpd.GeoDataFrame:
    """
    Read points and AOI from GeoPackages, generate Voronoi polygons for the points
    clipped to the AOI, copy all original point fields into the output, and save
    to a new GeoPackage.

    Parameters
    ----------
    points_gpkg_path : str
        Path to the input GeoPackage containing point features.
    aoi_gpkg_path : str
        Path to the input GeoPackage containing the AOI polygon(s).
    output_gpkg_path : str
        Path to the output GeoPackage to create/overwrite.

    Returns
    -------
    vor_gdf : geopandas.GeoDataFrame
        The clipped Voronoi polygons, with all input point attributes.
    """
    print(f"Creating Voronoi polygons from points in {points_gpkg_path}...")
    # 1. Read inputs
    pts = gpd.read_file(points_gpkg_path)
    aoi = gpd.read_file(aoi_gpkg_path)

    # 2. Harmonize CRS
    if pts.crs != aoi.crs:
        aoi = aoi.to_crs(pts.crs)

    # 3. Union AOI & Points into single geometries
    aoi_union = aoi.union_all()
    pts_union = pts.geometry.union_all()

    # 4. Compute full Voronoi diagram and clip to AOI
    raw_vor = voronoi_diagram(pts_union, envelope=aoi_union)
    clipped_cells = [
        cell.intersection(aoi_union)
        for cell in raw_vor.geoms
        if not cell.is_empty
    ]

    # 5. Build GeoDataFrame of clipped cells
    vor_gdf = gpd.GeoDataFrame(geometry=clipped_cells, crs=pts.crs)

    # 6. Spatial‐join to bring in all point attributes
    #    'intersects' is used in case points lie on shared edges
    vor_gdf = (
        gpd.sjoin(vor_gdf, pts, how="left", predicate="intersects")
        .drop(columns=["index_right"])
    )

    # 7. (Optional) Add a sequential ID for each cell
    vor_gdf.insert(0, "cell_id", range(1, len(vor_gdf) + 1))

    # 8. Write to GeoPackage (overwrites any existing file)
    vor_gdf.to_file(output_gpkg_path, driver="GPKG")
    print(f"[✔] Created Voronoi polygons GeoPackage: {output_gpkg_path}")
    return vor_gdf

import geopandas as gpd
import numpy as np
import random
from shapely.geometry import Point
from shapely.ops import triangulate
from shapely.prepared import prep

def create_random_points_within_polygons(
    polys_gpkg: str,
    rand_points_gpkg: str,
    num_points: int = 20
) -> gpd.GeoDataFrame:
    """
    Like the triangle‑sampling version, but skips or repairs bad geometries
    so you don’t get IndexErrors on empty triangulations.
    """
    polys = gpd.read_file(polys_gpkg)
    records = []

    for idx, row in polys.iterrows():
        poly = row.geometry
        attrs = row.drop(labels=polys.geometry.name).to_dict()

        # 1) Repair invalid geometries
        if not poly.is_valid:
            poly = poly.buffer(0)
        # 2) Skip empties or zero-area
        if poly.is_empty or poly.area == 0:
            print(f"[⚠] Skipping feature {idx}: empty or zero-area")
            continue

        # 3) Triangulate (or fallback to convex hull)
        triangles = triangulate(poly)
        if not triangles:
            # fallback: try convex hull
            hull = poly.convex_hull
            triangles = triangulate(hull)
            if not triangles:
                print(f"[⚠] No triangles for feature {idx}, skipping")
                continue

        areas = np.array([t.area for t in triangles], dtype=float)
        cum_areas = np.cumsum(areas)
        total_area = cum_areas[-1]
        prep_poly = prep(poly)
        count = 0

        while count < num_points:
            r = random.random() * total_area
            tri_idx = np.searchsorted(cum_areas, r)
            pt = _random_point_in_triangle(triangles[tri_idx])
            # should always hit, but double‑check
            if prep_poly.contains(pt):
                rec = attrs.copy()
                rec['geometry'] = pt
                records.append(rec)
                count += 1

    pts_gdf = gpd.GeoDataFrame(records, crs=polys.crs)
    pts_gdf.to_file(rand_points_gpkg, driver="GPKG")
    print(f"[✔] Created {len(records)} random points in {rand_points_gpkg}")
    return pts_gdf

def _random_point_in_triangle(tri):
    # helper to sample uniformly from a triangle
    coords = list(tri.exterior.coords)[:3]
    (x1, y1), (x2, y2), (x3, y3) = coords
    r1, r2 = random.random(), random.random()
    if r1 + r2 > 1:
        r1, r2 = 1 - r1, 1 - r2
    x = x1 + r1*(x2 - x1) + r2*(x3 - x1)
    y = y1 + r1*(y2 - y1) + r2*(y3 - y1)
    return Point(x, y)


points_gpkg = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\REM\Voronoi Method\20250725\min_elev_points_100m.gpkg"
aoi_gpkg = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\REM\grmw_vectors\Indicies\Grande_Ronde_D1_Boundary.shp"
voronoi_polys_gpkg = os.path.join(os.path.dirname(points_gpkg), "min_elev_voronoi_polys.gpkg")
rand_points_gpkg = os.path.join(os.path.dirname(points_gpkg), "min_elev_random_points_py.gpkg")

# create_clipped_voronoi(
#     points_gpkg,
#     aoi_gpkg,
#     voronoi_polys_gpkg
# )
create_random_points_within_polygons(
    voronoi_polys_gpkg,
    rand_points_gpkg,
    num_points=20
)