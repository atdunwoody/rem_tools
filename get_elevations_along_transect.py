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

def extract_min_points(transect_gpkg: str, dem_path: str, output_gpkg: str, layer_name: str = "min_elev_points"):
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
            values = [val[0] for val in src.sample(coords)]
            values = np.array(values, dtype=float)

            # Identify minimum
            min_idx = int(np.nanargmin(values))
            min_val = float(values[min_idx])
            min_pt = Point(coords[min_idx])

            # Endpoints
            start_pt = Point(line.coords[0])
            end_pt   = Point(line.coords[-1])

            # Append three points
            for geom in (min_pt, start_pt, end_pt):
                points.append({
                    "geometry": geom,
                    "elevation": min_val,
                    # optionally: "transect_id": row.get("id", idx)
                })

    # Build GeoDataFrame and write out
    gdf_pts = gpd.GeoDataFrame(points, crs=crs)
    gdf_pts.to_file(output_gpkg, driver="GPKG", layer=layer_name)
    print(f"Written {len(gdf_pts)} points to '{output_gpkg}' layer='{layer_name}'")


# Use CLI args if provided, else defaults
if __name__ == '__main__':
    # Default parameters for VSCode debugging
    default_transect_gpkg = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20250006_Wallowa R Remeander (AP)\07_GIS\Data\REM\Transects_250ft.gpkg"
    default_dem_path = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20250006_Wallowa R Remeander (AP)\07_GIS\Data\AP_WallowaSunriseDEM\WallowaSunriseDEM\GRMW_unclipped_1ft_DEM.tif"
    default_output_gpkg = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20250006_Wallowa R Remeander (AP)\07_GIS\Data\REM\min_elev_points.gpkg"
    
    if len(sys.argv) >= 4:
        parser = argparse.ArgumentParser(
            description="Create points at each transect's min-elevation pixel and its endpoints."
        )
        parser.add_argument("transect_gpkg",
                            help="Input GeoPackage with transect lines")
        parser.add_argument("dem_path",
                            help="Input DEM raster file")
        parser.add_argument("output_gpkg",
                            help="Output GeoPackage to write point layer")
        parser.add_argument("--layer-name", default="min_elev_points",
                            help="Name of the output layer (default: min_elev_points)")
        args = parser.parse_args()
        
        extract_min_points(
            transect_gpkg=args.transect_gpkg,
            dem_path=args.dem_path,
            output_gpkg=args.output_gpkg,
            layer_name=args.layer_name
        )
    else:
        extract_min_points(
            transect_gpkg=default_transect_gpkg,
            dem_path=default_dem_path,
            output_gpkg=default_output_gpkg,
            layer_name=None  # Use default layer name if not provided
        )
