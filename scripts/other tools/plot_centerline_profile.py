from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from shapely.geometry import LineString




def plot_dem_profiles(
    dem_path: str | Path,
    line_gpkg: str | Path,
    line_layer: Optional[str] = None,
    sample_spacing: float = 1.0,
    title: str = "DEM Profiles",
) -> None:
    """
    Plot separate and cumulative elevation profiles of a DEM along line features.

    Parameters
    ----------
    dem_path : str or Path
        Path to the DEM raster.
    line_gpkg : str or Path
        Path to the GeoPackage containing line features.
    line_layer : str, optional
        Name of the layer in the GeoPackage. If None, the first layer is used.
    sample_spacing : float, default=1.0
        Spacing (map units, e.g. meters) between sample points along each line.
    title : str, default="DEM Profiles"
        Title for the cumulative profile plot.
    """
    dem_path, line_gpkg = Path(dem_path), Path(line_gpkg)

    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")
    if not line_gpkg.exists():
        raise FileNotFoundError(f"GeoPackage not found: {line_gpkg}")

    # Load lines
    gdf = gpd.read_file(line_gpkg, layer=line_layer)
    if gdf.empty:
        raise ValueError("No line features found in the input GeoPackage.")

    # Open DEM
    with rasterio.open(dem_path) as src:
        crs_raster = src.crs
        crs_lines = gdf.crs

        if crs_lines != crs_raster:
            gdf = gdf.to_crs(crs_raster)

        all_distances = []
        all_elevations = []
        offset = 0.0

        for idx, geom in gdf.iterrows():
            if not isinstance(geom.geometry, LineString):
                continue

            # Sample along line
            line: LineString = geom.geometry
            num_samples = max(2, int(line.length // sample_spacing) + 1)
            distances = np.linspace(0, line.length, num_samples)
            points = [line.interpolate(d) for d in distances]

            coords = [(pt.x, pt.y) for pt in points]
            elev = list(src.sample(coords))
            elev = [val[0] if val[0] != src.nodata else np.nan for val in elev]

            # Plot individual profile
            plt.figure(figsize=(8, 4))
            plt.plot(distances, elev, label=f"Line {idx}")
            plt.xlabel("Distance (m)")
            plt.ylabel("Elevation (m)")
            plt.title(f"Profile along line {idx}")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()

            # Append to cumulative
            all_distances.extend(distances + offset)
            all_elevations.extend(elev)
            offset = all_distances[-1]

    # Cumulative plot
    plt.figure(figsize=(10, 5))
    plt.plot(all_distances, all_elevations, "-k")
    plt.xlabel("Cumulative Distance (m)")
    plt.ylabel("Elevation (m)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


# ------------------ Example usage ------------------ #
if __name__ == "__main__":
    dem = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20250005_Wallowa R Remeander (AP)\07_GIS\Data\LiDAR\bathymetry_dem.tif"
    centerline_gpkg = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20250005_Wallowa R Remeander (AP)\08_Documents\Geomorphic Assessment\Potential_ref_reaches\Ref Reach Centerlines.gpkg"

    # Adjust `line_layer` and optional `line_filter` to target a specific reference-reach line
    plot_dem_profiles(
        dem_path=dem,
        line_gpkg=centerline_gpkg,
    )