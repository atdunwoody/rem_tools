#!/usr/bin/env python3
"""
Burn polygons into an existing raster by assigning a user-specified value
to all raster cells whose centers fall within the polygon footprint.

This version uses paths set directly in the script.

Inputs
------
input_raster:
    Existing raster to modify.
input_polygons:
    Polygon layer in a GeoPackage.
burn_value:
    Value to assign where polygons cover the raster.
output_raster:
    Output raster path.

Notes
-----
- The raster extent, resolution, transform, CRS, dtype, and nodata are preserved.
- If the polygon layer CRS differs from the raster CRS, polygons are reprojected.
- By default, only pixels whose center falls inside a polygon are changed.
  Set all_touched=True to affect any pixel touched by a polygon.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize


# =========================
# User inputs
# =========================
input_raster = r"C:\L\Lichen\Lichen - Documents\Projects\20240005_Dry Creek Plan (YN)\07_GIS\Data_Out\HAWS\Confluence HAWS\HAWS_REM_3ft.tif"
input_polygons = r"C:\L\Lichen\Lichen - Documents\Projects\20240005_Dry Creek Plan (YN)\07_GIS\Data_Out\HAWS\Confluence HAWS\REM edits.gpkg"

# Value to assign to the raster wherever polygons overlap
burn_value = 15

# Output raster
output_raster = r"C:\L\Lichen\Lichen - Documents\Projects\20240005_Dry Creek Plan (YN)\07_GIS\Data_Out\HAWS\Confluence HAWS\HAWS_REM_3ft_edited.tif"

# Rasterization option
all_touched = False
# False = only pixels whose center is inside polygon
# True  = any pixel touched by polygon


def main() -> None:
    input_raster_path = Path(input_raster)
    input_polygons_path = Path(input_polygons)
    output_raster_path = Path(output_raster)

    if not input_raster_path.exists():
        raise FileNotFoundError(f"Input raster not found: {input_raster_path}")
    if not input_polygons_path.exists():
        raise FileNotFoundError(f"Input polygon file not found: {input_polygons_path}")

    # Read polygons
    gdf = gpd.read_file(input_polygons_path)
    if gdf.empty:
        raise ValueError("Polygon layer is empty.")
    if gdf.crs is None:
        raise ValueError("Polygon layer has no CRS defined.")

    with rasterio.open(input_raster_path) as src:
        profile = src.profile.copy()
        raster_crs = src.crs
        raster_transform = src.transform
        raster_shape = (src.height, src.width)
        raster_dtype = src.dtypes[0]
        raster_nodata = src.nodata

        if raster_crs is None:
            raise ValueError("Input raster has no CRS defined.")

        # Reproject polygons if needed
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        # Drop null/empty geometries
        gdf = gdf[~gdf.geometry.isna() & ~gdf.geometry.is_empty].copy()
        if gdf.empty:
            raise ValueError("No valid polygon geometries found after filtering.")

        # Read raster band 1
        arr = src.read(1)

        # Cast burn value to raster dtype
        try:
            typed_burn_value = np.array([burn_value], dtype=raster_dtype)[0]
        except Exception as exc:
            raise ValueError(
                f"burn_value={burn_value!r} is not compatible with raster dtype {raster_dtype!r}"
            ) from exc

        # Create mask of polygon-covered cells
        mask = rasterize(
            [(geom, 1) for geom in gdf.geometry],
            out_shape=raster_shape,
            transform=raster_transform,
            fill=0,
            default_value=1,
            all_touched=all_touched,
            dtype="uint8",
        ).astype(bool)

        # Assign burn value where mask is True
        out_arr = arr.copy()
        out_arr[mask] = typed_burn_value

        # Write output
        output_raster_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_raster_path, "w", **profile) as dst:
            dst.write(out_arr, 1)

    print(f"Saved output raster:\n{output_raster_path}")


if __name__ == "__main__":
    main()