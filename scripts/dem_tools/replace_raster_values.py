#!/usr/bin/env python3
"""
Replace values in an input DEM within a polygon using values from a replacement DEM.

Behavior
- Uses the input raster as the target grid.
- Reprojects the polygon to the input raster CRS if needed.
- Reprojects/resamples the replacement raster onto the input raster grid.
- Replaces input raster values inside the polygon with replacement raster values.
- Writes a new output raster.

Notes
- This version removes problematic TIFF tiling metadata from the copied profile.
- For DEMs, bilinear resampling is usually appropriate.
"""

from __future__ import annotations

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject


replacement_raster = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\DEMs\Yedlick\REM\HAWS_REM_3ft v1.tif"
input_raster = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\DEMs\Yedlick\yedlick_HAWS_REM.tif"
polygon = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\DEMs\Yedlick\REM\replacement_polygon.gpkg"
output_raster = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\DEMs\Yedlick\REM\yedlick_HAWS_REM_replaced.tif"


def _set_gdal_data_if_missing() -> None:
    """Set GDAL_DATA for common conda Windows installs if not already set."""
    if os.environ.get("GDAL_DATA"):
        return

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        return

    candidate = Path(conda_prefix) / "Library" / "share" / "gdal"
    if candidate.exists():
        os.environ["GDAL_DATA"] = str(candidate)


def _clean_output_profile(profile: dict) -> dict:
    """
    Remove problematic block/tile metadata inherited from the source raster.
    Write striped GeoTIFF unless you explicitly want valid tiling.
    """
    cleaned = profile.copy()

    # Remove inherited tiling/block settings that can trigger RasterBlockError
    for key in ["blockxsize", "blockysize", "tiled", "interleave"]:
        cleaned.pop(key, None)

    cleaned.update(
        driver="GTiff",
        compress="lzw",
        BIGTIFF="IF_SAFER",
    )
    return cleaned


def main(
    input_raster_path: str,
    replacement_raster_path: str,
    polygon_path: str,
    output_raster_path: str,
) -> None:
    _set_gdal_data_if_missing()

    polygon_gdf = gpd.read_file(polygon_path)
    if polygon_gdf.empty:
        raise ValueError("Polygon file contains no features.")

    polygon_gdf = polygon_gdf.loc[
        polygon_gdf.geometry.notnull() & ~polygon_gdf.geometry.is_empty
    ].copy()
    if polygon_gdf.empty:
        raise ValueError("Polygon file contains no valid geometries.")

    if polygon_gdf.crs is None:
        raise ValueError("Polygon file has no CRS defined.")

    with rasterio.open(input_raster_path) as src_in:
        input_profile = src_in.profile.copy()
        input_data = src_in.read()
        input_transform = src_in.transform
        input_crs = src_in.crs
        input_height = src_in.height
        input_width = src_in.width
        input_count = src_in.count
        input_dtype = src_in.dtypes[0]
        input_nodata = src_in.nodata

        if input_crs is None:
            raise ValueError("Input raster has no CRS defined.")

        polygon_gdf = polygon_gdf.to_crs(input_crs)
        polygon_geom = polygon_gdf.union_all()

        replace_mask = geometry_mask(
            [polygon_geom],
            out_shape=(input_height, input_width),
            transform=input_transform,
            invert=True,
            all_touched=False,
        )

        if not np.any(replace_mask):
            raise ValueError("Polygon does not overlap the input raster extent.")

        with rasterio.open(replacement_raster_path) as src_rep:
            if src_rep.crs is None:
                raise ValueError("Replacement raster has no CRS defined.")

            if src_rep.count != input_count:
                raise ValueError(
                    f"Band count mismatch: input has {input_count} band(s), "
                    f"replacement has {src_rep.count} band(s)."
                )

            # Reproject replacement raster to exactly match the input raster grid
            replacement_matched = np.empty(
                (input_count, input_height, input_width),
                dtype=np.dtype(input_dtype),
            )

            fill_value = input_nodata if input_nodata is not None else 0

            for band_idx in range(1, input_count + 1):
                destination = np.full(
                    (input_height, input_width),
                    fill_value,
                    dtype=np.dtype(input_dtype),
                )

                reproject(
                    source=rasterio.band(src_rep, band_idx),
                    destination=destination,
                    src_transform=src_rep.transform,
                    src_crs=src_rep.crs,
                    src_nodata=src_rep.nodata,
                    dst_transform=input_transform,
                    dst_crs=input_crs,
                    dst_nodata=fill_value,
                    resampling=Resampling.bilinear,
                )

                replacement_matched[band_idx - 1] = destination

        output_data = input_data.copy()

        for i in range(input_count):
            rep_band = replacement_matched[i]
            out_band = output_data[i]

            if input_nodata is None:
                valid_replacement = np.ones(rep_band.shape, dtype=bool)
            else:
                valid_replacement = rep_band != input_nodata

            write_mask = replace_mask & valid_replacement
            out_band[write_mask] = rep_band[write_mask]
            output_data[i] = out_band

        output_profile = _clean_output_profile(input_profile)

        output_path = Path(output_raster_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(output_path, "w", **output_profile) as dst:
            dst.write(output_data)

    replaced_pixels = int(np.count_nonzero(replace_mask))
    print(f"Saved output raster:\n{output_raster_path}")
    print(f"Pixels inside polygon: {replaced_pixels}")
    print("Replacement raster was reprojected/resampled to match the input raster grid.")


if __name__ == "__main__":
    main(
        input_raster_path=input_raster,
        replacement_raster_path=replacement_raster,
        polygon_path=polygon,
        output_raster_path=output_raster,
    )