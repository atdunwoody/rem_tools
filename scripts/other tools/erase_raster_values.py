#!/usr/bin/env python
"""
erase_raster_values.py

Hard-coded to the user-provided inputs:

Polygon (eraser) GPKG:
"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\DEMs\Trout-Bear Cr\temp\eraser poly.gpkg"

Raster to erase (in GPKG):
"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\DEMs\Trout-Bear Cr\Full length analysis\TB_lower_REM_classified.gpkg"

Notes
- This expects the raster dataset inside the raster GeoPackage to be readable by GDAL/rasterio.
- If the raster GPKG contains multiple raster subdatasets, you may need to point rasterio at the specific
  subdataset name (see the error message text if it fails).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask


# -------------------------
# USER INPUTS (hard-coded)
# -------------------------
POLYGON_GPKG = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\DEMs\Trout-Bear Cr\temp\eraser poly.gpkg"

# This is what you provided. It must resolve to a GDAL-readable raster dataset.
RASTER_IN = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\DEMs\Trout-Bear Cr\Full length analysis\TB lower REM classified.tif"

# Output path (same folder as raster input, with suffix)
RASTER_OUT = str(Path(RASTER_IN).with_name(f"{Path(RASTER_IN).stem}_erased.tif"))

# Optional: if your polygon gpkg has multiple layers, set this, else leave None
POLYGON_LAYER: Optional[str] = None

# Optional attribute filter (pandas query syntax), else None
WHERE: Optional[str] = None

# Optional fill value. If None, uses NoData (or SET_NODATA if raster has none)
FILL_VALUE: Optional[float] = None

# Optional: define NoData for output if input has none and FILL_VALUE is None
SET_NODATA: Optional[float] = -9999.0

# For large rasters, keep True
BLOCKWISE: bool = True


def _read_polygons(gpkg_path: str, layer: Optional[str], where: Optional[str]) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(gpkg_path, layer=layer)
    if where:
        gdf = gdf.query(where)

    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    gdf = gdf[gdf.geom_type.isin(["Polygon", "MultiPolygon"])]

    if gdf.empty:
        raise ValueError("No valid Polygon/MultiPolygon features found in the eraser GeoPackage.")
    if gdf.crs is None:
        raise ValueError("Eraser polygon layer has no CRS. Define it before running.")

    return gdf


def erase_raster_values(
    polygon_gpkg: str,
    raster_in: str,
    raster_out: str,
    *,
    polygon_layer: Optional[str] = None,
    where: Optional[str] = None,
    fill_value: Optional[float] = None,
    set_nodata: Optional[float] = None,
    blockwise: bool = True,
) -> None:
    with rasterio.open(raster_in) as src:
        if src.crs is None:
            raise ValueError("Input raster has no CRS. Cannot align polygons reliably.")

        # Load and reproject polygons to raster CRS
        gdf = _read_polygons(polygon_gpkg, polygon_layer, where)
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        geoms: Sequence[object] = list(gdf.geometry.values)

        profile = src.profile.copy()

        existing_nodata = src.nodata
        out_nodata = existing_nodata

        # Decide what to write inside polygons
        if fill_value is None:
            # Use NoData
            if existing_nodata is None and set_nodata is None:
                raise ValueError(
                    "Raster has no NoData defined and FILL_VALUE is None. "
                    "Set SET_NODATA or set FILL_VALUE."
                )
            if existing_nodata is None and set_nodata is not None:
                out_nodata = float(set_nodata)
            fill = out_nodata
        else:
            fill = float(fill_value)
            if set_nodata is not None:
                out_nodata = float(set_nodata)

        profile.update(nodata=out_nodata)

        Path(raster_out).parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(raster_out, "w", **profile) as dst:
            if not blockwise:
                data = src.read()  # (bands, rows, cols)
                mask = geometry_mask(
                    geoms,
                    out_shape=(src.height, src.width),
                    transform=src.transform,
                    invert=True,  # True where polygons cover
                    all_touched=False,
                )
                data[:, mask] = np.asarray(fill, dtype=data.dtype)
                dst.write(data)
            else:
                # Windowed processing (recommended)
                for _, window in src.block_windows(1):
                    transform = src.window_transform(window)
                    out_shape = (window.height, window.width)

                    mask = geometry_mask(
                        geoms,
                        out_shape=out_shape,
                        transform=transform,
                        invert=True,
                        all_touched=False,
                    )

                    block = src.read(window=window)
                    if mask.any():
                        block[:, mask] = np.asarray(fill, dtype=block.dtype)
                    dst.write(block, window=window)


def main() -> None:
    erase_raster_values(
        polygon_gpkg=POLYGON_GPKG,
        raster_in=RASTER_IN,
        raster_out=RASTER_OUT,
        polygon_layer=POLYGON_LAYER,
        where=WHERE,
        fill_value=FILL_VALUE,
        set_nodata=SET_NODATA,
        blockwise=BLOCKWISE,
    )
    print(f"Wrote: {RASTER_OUT}")


if __name__ == "__main__":
    main()