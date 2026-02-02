#!/usr/bin/env python3
import os
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union, transform as shp_transform
from pyproj import Transformer


def _safe_layer_name(name: str, max_len: int = 60) -> str:
    """
    GPKG layer name: keep it simple + portable.
    - letters, numbers, underscore only
    - no leading digit
    - length-limited
    """
    base = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    base = re.sub(r"_+", "_", base).strip("_")
    if not base:
        base = "layer"
    if base[0].isdigit():
        base = f"r_{base}"
    return base[:max_len]


def raster_valid_area_polygon(raster_path: Union[str, Path]) -> Tuple[Optional[object], object]:
    """
    Returns (geom, crs) where geom is a single (multi)polygon outlining
    all cells with value > 0 in band 1 (excluding nodata/masked).
    """
    raster_path = str(raster_path)
    with rasterio.open(raster_path) as src:
        band1 = src.read(1, masked=True)  # masked for nodata / internal mask

        # Valid where: not masked AND value > 0
        valid_mask = (~band1.mask) & (band1.filled(0) > 0)

        if int(valid_mask.sum()) == 0:
            return None, src.crs

        geoms = []
        for geom_mapping, value in shapes(
            valid_mask.astype(np.uint8),
            mask=valid_mask,
            transform=src.transform,
        ):
            if value == 1:
                geoms.append(shape(geom_mapping))

        if not geoms:
            return None, src.crs

        # One outline per raster (may be MultiPolygon)
        return unary_union(geoms), src.crs


def reproject_geom(geom, src_crs, dst_crs):
    if src_crs == dst_crs:
        return geom
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return shp_transform(transformer.transform, geom)


def footprints_from_rasters_to_layers(
    raster_paths: Sequence[Union[str, Path]],
    out_gpkg: Union[str, Path],
    simplify_tolerance: Optional[float] = None,
    target_crs: Optional[Union[str, object]] = None,
) -> None:
    """
    Polygonize each raster by valid cells (value > 0), and write each result
    as its own layer in the GeoPackage.
    """
    out_gpkg = str(out_gpkg)
    out_dir = os.path.dirname(out_gpkg)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Decide output CRS: forced target_crs, else first raster's CRS encountered
    out_crs = target_crs

    written = 0
    for rp in raster_paths:
        rp = Path(rp)
        print(f"Processing: {rp}")

        geom, crs = raster_valid_area_polygon(rp)
        if geom is None or geom.is_empty:
            print("  -> No valid cells > 0 found. Skipping.")
            continue

        if out_crs is None:
            out_crs = crs

        if crs != out_crs:
            print(f"  -> Reprojecting footprint from {crs} to {out_crs}")
            geom = reproject_geom(geom, crs, out_crs)

        if simplify_tolerance is not None:
            geom = geom.simplify(simplify_tolerance, preserve_topology=True)

        layer_name = _safe_layer_name(rp.stem)
        gdf = gpd.GeoDataFrame(
            [{"raster_path": str(rp), "raster_name": rp.name, "geometry": geom}],
            crs=out_crs,
        )

        # Write one layer per raster (overwrite that layer if it already exists)
        gdf.to_file(out_gpkg, layer=layer_name, driver="GPKG")
        print(f"  -> Wrote layer: {layer_name}")
        written += 1

    if written == 0:
        raise RuntimeError("No layers written (did all rasters have no valid cells > 0?).")

    print(f"\nDone. Wrote {written} layer(s) to: {out_gpkg}")


if __name__ == "__main__":
    rasters = [
        r"C:\L\OneDrive - Lichen\Documents\Projects\SFToutle\Tree Height Analysis\STHD\2023_dsm.tif",
        r"C:\L\OneDrive - Lichen\Documents\Projects\SFToutle\Tree Height Analysis\STHD\2024_dsm.tif",
        r"C:\L\OneDrive - Lichen\Documents\Projects\SFToutle\Tree Height Analysis\STHD\2006_dsm.tif",
        r"C:\L\OneDrive - Lichen\Documents\Projects\SFToutle\Tree Height Analysis\STHD\2017_dsm.tif",
        r"C:\L\OneDrive - Lichen\Documents\Projects\SFToutle\Tree Height Analysis\STHD\2019_dsm.tif",
    ]

    out_gpkg = r"C:\L\OneDrive - Lichen\Documents\Projects\SFToutle\Tree Height Analysis\STHD\raster_footprints.gpkg"

    # Option A: force everything into EPSG:2927 (WA North ftUS)
    # footprints_from_rasters_to_layers(rasters, out_gpkg, simplify_tolerance=None, target_crs="EPSG:2927")

    # Option B: keep each raster in first raster CRS (reproject others to match)
    footprints_from_rasters_to_layers(rasters, out_gpkg, simplify_tolerance=None, target_crs=None)
