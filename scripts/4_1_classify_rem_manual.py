from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape


def _choose_int_dtype(max_class: int):
    """Pick a compact integer dtype for class raster."""
    if max_class <= 255:
        return rasterio.uint8
    if max_class <= 65535:
        return rasterio.uint16
    return rasterio.uint32


def _format_bin_label(low: float, high: float, is_last: bool) -> str:
    """
    Human-readable label for bin edges.
    We use [low, high) for all but the last bin, and [low, high] for the last.
    """
    if is_last:
        return f"[{low:g}, {high:g}]"
    return f"[{low:g}, {high:g})"


def classify_raster_manual_bins(
    in_raster_path: str,
    output_class_raster_path: str,
    output_polygons_path: str,
    *,
    # Manual class edges (must be sorted ascending; defines bins between consecutive edges)
    class_edges: Sequence[float],
    # Raster / polygon output settings
    out_nodata: int = 0,
    polygon_driver: str = "GPKG",
    polygon_layer: str = "classes",
    dissolve_polygons: bool = True,
) -> None:
    """
    Classify a single-band raster using manual bins defined by class_edges.

    Manual binning via class_edges:
      class_edges = [e0, e1, ..., eN] creates N bins:
        class 1: e0 <= value < e1
        class 2: e1 <= value < e2
        ...
        class N: e{N-1} <= value <= eN   (last bin inclusive)

    Values outside [min(class_edges), max(class_edges)] are left as nodata/unclassified (out_nodata).

    Nodata rules:
      - Input nodata/masked -> output nodata
      - NaN/inf -> output nodata

    Outputs:
      1) Classified raster (integer classes, nodata=out_nodata)
      2) Polygon file with fields:
         - 'class_id' (int)
         - 'ClassRange' (string label for bin)
    """
    edges = [float(e) for e in class_edges]
    if len(edges) < 2:
        raise ValueError("class_edges must contain at least two values (e.g., [-2, -1, 0, 1, 2]).")
    if sorted(edges) != edges:
        raise ValueError("class_edges must be sorted ascending (e.g., [-2.6, -2, -1.5, ..., 2]).")

    n_classes = len(edges) - 1
    dtype = _choose_int_dtype(n_classes)

    print(f"[CLASSIFY] Raster: {in_raster_path}")
    print(f"[CLASSIFY] Class edges: {edges}  -> {n_classes} classes")
    print(f"[OUT] Class raster: {output_class_raster_path}")
    print(f"[OUT] Polygons: {output_polygons_path} (layer={polygon_layer})")

    # -------------------------
    # Read raster
    # -------------------------
    with rasterio.open(in_raster_path) as src:
        arr = src.read(1, masked=True)  # masked array
        meta = src.meta.copy()
        crs = src.crs
        transform = src.transform
        shape_hw = src.shape

    data = arr.data.astype("float32", copy=False)
    valid = (~arr.mask) & np.isfinite(data)

    # -------------------------
    # Classify using manual edges
    # -------------------------
    cls = np.full(shape_hw, out_nodata, dtype=np.dtype(dtype))

    in_range = valid & (data >= edges[0]) & (data <= edges[-1])

    for i in range(n_classes):
        lo = edges[i]
        hi = edges[i + 1]
        is_last = (i == n_classes - 1)
        if is_last:
            m = in_range & (data >= lo) & (data <= hi)
        else:
            m = in_range & (data >= lo) & (data < hi)
        cls[m] = i + 1  # class IDs start at 1

    # -------------------------
    # Write class raster
    # -------------------------
    out_meta = meta.copy()
    out_meta.update(
        dtype=dtype,
        count=1,
        nodata=out_nodata,
        compress="lzw",
    )

    out_dir = os.path.dirname(output_class_raster_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(output_class_raster_path, "w", **out_meta) as dst:
        dst.write(cls, 1)

    print(f"[✔] Wrote class raster ({n_classes} classes). Outside range -> nodata")
    print(f"    Range classified: [{edges[0]:g}, {edges[-1]:g}]  nodata={out_nodata}")

    # -------------------------
    # Polygonize
    # -------------------------
    mask = (cls != out_nodata)

    geoms = []
    vals = []
    for geom, val in shapes(cls, mask=mask, transform=transform):
        v = int(val)
        if v == out_nodata:
            continue
        geoms.append(shape(geom))
        vals.append(v)

    if len(geoms) == 0:
        raise RuntimeError("No polygons were created (all nodata or empty mask).")

    gdf = gpd.GeoDataFrame({"class_id": vals}, geometry=geoms, crs=crs)

    # Labels
    class_to_label: dict[int, str] = {}
    for i in range(n_classes):
        lo = edges[i]
        hi = edges[i + 1]
        is_last = (i == n_classes - 1)
        class_to_label[i + 1] = _format_bin_label(lo, hi, is_last)

    gdf["ClassRange"] = gdf["class_id"].map(class_to_label)

    if dissolve_polygons:
        gdf = (
            gdf.dissolve(by="class_id", as_index=False)
               .merge(
                   gpd.GeoDataFrame(
                       {
                           "class_id": list(class_to_label.keys()),
                           "ClassRange": list(class_to_label.values()),
                       }
                   ),
                   on="class_id",
                   how="left",
               )
        )

    out_poly_dir = os.path.dirname(output_polygons_path)
    if out_poly_dir:
        os.makedirs(out_poly_dir, exist_ok=True)

    gdf.to_file(output_polygons_path, layer=polygon_layer, driver=polygon_driver)
    print(f"[✔] Wrote polygons: {output_polygons_path} (features={len(gdf)})")


if __name__ == "__main__":
    in_raster = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\REMs\GGL\GGL_REM_1m.tif"

    out_class_raster = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\REMs\GGL\GGL Classified 4 class.tif"
    out_polygons = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\REMs\GGL\GGL Classified 4 class.gpkg"

    # class_edges = [-3, -2.5, -2, -1.5, -1, -0.5, -0.2, 0, 0.2, 0.5, 1, 1.5, 2, 2.5, 3]
    # class_edges = [-3, -1, -0.5, 0, 0.5, 1, 3]
    class_edges = [-3, -1, 0, 1, 3]
    
    classify_raster_manual_bins(
        in_raster_path=in_raster,
        output_class_raster_path=out_class_raster,
        output_polygons_path=out_polygons,
        class_edges=class_edges,
        polygon_layer="floodplain",
        dissolve_polygons=False,
    )
