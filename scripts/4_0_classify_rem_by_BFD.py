from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
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


def _format_bin_label(low: Optional[float], high: Optional[float]) -> str:
    """
    Human-readable label for the proportion bin.
      low=None means (-inf, high]
      high=None means (low, +inf)
    """
    if low is None and high is not None:
        return f"<= {high:g}x"
    if low is not None and high is not None:
        return f"> {low:g}x to <= {high:g}x"
    if low is not None and high is None:
        return f"> {low:g}x"
    return "unclassified"


def classify_rem_by_bankfull(
    rem_raster_path: str,
    output_class_raster_path: str,
    output_polygons_path: str,
    *,
    # Choose ONE of these:
    bf_raster_path: Optional[str] = None,
    bf_static_value: Optional[float] = None,
    # Classification thresholds (proportions of BF)
    thresholds: Sequence[float] = (0.5, 1.0, 2.0),
    # Raster / polygon output settings
    out_nodata: int = 0,
    polygon_driver: str = "GPKG",
    polygon_layer: str = "rem_bf_classes",
    dissolve_polygons: bool = True,
) -> None:
    """
    Classify a REM raster by how many multiples of bankfull stage the REM is above.

    Ratio = REM / BF

    thresholds=[t1, t2, ... tN] produces N classes (NO overflow class):
      class 1: ratio <= t1
      class 2: t1 < ratio <= t2
      ...
      class N: t{N-1} < ratio <= tN

    IMPORTANT: ratio > tN becomes nodata/unclassified (out_nodata).

    Nodata rules:
      - REM nodata -> output nodata
      - BF nodata -> output nodata (if using bf_raster_path)
      - BF <= 0 -> output nodata

    Outputs:
      1) Classified raster (integer classes, nodata=out_nodata)
      2) Polygon file with field 'Proportion of BF stage' storing a bin label (string)
         and 'class_id' storing the integer class.
    """
    thresholds = [float(t) for t in thresholds]
    if len(thresholds) == 0:
        raise ValueError("thresholds must contain at least one value.")
    if sorted(thresholds) != list(thresholds):
        raise ValueError("thresholds must be sorted ascending (e.g., [0.5, 1, 2]).")
    if any(t < 0 for t in thresholds):
        raise ValueError("thresholds should be non-negative proportions.")

    if (bf_raster_path is None) == (bf_static_value is None):
        raise ValueError("Provide exactly one of bf_raster_path OR bf_static_value (not both).")
    if bf_static_value is not None and bf_static_value <= 0:
        raise ValueError("bf_static_value must be > 0.")

    print(f"[CLASSIFY] REM: {rem_raster_path}")
    if bf_raster_path:
        print(f"[CLASSIFY] BF raster: {bf_raster_path}")
    else:
        print(f"[CLASSIFY] BF static: {bf_static_value:g}")
    print(f"[CLASSIFY] Thresholds: {thresholds}")
    print(f"[OUT] Class raster: {output_class_raster_path}")
    print(f"[OUT] Polygons: {output_polygons_path} (layer={polygon_layer})")

    # -------------------------
    # Read REM
    # -------------------------
    with rasterio.open(rem_raster_path) as src_rem:
        rem = src_rem.read(1, masked=True)  # masked array
        rem_meta = src_rem.meta.copy()
        rem_crs = src_rem.crs
        rem_transform = src_rem.transform
        rem_shape = src_rem.shape

        # -------------------------
        # Get BF array on REM grid
        # -------------------------
        if bf_static_value is not None:
            bf = np.ma.masked_array(
                np.full(rem_shape, float(bf_static_value), dtype="float32"),
                mask=np.zeros(rem_shape, dtype=bool),
            )
        else:
            with rasterio.open(bf_raster_path) as src_bf:
                bf_nodata = src_bf.nodata
                bf_resampled = np.empty(rem_shape, dtype="float32")

                reproject(
                    source=rasterio.band(src_bf, 1),
                    destination=bf_resampled,
                    src_transform=src_bf.transform,
                    src_crs=src_bf.crs,
                    dst_transform=rem_transform,
                    dst_crs=rem_crs,
                    dst_resolution=src_rem.res,
                    resampling=Resampling.bilinear,
                    dst_nodata=bf_nodata,
                )

                if bf_nodata is None:
                    # If BF raster has no explicit nodata, treat NaNs as nodata
                    bf = np.ma.masked_invalid(bf_resampled)
                else:
                    bf = np.ma.masked_equal(bf_resampled, bf_nodata)

    # -------------------------
    # Build valid mask
    # -------------------------
    rem_data = rem.data.astype("float32", copy=False)
    bf_data = bf.data.astype("float32", copy=False)

    valid = (~rem.mask) & (~bf.mask) & np.isfinite(rem_data) & np.isfinite(bf_data) & (bf_data > 0)

    # Ratio = REM / BF
    ratio = np.full(rem_shape, np.nan, dtype="float32")
    ratio[valid] = rem_data[valid] / bf_data[valid]

    # -------------------------
    # Classify (values > last threshold become NODATA)
    # -------------------------
    n_classes = len(thresholds)
    dtype = _choose_int_dtype(n_classes)

    cls = np.full(rem_shape, out_nodata, dtype=np.dtype(dtype))

    # Class 1: ratio <= t1
    t0 = thresholds[0]
    cls[(ratio <= t0) & valid] = 1

    # Middle (and last) classes: (t{i-1}, t{i}]
    for i in range(1, len(thresholds)):
        lo = thresholds[i - 1]
        hi = thresholds[i]
        cls[(ratio > lo) & (ratio <= hi) & valid] = i + 1

    # IMPORTANT: do not classify ratio > thresholds[-1]
    # Those remain out_nodata (unclassified)

    # -------------------------
    # Write class raster
    # -------------------------
    out_meta = rem_meta.copy()
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

    print(f"[✔] Wrote class raster ({n_classes} classes). Unclassified: ratio > {thresholds[-1]:g}x -> nodata")

    # -------------------------
    # Polygonize
    # -------------------------
    # Mask nodata and also ignore background 0
    mask = (cls != out_nodata)

    geoms = []
    vals = []
    for geom, val in shapes(cls, mask=mask, transform=rem_transform):
        v = int(val)
        if v == out_nodata:
            continue
        geoms.append(shape(geom))
        vals.append(v)

    if len(geoms) == 0:
        raise RuntimeError("No polygons were created (all nodata or empty mask).")

    gdf = gpd.GeoDataFrame({"class_id": vals}, geometry=geoms, crs=rem_crs)

    # Add bin labels per class (all classes are bounded above by thresholds)
    class_to_label = {}
    for c in range(1, n_classes + 1):
        if c == 1:
            class_to_label[c] = _format_bin_label(None, thresholds[0])
        else:
            low = thresholds[c - 2]
            high = thresholds[c - 1]
            class_to_label[c] = _format_bin_label(low, high)

    gdf["Proportion of BF stage"] = gdf["class_id"].map(class_to_label)

    if dissolve_polygons:
        # Merge polygons by class_id (and keep the label)
        gdf = (
            gdf.dissolve(by="class_id", as_index=False)
               .merge(
                   gpd.GeoDataFrame(
                       {
                           "class_id": list(class_to_label.keys()),
                           "Proportion of BF stage": list(class_to_label.values()),
                       }
                   ),
                   on="class_id",
                   how="left",
               )
        )

    # Write polygon output
    out_poly_dir = os.path.dirname(output_polygons_path)
    if out_poly_dir:
        os.makedirs(out_poly_dir, exist_ok=True)

    gdf.to_file(output_polygons_path, layer=polygon_layer, driver=polygon_driver)

    print(f"[✔] Wrote polygons: {output_polygons_path} (features={len(gdf)})")


if __name__ == "__main__":
    rem_raster = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\REMs\REM\HAWS_REM_1m.tif"

    # OPTION A: BF raster (reprojected to REM)
    # bf_raster = r"C:\path\to\bf_raster.tif"

    # OPTION B: Static BF stage (same units as REM). Uncomment to use.
    bf_static = 2.22  # ft from StreamStats average BF Depth

    out_class_raster = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\REMs\REM\jc bankfull thresholds 3 classes.tif"
    out_polygons = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\REMs\REM\jc bankfull thresholds 3 classes.gpkg"

    thresholds = [0.5, 1, 2]  # anything > 2x becomes nodata/unclassified

    classify_rem_by_bankfull(
        rem_raster_path=rem_raster,
        output_class_raster_path=out_class_raster,
        output_polygons_path=out_polygons,
        # bf_raster_path=bf_raster,     # <-- use BF raster
        bf_static_value=bf_static,  # <-- or use static BF
        thresholds=thresholds,
        polygon_layer="floodplain",
        dissolve_polygons=False,
    )
