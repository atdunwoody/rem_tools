from pathlib import Path
from typing import Optional, Literal
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio import shutil as rio_shutil
import glob
import os

def _pxsize(tr):
    return (abs(tr.a), abs(tr.e))

def _safe_delete(path: Path):
    """Delete a possibly-corrupt GeoTIFF and common sidecars without trying to open it."""
    try:
        # Use rasterio's delete when possible (handles overviews, etc.)
        rio_shutil.delete(str(path))
    except Exception:
        # Fall back to raw unlink + sidecars
        for p in [path,
                  Path(str(path) + ".aux.xml"),
                  Path(str(path) + ".ovr"),
                  Path(str(path) + ".msk")]:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
        # Also remove GDAL .xml sidecar variants, just in case
        for side in glob.glob(str(path) + ".*.aux.xml"):
            try:
                os.remove(side)
            except Exception:
                pass

def difference_rasters(
    raster_a_path: str,
    raster_b_path: str,
    out_path: str,
    *,
    order: Literal["A-B", "B-A"] = "A-B",
    align_to: Literal["A", "B"] = "B",
    resampling: Resampling = Resampling.bilinear,
    out_nodata: Optional[float] = np.nan,
    overwrite: bool = False,
) -> Path:
    """
    Difference two rasters after aligning grid (CRS, transform, resolution).
    """
    out_path = Path(out_path)

    # Proactively delete a corrupt/partial prior output to avoid GDAL open errors.
    if out_path.exists():
        if not overwrite:
            raise FileExistsError(f"{out_path} already exists. Set overwrite=True to replace.")
        _safe_delete(out_path)

    with rasterio.open(raster_a_path) as src_a, rasterio.open(raster_b_path) as src_b:
        # Choose reference grid
        ref_src = src_a if align_to.upper() == "A" else src_b
        mov_src = src_b if align_to.upper() == "A" else src_a

        print("[info] Reference:", ("A" if ref_src is src_a else "B"))
        print("[info] Ref CRS:", ref_src.crs)
        print("[info] Mov CRS:", mov_src.crs)
        if ref_src.crs != mov_src.crs:
            print("[warn] CRS differ; the moving raster will be reprojected to reference CRS.")

        ref_px = _pxsize(ref_src.transform)
        mov_px = _pxsize(mov_src.transform)
        if not np.isclose(ref_px[0], mov_px[0]) or not np.isclose(ref_px[1], mov_px[1]):
            print("[warn] Pixel sizes/resolution differ; the moving raster will be resampled.")

        if (ref_src.width != mov_src.width) or (ref_src.height != mov_src.height) \
           or (ref_src.transform != mov_src.transform):
            print("[info] Grids differ (extent/shape/transform); using reference grid for output.")

        # Build a VRT for the moving raster aligned to the reference grid
        vrt_opts = {
            "crs": ref_src.crs,
            "transform": ref_src.transform,
            "height": ref_src.height,
            "width": ref_src.width,
            "resampling": resampling,
        }
        with WarpedVRT(mov_src, **vrt_opts) as mov_vrt:
            ref = ref_src.read(1, out_dtype="float32", masked=True)
            mov = mov_vrt.read(1, out_dtype="float32", masked=True)

            mask = np.ma.getmaskarray(ref) | np.ma.getmaskarray(mov)

            if order == "A-B":
                arr = (ref if align_to.upper() == "A" else mov) - (mov if align_to.upper() == "A" else ref)
            elif order == "B-A":
                arr = (mov if align_to.upper() == "A" else ref) - (ref if align_to.upper() == "A" else mov)
            else:
                raise ValueError("order must be 'A-B' or 'B-A'")

            # Apply mask and fill with NoData
            fill_val = np.nan if out_nodata is None else float(out_nodata)
            arr = np.where(mask, fill_val, arr).astype("float32")

            # Prepare output profile
            profile = ref_src.profile.copy()
            # Choose conservative tiling to avoid weird tile math on small rasters
            blocksize = 256
            profile.update(
                driver="GTiff",
                dtype="float32",
                count=1,
                nodata=fill_val,
                compress="DEFLATE",
                predictor=3,            # good for float + DEFLATE
                tiled=True,
                blockxsize=blocksize,
                blockysize=blocksize,
                BIGTIFF="IF_SAFER",     # auto-promote if needed
            )

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(arr, 1)

            print(f"[done] Wrote {out_path}")
            print(f"[info] Output CRS: {profile['crs']}")
            print(f"[info] Output shape: {profile['height']} x {profile['width']}")
            print(f"[info] Output pixel size: {abs(profile['transform'].a)}, {abs(profile['transform'].e)}")
            print(f"[info] NoData: {profile['nodata']}")

    return out_path


#"C:\L\OneDrive - Lichen\Documents\Projects\Salmon Cr\HAWS\transects_by_site\interpolated_WSE_4.tif"

difference_rasters(
    raster_a_path = r"C:\L\OneDrive - Lichen\Documents\Projects\Salmon Cr\HAWS\DEM\2024_dtm_3ft.tif",
    raster_b_path =r"C:\L\OneDrive - Lichen\Documents\Projects\Salmon Cr\HAWS\interpolated_WSE_2p.tif",
    out_path = r"C:\L\OneDrive - Lichen\Documents\Projects\Salmon Cr\HAWS\REM_upper_2024_2p.tif",
    order="A-B",
    align_to="B",
    resampling=Resampling.bilinear,  # or Resampling.nearest for categorical inputs
    out_nodata=np.nan,
    overwrite=True,
)
