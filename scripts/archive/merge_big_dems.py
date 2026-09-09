#!/usr/bin/env python3
r"""
Merge all GeoTIFFs in one or more folders into single DEM GeoTIFFs, while
avoiding the large in-memory allocations caused by rasterio.merge().

For each input folder:
- Output is saved in the parent folder with the same name as the input folder:
  e.g.  C:\...\STHD\2006_dtm  ->  C:\...\STHD\2006_dtm.tif

Approach
--------
This script does not build the full mosaic in memory. Instead it:

1. Scans input rasters and validates compatibility
2. Computes the output mosaic extent/grid
3. Creates the output raster
4. Copies each source raster into the output one block at a time
5. Computes output stats blockwise

Notes
-----
- Inputs must already be in the same CRS.
- Inputs should have the same pixel size and band count.
- The script assumes a north-up grid with no rotation.
- Overlap handling uses "first valid wins" in folder sort order:
    earlier rasters in the sorted list keep precedence.
- This is intended for DEM/DSM-style rasters with a single numeric band.

Why this avoids memory errors
-----------------------------
- No call to rasterio.merge.merge()
- No full-raster masked-array reads for stats
- No creation of extra full-size boolean masks from np.ma.masked_equal()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import math
import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import from_bounds


# -----------------------------------------------------------------------------
# User inputs
# -----------------------------------------------------------------------------

INPUT_FOLDERS = [
    r"C:\L\OneDrive - Lichen\Documents\Projects\Tucannon\Wenaha\DEMs\Tuccannon DEM 1m\WA LiDAR 2018\2018 DEM",
    r"C:\L\OneDrive - Lichen\Documents\Projects\Tucannon\Wenaha\DEMs\Tuccannon DEM 1m\WA LiDAR 2018\2018 DSM",
]

OUT_NODATA = 0.0
OVERWRITE = True
SANITY_PRINT = True

# Internal processing block size for output windows.
# Larger blocks reduce overhead but use more memory.
OUT_BLOCKSIZE = 512


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RasterInfo:
    path: Path
    crs: object
    transform: Affine
    width: int
    height: int
    count: int
    dtype: str
    nodata: Optional[float]
    xres: float
    yres: float
    left: float
    bottom: float
    right: float
    top: float


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def _iter_tifs(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".tif")


def _read_info(path: Path) -> RasterInfo:
    with rasterio.open(path) as src:
        if src.transform.is_rectilinear is False:
            raise ValueError(f"Raster has non-rectilinear transform: {path}")

        left, bottom, right, top = src.bounds
        xres, yres = src.res

        return RasterInfo(
            path=path,
            crs=src.crs,
            transform=src.transform,
            width=src.width,
            height=src.height,
            count=src.count,
            dtype=src.dtypes[0],
            nodata=src.nodata,
            xres=float(abs(xres)),
            yres=float(abs(yres)),
            left=float(left),
            bottom=float(bottom),
            right=float(right),
            top=float(top),
        )


def _validate_inputs(infos: Sequence[RasterInfo], tol: float = 1e-9) -> None:
    if not infos:
        raise ValueError("No rasters provided for validation.")

    ref = infos[0]

    for info in infos[1:]:
        if info.crs != ref.crs:
            raise ValueError(
                f"CRS mismatch:\n"
                f"  {ref.path.name}: {ref.crs}\n"
                f"  {info.path.name}: {info.crs}"
            )

        if info.count != ref.count:
            raise ValueError(
                f"Band count mismatch:\n"
                f"  {ref.path.name}: {ref.count}\n"
                f"  {info.path.name}: {info.count}"
            )

        if not math.isclose(info.xres, ref.xres, rel_tol=0.0, abs_tol=tol):
            raise ValueError(
                f"X resolution mismatch:\n"
                f"  {ref.path.name}: {ref.xres}\n"
                f"  {info.path.name}: {info.xres}"
            )

        if not math.isclose(info.yres, ref.yres, rel_tol=0.0, abs_tol=tol):
            raise ValueError(
                f"Y resolution mismatch:\n"
                f"  {ref.path.name}: {ref.yres}\n"
                f"  {info.path.name}: {info.yres}"
            )

        # Reject rotated/sheared grids
        if not (
            math.isclose(info.transform.b, 0.0, abs_tol=tol)
            and math.isclose(info.transform.d, 0.0, abs_tol=tol)
        ):
            raise ValueError(f"Rotated/sheared transform not supported: {info.path}")

    if not (
        math.isclose(ref.transform.b, 0.0, abs_tol=tol)
        and math.isclose(ref.transform.d, 0.0, abs_tol=tol)
    ):
        raise ValueError(f"Rotated/sheared transform not supported: {ref.path}")


def _print_blockwise_minmax(path: Path) -> None:
    with rasterio.open(path) as src:
        vmin = np.inf
        vmax = -np.inf
        found = False

        for _, window in src.block_windows(1):
            arr = src.read(1, window=window, masked=True)
            if arr.count() == 0:
                continue

            vals = arr.compressed()
            if vals.size == 0:
                continue

            vmin = min(vmin, float(vals.min()))
            vmax = max(vmax, float(vals.max()))
            found = True

        if found:
            print(
                f"{path.name}\n"
                f"  CRS:    {src.crs}\n"
                f"  RES:    {src.res}\n"
                f"  DTYPE:  {src.dtypes[0]}\n"
                f"  NODATA: {src.nodata}\n"
                f"  MIN/MAX:{vmin} / {vmax}\n"
            )
        else:
            print(
                f"{path.name}\n"
                f"  CRS:    {src.crs}\n"
                f"  RES:    {src.res}\n"
                f"  DTYPE:  {src.dtypes[0]}\n"
                f"  NODATA: {src.nodata}\n"
                f"  MIN/MAX:ALL_MASKED / ALL_MASKED\n"
            )


def _print_tile_sanity(tifs: Sequence[Path]) -> None:
    print("\n=== Tile sanity (band 1, blockwise) ===")
    for p in tifs:
        _print_blockwise_minmax(p)


def _compute_union_grid(infos: Sequence[RasterInfo]) -> tuple[Affine, int, int, tuple[float, float, float, float]]:
    ref = infos[0]
    xres = ref.xres
    yres = ref.yres

    left = min(i.left for i in infos)
    bottom = min(i.bottom for i in infos)
    right = max(i.right for i in infos)
    top = max(i.top for i in infos)

    # Snap union bounds to reference grid
    origin_x = ref.transform.c
    origin_y = ref.transform.f

    col_min = math.floor((left - origin_x) / xres)
    col_max = math.ceil((right - origin_x) / xres)
    row_min = math.floor((origin_y - top) / yres)
    row_max = math.ceil((origin_y - bottom) / yres)

    out_left = origin_x + col_min * xres
    out_top = origin_y - row_min * yres
    out_right = origin_x + col_max * xres
    out_bottom = origin_y - row_max * yres

    out_width = int(col_max - col_min)
    out_height = int(row_max - row_min)

    out_transform = Affine(xres, 0.0, out_left, 0.0, -yres, out_top)
    out_bounds = (out_left, out_bottom, out_right, out_top)

    return out_transform, out_width, out_height, out_bounds


def _window_grid(width: int, height: int, blocksize: int) -> Iterable[Window]:
    for row_off in range(0, height, blocksize):
        h = min(blocksize, height - row_off)
        for col_off in range(0, width, blocksize):
            w = min(blocksize, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=w, height=h)


def _intersects(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    a_left, a_bottom, a_right, a_top = a
    b_left, b_bottom, b_right, b_top = b
    return not (a_right <= b_left or a_left >= b_right or a_top <= b_bottom or a_bottom >= b_top)


def _safe_round_window(win: Window) -> Window:
    return Window(
        col_off=int(round(win.col_off)),
        row_off=int(round(win.row_off)),
        width=int(round(win.width)),
        height=int(round(win.height)),
    )


def _copy_sources_to_output(
    src_paths: Sequence[Path],
    out_path: Path,
    out_nodata: float,
    out_bounds: tuple[float, float, float, float],
    out_blocksize: int = 512,
) -> None:
    """
    Copy source rasters into output blockwise.

    Overlap rule:
    - Earlier rasters in src_paths have priority.
    - A destination pixel is only filled where it still equals out_nodata.
    """
    with rasterio.open(out_path, "r+") as dst:
        total_blocks = math.ceil(dst.height / out_blocksize) * math.ceil(dst.width / out_blocksize)
        processed = 0

        for src_idx, src_path in enumerate(src_paths, start=1):
            with rasterio.open(src_path) as src:
                src_bounds = src.bounds
                if not _intersects(
                    out_bounds,
                    (src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top),
                ):
                    print(f"Skipping non-overlapping source: {src_path.name}")
                    continue

                print(f"\nCopying source {src_idx}/{len(src_paths)}: {src_path.name}")

                for win in _window_grid(dst.width, dst.height, out_blocksize):
                    processed += 1

                    dst_win_bounds = window_bounds(win, dst.transform)
                    if not _intersects(
                        dst_win_bounds,
                        (src_bounds.left, src_bounds.bottom, src_bounds.right, src_bounds.top),
                    ):
                        continue

                    src_win = from_bounds(*dst_win_bounds, transform=src.transform)
                    src_win = _safe_round_window(src_win)

                    if src_win.width <= 0 or src_win.height <= 0:
                        continue

                    full_src = Window(0, 0, src.width, src.height)
                    src_win = src_win.intersection(full_src)

                    if src_win.width <= 0 or src_win.height <= 0:
                        continue

                    # Read source resampled onto destination window shape
                    src_data = src.read(
                        indexes=1,
                        window=src_win,
                        out_shape=(int(win.height), int(win.width)),
                        masked=False,
                    )

                    dst_data = dst.read(1, window=win, masked=False)

                    if src.nodata is None:
                        src_valid = np.ones(src_data.shape, dtype=bool)
                    else:
                        src_valid = src_data != src.nodata

                    dst_empty = dst_data == out_nodata
                    write_mask = src_valid & dst_empty

                    if np.any(write_mask):
                        dst_data[write_mask] = src_data[write_mask]
                        dst.write(dst_data, 1, window=win)

                print(f"Finished source: {src_path.name}")


def _print_raster_stats_blockwise(raster_path: Path) -> None:
    with rasterio.open(raster_path) as ds:
        bounds = ds.bounds
        res = ds.res

        valid_px = 0
        vmin = np.inf
        vmax = -np.inf
        total = 0.0
        total_sq = 0.0

        for _, window in ds.block_windows(1):
            arr = ds.read(1, window=window, masked=True)

            if arr.count() == 0:
                continue

            vals = arr.compressed().astype("float64")
            if vals.size == 0:
                continue

            valid_px += vals.size
            total += vals.sum()
            total_sq += np.square(vals).sum()
            vmin = min(vmin, float(vals.min()))
            vmax = max(vmax, float(vals.max()))

        print("\n=== Merged DEM ===")
        print(f"Path:    {raster_path}")
        print(f"Driver:  {ds.driver}")
        print(f"CRS:     {ds.crs}")
        print(f"Size:    {ds.width} x {ds.height} px")
        print(f"Res:     {res[0]} x {res[1]} (map units/pixel)")
        print(f"Bands:   {ds.count}")
        print(f"NoData:  {ds.nodata}")
        print(
            f"Bounds:  left={bounds.left:.3f}, bottom={bounds.bottom:.3f}, "
            f"right={bounds.right:.3f}, top={bounds.top:.3f}"
        )

        if valid_px > 0:
            mean = total / valid_px
            if valid_px > 1:
                var = (total_sq - (total ** 2) / valid_px) / (valid_px - 1)
                std = math.sqrt(max(var, 0.0))
            else:
                std = 0.0

            print(
                f"Stats:   min={vmin:.3f}, max={vmax:.3f}, "
                f"mean={mean:.3f}, std={std:.3f}, valid_px={valid_px}"
            )
        else:
            print("Stats:   all NoData")


def _prepare_output_profile(
    ref_path: Path,
    out_transform: Affine,
    out_width: int,
    out_height: int,
    out_nodata: float,
    out_blocksize: int,
) -> dict:
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()

    profile.update(
        {
            "driver": "GTiff",
            "height": out_height,
            "width": out_width,
            "transform": out_transform,
            "count": 1,
            "dtype": "float32",
            "nodata": float(out_nodata),
            "compress": "deflate",
            "predictor": 2,
            "tiled": True,
            "blockxsize": out_blocksize,
            "blockysize": out_blocksize,
            "BIGTIFF": "IF_SAFER",
        }
    )
    return profile


# -----------------------------------------------------------------------------
# Main merge functions
# -----------------------------------------------------------------------------

def merge_folder_to_dem(
    input_folder: Union[str, Path],
    output_folder: Union[str, Path],
    out_name: str = "merged_dem.tif",
    nodata: Optional[float] = -9999.0,
    overwrite: bool = False,
    sanity_print: bool = True,
    out_blocksize: int = 512,
) -> Path:
    """
    Merge all GeoTIFFs in `input_folder` into a single DEM in `output_folder`,
    using blockwise writing to avoid large memory allocations.
    """
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    if out_path.exists():
        if overwrite:
            out_path.unlink()
        else:
            raise FileExistsError(f"Output exists: {out_path} (set overwrite=True to replace)")

    tifs = _iter_tifs(in_dir)
    if not tifs:
        raise FileNotFoundError(f"No .tif files found in: {in_dir}")

    if sanity_print:
        _print_tile_sanity(tifs)

    infos = [_read_info(p) for p in tifs]
    _validate_inputs(infos)

    out_nodata = float(-9999.0 if nodata is None else nodata)
    out_transform, out_width, out_height, out_bounds = _compute_union_grid(infos)

    print("\nMerging DEMs:")
    for p in tifs:
        print(f"  - {p}")

    print("\nOutput grid:")
    print(f"  Width x Height: {out_width} x {out_height}")
    print(f"  Transform:      {out_transform}")
    print(
        f"  Bounds:         left={out_bounds[0]:.3f}, bottom={out_bounds[1]:.3f}, "
        f"right={out_bounds[2]:.3f}, top={out_bounds[3]:.3f}"
    )

    profile = _prepare_output_profile(
        ref_path=tifs[0],
        out_transform=out_transform,
        out_width=out_width,
        out_height=out_height,
        out_nodata=out_nodata,
        out_blocksize=out_blocksize,
    )

    print(f"\nCreating output raster: {out_path}")
    with rasterio.open(out_path, "w", **profile) as dst:
        fill_block = np.full((1, min(out_blocksize, out_height), min(out_blocksize, out_width)), out_nodata, dtype=np.float32)
        # Initialize by blocks to avoid one giant array allocation
        for win in _window_grid(dst.width, dst.height, out_blocksize):
            arr = np.full((1, int(win.height), int(win.width)), out_nodata, dtype=np.float32)
            dst.write(arr, window=win)

    _copy_sources_to_output(
        src_paths=tifs,
        out_path=out_path,
        out_nodata=out_nodata,
        out_bounds=out_bounds,
        out_blocksize=out_blocksize,
    )

    _print_raster_stats_blockwise(out_path)

    return out_path


def merge_folders_to_parent_named_outputs(
    input_folders: Sequence[Union[str, Path]],
    nodata: Optional[float] = -9999.0,
    overwrite: bool = False,
    sanity_print: bool = True,
    out_blocksize: int = 512,
) -> list[Path]:
    r"""
    For each folder in `input_folders`, merge its .tifs and write:
      <parent_of_folder>\<folder_name>.tif
    """
    outputs: list[Path] = []

    for folder in map(Path, input_folders):
        folder = folder.resolve()
        if not folder.exists() or not folder.is_dir():
            raise NotADirectoryError(f"Not a folder: {folder}")

        out_dir = folder.parent
        out_name = f"{folder.name}.tif"

        print("\n" + "=" * 80)
        print(f"INPUT : {folder}")
        print(f"OUTPUT: {out_dir / out_name}")

        out_path = merge_folder_to_dem(
            input_folder=folder,
            output_folder=out_dir,
            out_name=out_name,
            nodata=nodata,
            overwrite=overwrite,
            sanity_print=sanity_print,
            out_blocksize=out_blocksize,
        )
        outputs.append(out_path)

    return outputs


# -----------------------------------------------------------------------------
# Script entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    merge_folders_to_parent_named_outputs(
        input_folders=INPUT_FOLDERS,
        nodata=OUT_NODATA,
        overwrite=OVERWRITE,
        sanity_print=SANITY_PRINT,
        out_blocksize=OUT_BLOCKSIZE,
    )