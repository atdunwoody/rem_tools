"""Mosaic Silver Creek bare-earth LiDAR DEM tiles for subsequent REM analysis.

Run: python mosaic_silver_creek_lidar.py
Dependencies: numpy, rasterio, pyproj (pyproj >= 3.6).

Edit USER SETTINGS below. Reads .tif/.tiff files recursively. The first folder
has highest overlap priority; within a folder, the first sorted filename wins.
Lower-priority data fill only cells lacking valid higher-priority data.

Produces a compressed, tiled Float32 BigTIFF and a source-inventory CSV.
The first tile supplies the default horizontal CRS, resolution, and grid origin.
Other tiles are resampled onto this grid. Only small blocks are held in memory.

Inputs must be bare-earth DEMs, not hillshades, intensity images, or canopy DSMs.
Elevation units and vertical datums must already be compatible. Known metadata
conflicts stop the run; missing metadata cannot be verified. Horizontal
reprojection does NOT convert elevation units or vertical datums. Band scale
and offset metadata are applied to obtain physical elevation values.
This script does not calculate the REM, fill terrain gaps, or adjust survey seams.

API references:
https://rasterio.readthedocs.io/en/stable/topics/virtual-warping.html
https://rasterio.readthedocs.io/en/stable/topics/masks.html
https://pyproj4.github.io/pyproj/stable/api/crs/crs.html
"""

from __future__ import annotations

import csv
import math
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS as ProjCRS
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, transform_bounds
from rasterio.windows import Window, bounds as window_bounds


# =============================== USER SETTINGS ===============================
LIDAR_FOLDER = Path(
    r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\CFC Silver Creek"
    r"\Field Data\LiDAR"
)

# Highest priority FIRST. Change this order if a different survey is preferred.
INPUT_FOLDERS = [
    LIDAR_FOLDER / "USGS 2022-2023",
    LIDAR_FOLDER / "2022",
    LIDAR_FOLDER / "2020",
]
OUTPUT_TIF = LIDAR_FOLDER / "Silver_Creek_DEM_mosaic.tif"

# None uses the first tile's horizontal CRS. Otherwise specify a projected CRS,
# e.g. "EPSG:26910", after confirming it is appropriate for your project.
OUTPUT_CRS = "EPSG:6599" # 

# None uses the first tile's resolution, or its suggested resolution after
# reprojection if OUTPUT_CRS differs. A number sets square cells; (x, y) sets
# rectangular cells. Units are OUTPUT_CRS horizontal units, NOT elevation units.
OUTPUT_RESOLUTION = 3

RESAMPLING = "bilinear"  # "nearest" preserves sampled source values.
OUTPUT_NODATA = -9999.0

# Optional substring filter for DEM filenames when folders contain other TIFFs.
# Example: "dem". None includes ALL .tif/.tiff files in the listed folders.
FILENAME_CONTAINS = None

# Override an incorrect/missing NoData tag by folder name. Values are RAW stored
# pixel values before band scale/offset. Otherwise embedded tags/masks are used.
# Example: {"2020": -32767.0}. Do not mark zero as NoData unless documented.
NODATA_OVERRIDES = {}

BLOCK_SIZE = 1024       # Processing block width/height in pixels.
MAX_OPEN_FILES = 24     # Limit simultaneously open source rasters.
GDAL_CACHE_MB = 256
OVERWRITE = False      # True replaces existing outputs only after success.
# =============================================================================


def find_tiles(folders, output_path):
    """Find unique TIFFs in explicit priority order; never include our output."""
    tiles, seen = [], {Path(output_path).resolve()}
    for priority, folder in enumerate(map(Path, folders), start=1):
        if not folder.is_dir():
            raise FileNotFoundError(f"Input folder does not exist: {folder}")
        files = sorted(
            (p for p in folder.rglob("*")
             if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
             and (not FILENAME_CONTAINS
                  or FILENAME_CONTAINS.lower() in p.name.lower())),
            key=lambda p: (str(p).casefold(), str(p)),
        )
        if not files:
            raise FileNotFoundError(f"No matching TIFF files found in: {folder}")
        added = 0
        for path in files:
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                tiles.append({"path": resolved, "folder": folder.name,
                              "priority": priority})
                added += 1
        print(f"Priority {priority}: {folder.name}: {added:,} TIFFs", flush=True)
    if not tiles:
        raise ValueError("No input TIFFs remain after excluding duplicates/output.")
    return tiles


def normalize_unit(unit):
    """Normalize common elevation unit labels without inferring from XY units."""
    value = (unit or "").strip().lower().replace("_", " ")
    aliases = {
        "meter": "m", "meters": "m", "metre": "m", "metres": "m",
        "foot": "ft", "feet": "ft", "international foot": "ft",
        "us survey foot": "us-ft", "us survey feet": "us-ft",
        "foot us": "us-ft", "ftus": "us-ft", "ft us": "us-ft",
    }
    return aliases.get(value, value)


def inspect_tiles(tiles):
    """Read metadata only, and reject identifiable vertical-reference conflicts."""
    for index, item in enumerate(tiles, start=1):
        with rasterio.open(item["path"]) as src:
            if src.count != 1 or np.dtype(src.dtypes[0]).kind not in "iuf":
                raise ValueError(f"Expected a single-band numeric DEM: {item['path']}")
            if src.crs is None:
                raise ValueError(f"Missing CRS; assign the correct CRS first: {item['path']}")
            full_crs = ProjCRS.from_wkt(src.crs.to_wkt())
            horizontal_crs = CRS.from_wkt(full_crs.to_2d().to_wkt())
            vertical_parts = [c for c in full_crs.sub_crs_list if c.is_vertical]
            vertical = vertical_parts[0] if vertical_parts else None
            band_unit = normalize_unit(src.units[0])
            vertical_unit = normalize_unit(vertical.axis_info[0].unit_name) if vertical else ""
            if band_unit and vertical_unit and band_unit != vertical_unit:
                raise ValueError(f"Band and vertical CRS units disagree: {item['path']}")
            scale, offset = src.scales[0], src.offsets[0]
            if not np.isfinite([scale, offset]).all() or scale == 0:
                raise ValueError(f"Invalid band scale/offset: {item['path']}")
            nodata = NODATA_OVERRIDES.get(item["folder"], src.nodata)
            item.update(
                crs=horizontal_crs, full_crs=src.crs.to_wkt(),
                transform=src.transform, bounds=tuple(src.bounds),
                width=src.width, height=src.height, resolution=src.res,
                nodata=nodata, scale=scale, offset=offset,
                z_unit=band_unit or vertical_unit,
                vertical_crs=vertical.to_wkt() if vertical else "",
                vertical_name=vertical.name if vertical else "",
            )
        if index == len(tiles) or index % 100 == 0:
            print(f"Inspected {index:,}/{len(tiles):,} tiles", flush=True)

    units = {s["z_unit"] for s in tiles if s["z_unit"]}
    if len(units) > 1:
        raise ValueError(f"Mixed elevation units: {sorted(units)}. Convert inputs first.")
    verticals = [ProjCRS(s["vertical_crs"]) for s in tiles if s["vertical_crs"]]
    if verticals and any(not v.equals(verticals[0]) for v in verticals[1:]):
        names = sorted({s["vertical_name"] for s in tiles if s["vertical_name"]})
        raise ValueError(f"Different vertical CRSs: {names}. Reconcile them first.")
    missing_units = sum(not s["z_unit"] for s in tiles)
    missing_vertical = sum(not s["vertical_crs"] for s in tiles)
    print(f"Elevation units in available metadata: {', '.join(sorted(units)) or 'unknown'}")
    if missing_units or missing_vertical:
        print(
            f"NOTE: {missing_units} tiles lack elevation-unit metadata; "
            f"{missing_vertical} lack a vertical CRS. Confirm compatibility from "
            "survey documentation. XY reprojection does not reconcile elevations.",
            flush=True,
        )


def choose_grid(tiles):
    """Cover the union of footprints on a grid anchored to the reference tile."""
    first = tiles[0]
    target = ProjCRS.from_user_input(OUTPUT_CRS or first["crs"])
    if target.is_compound or len(target.axis_info) != 2 or not target.is_projected:
        raise ValueError("OUTPUT_CRS must be a 2D projected CRS appropriate for the site.")
    target_crs = CRS.from_wkt(target.to_wkt())
    native = first["transform"]
    if (first["crs"] == target_crs and native.b == native.d == 0
            and native.a > 0 and native.e < 0):
        reference = native
    else:
        reference, _, _ = calculate_default_transform(
            first["crs"], target_crs, first["width"], first["height"],
            *first["bounds"],
        )
    resolution = OUTPUT_RESOLUTION
    if resolution is None:
        xres, yres = abs(reference.a), abs(reference.e)
    elif np.isscalar(resolution):
        xres = yres = float(resolution)
    else:
        xres, yres = map(float, resolution)
    if not np.isfinite([xres, yres]).all() or min(xres, yres) <= 0:
        raise ValueError("OUTPUT_RESOLUTION must contain positive finite values.")

    for item in tiles:
        item["target_bounds"] = transform_bounds(
            item["crs"], target_crs, *item["bounds"], densify_pts=41,
        )
        if not np.isfinite(item["target_bounds"]).all():
            raise ValueError(f"Cannot transform footprint: {item['path']}")
    footprints = np.array([s["target_bounds"] for s in tiles])
    left, bottom = footprints[:, :2].min(axis=0)
    right, top = footprints[:, 2:].max(axis=0)
    # Retain the first tile's pixel origin. Avoid half-cell shifts in that survey.
    col0 = math.floor((left - reference.c) / xres + 1e-8)
    col1 = math.ceil((right - reference.c) / xres - 1e-8)
    row0 = math.floor((reference.f - top) / yres + 1e-8)
    row1 = math.ceil((reference.f - bottom) / yres - 1e-8)
    width, height = col1 - col0, row1 - row0
    if min(width, height) <= 0:
        raise ValueError("Output grid has no area.")
    transform = Affine(xres, 0, reference.c + col0 * xres,
                       0, -yres, reference.f - row0 * yres)
    print(f"Output CRS: {target.to_string()}")
    print(f"Cell size: {xres:g} x {yres:g} {target.axis_info[0].unit_name}")
    print(f"Grid: {width:,} columns x {height:,} rows; "
          f"{width * height * 4 / 1024**3:.2f} GiB before compression", flush=True)
    return target_crs, transform, width, height


@contextmanager
def vrt_cache(crs, transform, width, height, resampling):
    """Keep a bounded number of open, lazily reprojected source datasets."""
    cache = OrderedDict()

    def get(item):
        key = item["path"]
        if key in cache:
            cache.move_to_end(key)
            return cache[key][1]
        if len(cache) >= MAX_OPEN_FILES:
            _, (old_src, old_vrt) = cache.popitem(last=False)
            old_vrt.close()
            old_src.close()
        src = rasterio.open(key)
        try:
            # Explicit horizontal-only source/target CRSs avoid vertical shifts.
            # NaN internally avoids collision with real elevations such as zero.
            vrt = WarpedVRT(
                src, src_crs=item["crs"], crs=crs, transform=transform,
                width=width, height=height, dtype="float32",
                src_nodata=item["nodata"], nodata=float("nan"),
                resampling=resampling, warp_mem_limit=64,
            )
        except Exception:
            src.close()
            raise
        cache[key] = (src, vrt)
        return vrt

    try:
        yield get
    finally:
        for src, vrt in cache.values():
            vrt.close()
            src.close()


def write_mosaic(tiles, path, grid):
    """Write each output block once using the first valid elevation at each cell."""
    crs, transform, width, height = grid
    resampling = Resampling[RESAMPLING]
    output_nodata = np.float32(OUTPUT_NODATA)
    profile = dict(
        driver="GTiff", width=width, height=height, count=1, dtype="float32",
        crs=crs, transform=transform, nodata=float(output_nodata),
        tiled=True, blockxsize=512, blockysize=512,
        compress="DEFLATE", predictor=3, BIGTIFF="YES",
    )
    footprints = np.array([s["target_bounds"] for s in tiles])
    total = math.ceil(width / BLOCK_SIZE) * math.ceil(height / BLOCK_SIZE)
    done, next_report, valid_count = 0, 5, 0
    zmin, zmax = math.inf, -math.inf
    with rasterio.Env(GDAL_CACHEMAX=GDAL_CACHE_MB * 1024**2):
        with rasterio.open(path, "w", **profile) as dst:
            with vrt_cache(*grid, resampling) as get_vrt:
                for row in range(0, height, BLOCK_SIZE):
                    for col in range(0, width, BLOCK_SIZE):
                        h, w = min(BLOCK_SIZE, height - row), min(BLOCK_SIZE, width - col)
                        window = Window(col, row, w, h)
                        left, bottom, right, top = window_bounds(window, transform)
                        intersects = (
                            (footprints[:, 0] < right) & (footprints[:, 2] > left)
                            & (footprints[:, 1] < top) & (footprints[:, 3] > bottom)
                        )
                        block = np.full((h, w), output_nodata, dtype="float32")
                        empty = np.ones((h, w), dtype=bool)
                        for index in np.flatnonzero(intersects):
                            item = tiles[index]
                            data = get_vrt(item).read(1, window=window, masked=True)
                            values = data.data * item["scale"] + item["offset"]
                            use = empty & ~np.ma.getmaskarray(data) & np.isfinite(values)
                            if np.any(values[use] == output_nodata):
                                raise ValueError("A valid elevation equals OUTPUT_NODATA; "
                                                 "choose a different output sentinel.")
                            block[use] = values[use]
                            empty[use] = False
                            if not empty.any():
                                break
                        valid = block[~empty]
                        valid_count += valid.size
                        if valid.size:
                            zmin, zmax = min(zmin, float(valid.min())), max(zmax, float(valid.max()))
                        dst.write(block, 1, window=window)
                        done += 1
                        while next_report <= 100 and 100 * done / total >= next_report:
                            print(f"Mosaic: {next_report}%", flush=True)
                            next_report += 5
            if not valid_count:
                raise ValueError("No valid elevations were found in the mosaic.")
            dst.set_band_description(1, "Bare-earth elevation")
            units = {s["z_unit"] for s in tiles}
            if len(units) == 1 and "" not in units:
                dst.set_band_unit(1, next(iter(units)))
            dst.update_tags(
                SOURCE_TILE_COUNT=str(len(tiles)), OVERLAP_RULE="first valid source",
                RESAMPLING=RESAMPLING, VERTICAL_TRANSFORMATION="none",
                VERTICAL_REFERENCE_STATUS="See source inventory and survey documentation",
            )
    print(f"Valid coverage: {100 * valid_count / (width * height):.2f}% of output rectangle")
    print(f"Elevation range: {zmin:g} to {zmax:g} (input elevation units)", flush=True)


def write_inventory(tiles, path):
    """Record source order and metadata for reviewing survey compatibility."""
    fields = ["order", "priority", "folder", "path", "full_crs", "resolution",
              "nodata", "scale", "offset", "z_unit", "vertical_name", "vertical_crs"]
    with open(path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for order, item in enumerate(tiles, start=1):
            writer.writerow(dict(item, order=order))


def main():
    output = Path(OUTPUT_TIF)
    inventory = output.with_name(output.stem + "_sources.csv")
    if BLOCK_SIZE < 1 or MAX_OPEN_FILES < 1 or GDAL_CACHE_MB < 1:
        raise ValueError("Block size, maximum open files, and cache must be positive.")
    if RESAMPLING not in {"nearest", "bilinear", "cubic", "average"}:
        raise ValueError("RESAMPLING must be nearest, bilinear, cubic, or average.")
    if not np.isfinite(np.float32(OUTPUT_NODATA)):
        raise ValueError("OUTPUT_NODATA must be a finite Float32 value.")
    for path in (output, inventory):
        if path.exists() and not OVERWRITE:
            raise FileExistsError(f"Output exists: {path}. Set OVERWRITE=True to replace it.")
    tiles = find_tiles(INPUT_FOLDERS, output)
    inspect_tiles(tiles)
    grid = choose_grid(tiles)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the previous mosaic intact until the new one is complete. An exception
    # removes temporary files. OVERWRITE=True requires room for both mosaics.
    with tempfile.TemporaryDirectory(prefix="mosaic_work_", dir=output.parent) as tmp:
        temporary_tif = Path(tmp) / output.name
        temporary_csv = Path(tmp) / inventory.name
        write_mosaic(tiles, temporary_tif, grid)
        write_inventory(tiles, temporary_csv)
        temporary_tif.replace(output)
        temporary_csv.replace(inventory)
    print(f"Saved DEM: {output}")
    print(f"Saved source inventory: {inventory}")
    print("Review survey boundaries for elevation steps before calculating the REM.")


if __name__ == "__main__":
    main()
