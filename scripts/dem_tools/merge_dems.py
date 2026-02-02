#!/usr/bin/env python3
"""
Merge all GeoTIFFs in one or more folders into single DEM GeoTIFFs.

For each input folder:
- Output is saved in the parent folder with the same name as the input folder:
  e.g.  C:\...\STHD\2006_dtm  ->  C:\...\STHD\2006_dtm.tif
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge


def _print_tile_sanity(tifs: Sequence[Path]) -> None:
    print("\n=== Tile sanity (band 1, masked) ===")
    for p in tifs:
        with rasterio.open(p) as s:
            a = s.read(1, masked=True)
            if np.ma.is_masked(a) and a.count() == 0:
                amin = amax = "ALL_MASKED"
            else:
                amin = float(a.min())
                amax = float(a.max())
            print(
                f"{p.name}\n"
                f"  CRS:    {s.crs}\n"
                f"  RES:    {s.res}\n"
                f"  DTYPE:  {a.dtype}\n"
                f"  NODATA: {s.nodata}\n"
                f"  MIN/MAX:{amin} / {amax}\n"
            )


def merge_folder_to_dem(
    input_folder: Union[str, Path],
    output_folder: Union[str, Path],
    out_name: str = "merged_dem.tif",
    nodata: Optional[float] = -9999.0,
    overwrite: bool = False,
    sanity_print: bool = True,
) -> Path:
    """
    Merge all GeoTIFFs in `input_folder` into a single DEM in `output_folder`.

    Notes:
    - This does NOT reproject. Inputs must already be in the same CRS.
    - If inputs have different transforms/resolutions, rasterio.merge will choose a grid.
    """
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_path} (set overwrite=True to replace)")

    tifs: Sequence[Path] = sorted(p for p in in_dir.iterdir() if p.suffix.lower() == ".tif")
    if not tifs:
        raise FileNotFoundError(f"No .tif files found in: {in_dir}")

    if sanity_print:
        _print_tile_sanity(tifs)

    srcs = [rasterio.open(p) for p in tifs]
    try:
        out_nodata = float(-9999.0 if nodata is None else nodata)

        print("\nMerging DEMs:")
        for p in tifs:
            print(f"  - {p}")

        mosaic, transform = rio_merge(
            srcs,
            nodata=out_nodata,
            masked=True,
            method="first",
            resampling=Resampling.bilinear,
        )

        mosaic_filled = np.ma.filled(mosaic, out_nodata).astype("float32")

        profile = srcs[0].profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": mosaic_filled.shape[1],
                "width": mosaic_filled.shape[2],
                "transform": transform,
                "count": mosaic_filled.shape[0],
                "nodata": out_nodata,
                "dtype": "float32",
                "compress": "deflate",
                "predictor": 2,
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "BIGTIFF": "IF_SAFER",
            }
        )

        print(f"Writing output to {out_path}")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic_filled)

        with rasterio.open(out_path) as ds:
            bounds = ds.bounds
            res: Tuple[float, float] = ds.res
            data = ds.read(1, masked=True)
            if ds.nodata is not None:
                data = np.ma.masked_equal(data, ds.nodata)

            print("\n=== Merged DEM ===")
            print(f"Path:    {out_path}")
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

            if data.count() > 0:
                arr = data.compressed()
                print(
                    f"Stats:   min={arr.min():.3f}, max={arr.max():.3f}, "
                    f"mean={arr.mean():.3f}, std={arr.std(ddof=1):.3f}, "
                    f"valid_px={arr.size}"
                )
            else:
                print("Stats:   all NoData")

        return out_path

    finally:
        for s in srcs:
            s.close()


def merge_folders_to_parent_named_outputs(
    input_folders: Sequence[Union[str, Path]],
    nodata: Optional[float] = -9999.0,
    overwrite: bool = False,
    sanity_print: bool = True,
) -> list[Path]:
    """
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
        )
        outputs.append(out_path)

    return outputs


if __name__ == "__main__":
    input_folders = [
        r"C:\L\OneDrive - Lichen\Documents\Projects\SFToutle\Tree Height Analysis\STHD\2019_dsm",
        r"C:\L\OneDrive - Lichen\Documents\Projects\SFToutle\Tree Height Analysis\STHD\2019_dtm",
    ]

    merge_folders_to_parent_named_outputs(
        input_folders=input_folders,
        nodata=0,          # set to whatever you want the merged nodata to be
        overwrite=True,
        sanity_print=True,
    )
