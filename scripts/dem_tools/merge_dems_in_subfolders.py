from __future__ import annotations

import os
from pathlib import Path

import rasterio
from rasterio.merge import merge


def merge_folder_tifs_to_parent_name(
    input_dir: str | Path,
    *,
    overwrite: bool = False,
    glob_patterns: tuple[str, ...] = ("*.tif", "*.tiff"),
) -> None:
    """
    Recursively scan all subfolders of input_dir.
    For each folder containing TIFFs, merge them and write:
        input_dir / f"{folder.name}.tif"
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {input_dir}")

    # Walk all subdirectories (recursive). Skip the root itself.
    subdirs = [p for p in input_dir.rglob("*") if p.is_dir()]

    for folder in subdirs:
        # Collect TIFFs directly inside this folder (not from nested subfolders)
        tifs: list[Path] = []
        for pat in glob_patterns:
            tifs.extend(sorted(folder.glob(pat)))

        if not tifs:
            continue

        out_tif = input_dir / f"{folder.name}.tif"
        if out_tif.exists() and not overwrite:
            print(f"SKIP (exists): {out_tif}")
            continue

        print(f"Merging {len(tifs)} tif(s) from: {folder}")
        print(f"  -> {out_tif}")

        # Open sources
        srcs = []
        try:
            for tif in tifs:
                srcs.append(rasterio.open(tif))

            # Merge (mosaic). If rasters overlap, later ones can overwrite earlier ones.
            mosaic, out_transform = merge(srcs)

            # Use metadata from first raster
            out_meta = srcs[0].meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_transform,
                    "count": mosaic.shape[0],
                    # Good defaults for size/performance:
                    "compress": "deflate",
                    "predictor": 2 if out_meta.get("dtype") in ("int16", "int32", "float32", "float64") else 1,
                    "tiled": True,
                    "blockxsize": 256,
                    "blockysize": 256,
                    "BIGTIFF": "IF_SAFER",
                }
            )

            # Write output
            with rasterio.open(out_tif, "w", **out_meta) as dst:
                dst.write(mosaic)

        finally:
            for s in srcs:
                try:
                    s.close()
                except Exception:
                    pass


if __name__ == "__main__":
    input_folder = r"C:\L\OneDrive - Lichen\Documents\Projects\SFToutle\Tree Height Analysis\STHD"
    merge_folder_tifs_to_parent_name(input_folder, overwrite=False)
