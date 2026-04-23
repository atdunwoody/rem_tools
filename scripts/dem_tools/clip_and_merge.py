#!/usr/bin/env python3
"""
Clip all GeoTIFFs in one or more folders by a clipping polygon, then merge the
clipped tiles into a single GeoTIFF per folder.

For each input folder:
- Output is saved in the parent folder with the same name as the input folder:
  e.g.  C:\...\STHD\2006_dtm  ->  C:\...\STHD\2006_dtm.tif

Notes:
- The clipping polygon is reprojected to each raster's CRS before clipping.
- Tiles that do not intersect the clipping polygon are skipped.
- This does NOT reproject rasters prior to merge. Input rasters within a folder
  should already share the same CRS or otherwise be compatible with rasterio.merge.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge as rio_merge
from shapely.geometry import box


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


def _load_clip_gdf(clip_vector: Union[str, Path]) -> gpd.GeoDataFrame:
    clip_gdf = gpd.read_file(clip_vector)
    if clip_gdf.empty:
        raise ValueError(f"Clipping vector has no features: {clip_vector}")
    if clip_gdf.crs is None:
        raise ValueError(f"Clipping vector has no CRS: {clip_vector}")

    clip_gdf = clip_gdf[~clip_gdf.geometry.is_empty & clip_gdf.geometry.notnull()].copy()
    if clip_gdf.empty:
        raise ValueError(f"Clipping vector has no valid geometries: {clip_vector}")

    return clip_gdf


def _clip_raster_to_gdf(
    raster_path: Path,
    clip_gdf: gpd.GeoDataFrame,
    out_nodata: float,
) -> Optional[tuple[np.ndarray, dict]]:
    """
    Clip one raster to the clipping geometry. Returns:
        (clipped_array, clipped_profile)
    or None if there is no intersection.
    """
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS: {raster_path}")

        clip_in_raster_crs = clip_gdf.to_crs(src.crs)

        raster_bounds_geom = box(*src.bounds)
        clip_union = clip_in_raster_crs.union_all()

        if not clip_union.intersects(raster_bounds_geom):
            print(f"  - Skipping {raster_path.name}: no intersection with clip polygon")
            return None

        shapes = [geom.__geo_interface__ for geom in clip_in_raster_crs.geometry if geom and not geom.is_empty]
        if not shapes:
            print(f"  - Skipping {raster_path.name}: no valid clip geometries after reprojection")
            return None

        clipped, clipped_transform = mask(
            src,
            shapes=shapes,
            crop=True,
            nodata=out_nodata,
            filled=False,
            all_touched=False,
        )

        if np.ma.is_masked(clipped) and clipped.count() == 0:
            print(f"  - Skipping {raster_path.name}: clipped result contains no valid pixels")
            return None

        clipped_filled = np.ma.filled(clipped, out_nodata)

        profile = src.profile.copy()
        profile.update(
            {
                "height": clipped_filled.shape[1],
                "width": clipped_filled.shape[2],
                "transform": clipped_transform,
                "nodata": out_nodata,
            }
        )

        return clipped_filled, profile


def merge_folder_to_dem(
    input_folder: Union[str, Path],
    output_folder: Union[str, Path],
    clip_vector: Union[str, Path],
    out_name: str = "merged_dem.tif",
    nodata: Optional[float] = -9999.0,
    overwrite: bool = False,
    sanity_print: bool = True,
) -> Path:
    """
    Clip all GeoTIFFs in `input_folder` by `clip_vector`, then merge into a
    single DEM in `output_folder`.

    Notes:
    - The clip vector is reprojected to each raster CRS before clipping.
    - This does NOT reproject rasters. Inputs should already be in the same CRS.
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

    clip_gdf = _load_clip_gdf(clip_vector)
    out_nodata = float(-9999.0 if nodata is None else nodata)

    memfiles: list[MemoryFile] = []
    srcs = []

    try:
        print("\nClipping and preparing DEM tiles:")
        for tif in tifs:
            print(f"  - {tif}")
            clipped_result = _clip_raster_to_gdf(
                raster_path=tif,
                clip_gdf=clip_gdf,
                out_nodata=out_nodata,
            )

            if clipped_result is None:
                continue

            clipped_arr, clipped_profile = clipped_result
            clipped_profile.update(
                {
                    "driver": "GTiff",
                    "dtype": str(clipped_arr.dtype),
                    "count": clipped_arr.shape[0],
                }
            )

            memfile = MemoryFile()
            memfiles.append(memfile)

            ds = memfile.open(**clipped_profile)
            ds.write(clipped_arr)
            srcs.append(ds)

        if not srcs:
            raise ValueError(
                f"No clipped rasters intersected the clipping polygon in folder: {in_dir}"
            )

        print("\nMerging clipped DEMs:")
        for tif in tifs:
            print(f"  - {tif.name}")

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
        for ds in srcs:
            ds.close()
        for memfile in memfiles:
            memfile.close()


def merge_folders_to_parent_named_outputs(
    input_folders: Sequence[Union[str, Path]],
    clip_vector: Union[str, Path],
    nodata: Optional[float] = -9999.0,
    overwrite: bool = False,
    sanity_print: bool = True,
) -> list[Path]:
    """
    For each folder in `input_folders`, clip its .tifs by `clip_vector` and write:
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
        print(f"CLIP  : {Path(clip_vector).resolve()}")
        print(f"OUTPUT: {out_dir / out_name}")

        out_path = merge_folder_to_dem(
            input_folder=folder,
            output_folder=out_dir,
            clip_vector=clip_vector,
            out_name=out_name,
            nodata=nodata,
            overwrite=overwrite,
            sanity_print=sanity_print,
        )
        outputs.append(out_path)

    return outputs


if __name__ == "__main__":
    clip_vector = (
        r"C:\L\Lichen\Lichen - Documents\Projects\20240001.4_Tucan 5-15 (CTUIR)\07_GIS\Wenaha\Tucannon REM\tucannon_buffer.gpkg"
    )

    input_folders = [
        r"C:\Users\AlexThornton-Dunwood\Downloads\custom_download\datasetsA\columbia_garfield_walla_2018\dsm",
        r"C:\Users\AlexThornton-Dunwood\Downloads\custom_download\datasetsA\columbia_garfield_walla_2018\dtm"
    ]

    merge_folders_to_parent_named_outputs(
        input_folders=input_folders,
        clip_vector=clip_vector,
        nodata=0,
        overwrite=True,
        sanity_print=True,
    )