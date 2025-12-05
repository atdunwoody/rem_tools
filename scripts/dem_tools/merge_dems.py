from pathlib import Path
from typing import Optional, Sequence, Tuple
import numpy as np
import rasterio
from rasterio.merge import merge as rio_merge
from rasterio.enums import Resampling

def merge_folder_to_dem(
    input_folder: str,
    output_folder: str,
    out_name: str = "merged_dem.tif",
    nodata: Optional[float] = None,
    overwrite: bool = False,
) -> Path:
    """
    Merge all GeoTIFFs in `input_folder` into a single DEM in `output_folder`.

    Parameters
    ----------
    input_folder : str
        Directory containing input .tif files (non-recursive).
    output_folder : str
        Directory to write the merged raster.
    out_name : str, optional
        Output filename (default "merged_dem.tif").
    nodata : float, optional
        NoData value to assign in the output. If None, will use the first
        source's nodata; if that is None, defaults to -9999.0.
    overwrite : bool, optional
        Overwrite output if it exists.

    Returns
    -------
    Path
        Path to the written merged DEM.
    """
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name

    if out_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {out_path} (set overwrite=True to replace)")

    # Collect input tiles
    tifs: Sequence[Path] = sorted(p for p in in_dir.iterdir() if p.suffix.lower() == ".tif")
    if not tifs:
        raise FileNotFoundError(f"No .tif files found in: {in_dir}")

    # Open sources
    srcs = [rasterio.open(p) for p in tifs]
    try:
        # Determine nodata
        out_nodata = (
            nodata if nodata is not None
            else (srcs[0].nodata if srcs[0].nodata is not None else -9999.0)
        )

        print("Merging DEMs:")
        print(tifs)
        # Merge
        mosaic, transform = rio_merge(
            srcs,
            nodata=out_nodata,
            res=None,              # use native resolutions; assumes consistent inputs
            method="first",        # keep first where overlaps (change to 'last' if preferred)
            dtype=srcs[0].dtypes[0],
            resampling=Resampling.bilinear,
        )

        # Build output profile
        profile = srcs[0].profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "count": mosaic.shape[0],
                "nodata": out_nodata,
                # sensible defaults for DEMs
                "compress": "deflate",
                "predictor": 2,         # good for continuous data
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "BIGTIFF": "IF_SAFER",
            }
        )
        print(f"Writing output to {out_path}")
        # Write output
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mosaic)

        # Print basic info
        with rasterio.open(out_path) as ds:
            bounds = ds.bounds
            res: Tuple[float, float] = ds.res
            data = ds.read(1, masked=True)
            # Mask NoData explicitly in case nodata is not recognized by driver read
            if ds.nodata is not None:
                data = np.ma.masked_equal(data, ds.nodata)

            print("=== Merged DEM ===")
            print(f"Path:    {out_path}")
            print(f"Driver:  {ds.driver}")
            print(f"CRS:     {ds.crs}")
            print(f"Size:    {ds.width} x {ds.height} px")
            print(f"Res:     {res[0]} x {res[1]} (map units/pixel)")
            print(f"Bands:   {ds.count}")
            print(f"NoData:  {ds.nodata}")
            print(f"Bounds:  left={bounds.left:.3f}, bottom={bounds.bottom:.3f}, "
                  f"right={bounds.right:.3f}, top={bounds.top:.3f}")
            # Basic stats (ignoring NoData)
            if data.count() > 0:
                arr = data.compressed()
                print(f"Stats:   min={arr.min():.3f}, max={arr.max():.3f}, "
                      f"mean={arr.mean():.3f}, std={arr.std(ddof=1):.3f}, "
                      f"valid_px={arr.size}")
            else:
                print("Stats:   all NoData")

        return out_path

    finally:
        for s in srcs:
            s.close()


if __name__ == "__main__":
    input_folder = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\DEMs\Trout-Bear Cr"
    output_folder = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\DEMs\Trout-Bear Cr"

    merge_folder_to_dem(
        input_folder=input_folder,
        output_folder=output_folder,
        out_name="TB_DEM_merge.tif",
        nodata=None,         # use first tile's nodata if present, else -9999.0
        overwrite=True,      # set False to prevent accidental overwrite
    )
