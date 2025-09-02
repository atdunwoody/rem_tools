
# get_watersheds.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Literal
import numpy as np
import geopandas as gpd
import fiona
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.mask import mask
from shapely.geometry import mapping



def add_point_elevations_from_dem(
    points_gpkg: str,
    dem_path: str,
    *,
    out_gpkg: str,
    points_layer: Optional[str] = None,
    out_layer: Optional[str] = None,
    elevation_field: str = "elevation",
    band: int = 1,
    sample_method: Literal["nearest", "bilinear", "cubic"] = "bilinear",
    update_nulls_if_exists: bool = True,
    chunk_size: int = 50000,
) -> str:
    """
    Add DEM elevations to point features.

    I/O
    ---
    - Reads points from `points_gpkg` (GeoPackage). If `points_layer` is not provided,
      and the GPKG contains exactly one layer, it is used. Otherwise an error is raised.
    - Writes to `out_gpkg` under `out_layer` (defaults to input layer name). If the
      target layer already exists, it is replaced.

    Behavior
    --------
    - Samples DEM values at point locations (default `nearest`), using `band`.
    - Creates a new column named `elevation_field` (default "elevation") if it doesn't
      exist. If it exists:
        * when `update_nulls_if_exists=True`, only fills rows where it is null;
        * otherwise, raises a ValueError and does not modify values.
    - Preserves the original point CRS in the output. Points are reprojected to the DEM
      CRS only for sampling.

    Assumptions
    -----------
    - DEM vertical units are meters.
    - Input layer geometries are Points. (MultiPoint/other types raise an error.)

    Returns
    -------
    str
        Path to `out_gpkg`.
    """
    # ---- Validate paths
    pts_path = Path(points_gpkg)
    dem_fp = Path(dem_path)
    out_path = Path(out_gpkg)
    if not pts_path.exists():
        raise FileNotFoundError(f"Points GPKG not found: {pts_path}")
    if not dem_fp.exists():
        raise FileNotFoundError(f"DEM not found: {dem_fp}")

    # ---- Resolve layer
    layers = fiona.listlayers(points_gpkg)
    if points_layer is None:
        if len(layers) == 1:
            points_layer = layers[0]
        else:
            raise ValueError(
                f"`points_layer` not provided and GPKG has {len(layers)} layers: {layers}"
            )
    if points_layer not in layers:
        raise ValueError(f"Layer '{points_layer}' not found in {points_gpkg}. Available: {layers}")

    out_layer = out_layer or points_layer

    # ---- Read points
    gdf = gpd.read_file(points_gpkg, layer=points_layer)
    if gdf.empty:
        raise ValueError("Input points layer is empty.")
    if gdf.geometry.is_empty.any():
        raise ValueError("Some point geometries are empty; clean the input and retry.")
    if gdf.crs is None:
        raise ValueError("Input points have no CRS; define a CRS before sampling.")
    # Geometry type check: strictly Points
    geom_types = set(gdf.geometry.geom_type.unique())
    if not geom_types.issubset({"Point"}):
        raise TypeError(f"Layer must contain only Points. Found geometry types: {sorted(geom_types)}")

    # ---- Open DEM
    with rasterio.open(dem_path) as src:
        if src.count < band or band < 1:
            raise ValueError(f"DEM band {band} invalid. DEM has {src.count} band(s).")
        if src.crs is None:
            raise ValueError("DEM has no CRS; cannot reproject points for sampling.")

        dem_crs = src.crs

        # Build a sampling dataset (optionally through a VRT to change resampling)
        if sample_method == "nearest":
            samp_ds = src
            close_ds = lambda: None  # no-op
        else:
            res_map = {
                "bilinear": Resampling.bilinear,
                "cubic": Resampling.cubic,
            }
            if sample_method not in res_map:
                raise ValueError(f"Unsupported sample_method: {sample_method}")
            vrt = WarpedVRT(src, resampling=res_map[sample_method])
            samp_ds = vrt
            close_ds = vrt.close  # to close VRT later

        try:
            # ---- Reproject points to DEM CRS for sampling
            pts_samp = gdf.to_crs(dem_crs)
            xs = pts_samp.geometry.x.to_numpy()
            ys = pts_samp.geometry.y.to_numpy()

            # ---- Chunked sampling (memory-safe)
            n = len(pts_samp)
            elevations = np.full(n, np.nan, dtype="float64")
            nodata = samp_ds.nodata

            # Use masked sampling to respect NoData if available
            masked = True
            for start in range(0, n, max(1, chunk_size)):
                stop = min(start + chunk_size, n)
                coords = list(zip(xs[start:stop], ys[start:stop]))
                # Each yielded sample is a 1-element masked array for the requested band
                for idx, val in enumerate(samp_ds.sample(coords, indexes=band, masked=masked)):
                    # val can be numpy array or masked array of shape (1,)
                    v = val[0]
                    if getattr(v, "mask", False) is True:
                        elevations[start + idx] = np.nan
                    else:
                        vv = float(v)
                        if nodata is not None and np.isfinite(nodata) and vv == nodata:
                            elevations[start + idx] = np.nan
                        else:
                            elevations[start + idx] = vv
        finally:
            close_ds()

    # ---- Attach to original GDF (retain original CRS/attrs)
    if elevation_field not in gdf.columns:
        gdf[elevation_field] = elevations
    else:
        if update_nulls_if_exists:
            # only fill where existing is NaN
            mask = gdf[elevation_field].isna()
            gdf.loc[mask, elevation_field] = elevations[mask.to_numpy()]
        else:
            raise ValueError(
                f"Field '{elevation_field}' already exists. "
                "Set update_nulls_if_exists=True to fill only missing values."
            )

    # ---- Write to output (replace layer if present)
    if out_path.exists():
        existing_out_layers = set(fiona.listlayers(out_gpkg))
        if out_layer in existing_out_layers:
            # Remove existing layer to replace it cleanly
            fiona.remove(out_gpkg, driver="GPKG", layer=out_layer)

    gdf.to_file(out_gpkg, layer=out_layer, driver="GPKG")
    return str(out_path)




def add_DA_from_flow_accum(
    points_gpkg: str,
    flow_accum_raster: str,
    out_gpkg: str,
    *,
    points_layer: str | None = None,
    out_layer: str | None = None,
    buffer_m: float = 100.0
) -> str:
    """
    Adds a 'DA_km2' field to points by buffering each point and extracting
    the maximum flow accumulation value within the buffer.

    Parameters
    ----------
    points_gpkg : str
        Path to input GeoPackage with point features.
    flow_accum_raster : str
        Path to flow accumulation raster (units = m²).
    out_gpkg : str
        Path to output GeoPackage with updated points.
    points_layer : str, optional
        Layer name of points in input GeoPackage.
    out_layer : str, optional
        Output layer name. Defaults to "<points_layer>_DA".
    buffer_m : float
        Buffer radius in meters for sampling flow accumulation.

    Returns
    -------
    str
        Path to output GeoPackage.
    """
    # Load points
    gdf = gpd.read_file(points_gpkg, layer=points_layer)
    if gdf.empty:
        raise ValueError("Input point layer is empty.")

    # Open raster
    with rasterio.open(flow_accum_raster) as src:
        raster_crs = src.crs

        # Reproject points if needed
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)

        da_vals = []
        for geom in gdf.geometry:
            if geom.is_empty:
                da_vals.append(np.nan)
                continue

            # Buffer point
            buf = geom.buffer(buffer_m)

            # Mask raster with buffer polygon
            try:
                out_image, _ = rasterio.mask.mask(src, [mapping(buf)], crop=True)
                out_data = out_image[0]
                # Get max ignoring nodata
                if np.all(out_data == src.nodata) or out_data.size == 0:
                    da_vals.append(np.nan)
                else:
                    da_vals.append(float(np.nanmax(np.where(out_data == src.nodata, np.nan, out_data))))
            except ValueError:
                da_vals.append(np.nan)

    # Convert to km²
    gdf["DA_km2"] = np.array(da_vals) / 1_000_000.0

    # Save
    out_layer = out_layer or (points_layer + "_DA" if points_layer else "points_DA")
    gdf.to_file(out_gpkg, layer=out_layer, driver="GPKG")
    return out_gpkg


if __name__ == "__main__":
    

    pour_pts = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\Web App Processing\FLIR Data\CCR_2010_FLIR_CRITFC.gpkg"
    pour_pts_ugr = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\Web App Processing\FLIR Data\UGR_2010_FLIR_CRITFC.gpkg"
    dem = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\LiDAR\rasters_USGS10m\USGS 10m DEM Clip.tif"
    flow_accum = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\Web App Processing\FLIR Data\burned_streams\flow_accum.tif"
    output_dir = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\Web App Processing"
    

    output_gpkg = os.path.join(output_dir, "UGR_2010_FLIR_CRITFC_elev-DA.gpkg")
    add_DA_from_flow_accum(
        points_gpkg=pour_pts_ugr,
        flow_accum_raster=flow_accum,
        out_gpkg=output_gpkg,
        buffer_m=200.0
    )

    add_point_elevations_from_dem(
        points_gpkg=output_gpkg,
        dem_path=dem,
        out_gpkg=output_gpkg
    )
