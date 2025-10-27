# get_watersheds.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Literal
import numpy as np
import geopandas as gpd
import fiona
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.mask import mask
from shapely.geometry import mapping


import os
import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape, LineString
from shapely.ops import unary_union, split
from whitebox_workflows import WbEnvironment
import whitebox
from osgeo import ogr

import fiona
from tqdm import tqdm  # Import tqdm for progress bar
import pandas as pd
import warnings

def get_watersheds(d8_pntr, output_dir, watershed_name = None, pour_points=None, pour_points_snapped=None,
                   watershed_join_field='DN',
                      stream_raster=None, stream_vector=None, perpendiculars=None, aggregate=True):
    """
    Processes a list of DEM files to extract streams and convert them to GeoPackage format.

    Parameters:
    - d8_pntr (str): Path to the D8 pointer raster file.
    - output_dir (str): Path to the output directory.
    - pour_points (str, Optional): Path to the pour points shapefile or GeoPackage.
    - stream_vector (str, Optional): Path to the stream vector file.
    - perpendiculars (str, Optional): Path to the perpendiculars vector file.

    Returns:
    - None
    """
    print("Processing Watersheds...")
    # Initialize WhiteboxTools
    wbt = whitebox.WhiteboxTools()

    # Set the environment
    wbe = WbEnvironment()

    # Create working directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check if pour points is a shapefile, if not, convert it to a shapefile
    if pour_points is not None:
        if pour_points.endswith('.shp'):
            pass
        else:
            pour_points_gdf = gpd.read_file(pour_points)
            pour_points_name = os.path.basename(pour_points).replace('.gpkg', '.shp')
            pour_points_shp = os.path.join(output_dir, pour_points_name)
            pour_points_gdf.to_file(pour_points_shp)
            pour_points = pour_points_shp
    
    if perpendiculars is not None or stream_vector is not None:
        pour_points = os.path.join(output_dir, "pour_points.shp")
        find_intersections(stream_vector, perpendiculars, pour_points)
    elif stream_raster is not None and pour_points is not None:
        #check if crs match
        pour_points_gdf = gpd.read_file(pour_points)
        with rasterio.open(stream_raster) as src:
            stream_crs = src.crs
        if pour_points_gdf.crs != stream_crs:
            print("Reprojecting pour points to match stream raster CRS...")
            pour_points_gdf = pour_points_gdf.to_crs(stream_crs)
            pour_points_gdf.to_file(pour_points)
        else:
            print("CRS match confirmed.")
        

    if pour_points_snapped is None:
        if watershed_name is None:
            pour_points_snapped = os.path.join(output_dir, f"pour_points_snapped.shp")
        else:
            pour_points_snapped = os.path.join(output_dir, f"{watershed_name}_pour_points_snapped.shp")
    
        print("Snapping pour points to streams...")
        wbt.jenson_snap_pour_points(
            pour_pts=pour_points,
            streams=stream_raster,
            output=pour_points_snapped,
            snap_dist=50,
            )
        if watershed_name is not None:
            pour_points_snapped_gpkg = os.path.join(output_dir, f"{watershed_name}_pour_points_snapped.gpkg")
        else:
            pour_points_snapped_gpkg = os.path.join(output_dir, f"pour_points_snapped.gpkg")
        pour_points_snapped_gdf = gpd.read_file(pour_points_snapped)
        pour_points_snapped_gdf.to_file(pour_points_snapped_gpkg, driver='GPKG')

    pour_points_snapped_elev = os.path.join(output_dir, "pour_points_snapped_with_elev.shp")
    add_point_elevations_from_dem(
        points_gpkg=pour_points_snapped,
        dem_path=dem,
        out_gpkg=pour_points_snapped_elev
    )
    
    if watershed_name is not None:
        watershed_raster = os.path.join(output_dir, f"{watershed_name}_watersheds.tif")
        watershed_vector = os.path.join(output_dir, f"{watershed_name}_watersheds.gpkg")
        unnested_watersheds = os.path.join(output_dir, f"{watershed_name}_unnested_watersheds.gpkg")
    else:
        watershed_raster = os.path.join(output_dir, f"watersheds.tif")
        watershed_vector = os.path.join(output_dir, f"watersheds.gpkg")
        unnested_watersheds = os.path.join(output_dir, f"unnested_watersheds.gpkg")

    if not os.path.exists(watershed_raster):
        print("Generating watershed raster...")
        print("D8 Pointer:", d8_pntr)
        print("Pour Points Snapped:", pour_points_snapped)
        wbt.watershed(
            d8_pntr,
            pour_points_snapped,
            watershed_raster,
        )

    
    if not os.path.exists(watershed_vector):
        print("Polygonizing watershed raster...")
        polygonize_raster(watershed_raster, watershed_vector, attribute_name='WS_ID')
    
    #####################################################
    ################ ADD WSE TO WATERSHEDS ################
    #####################################################

    # 1. read
    pts = gpd.read_file(pour_points_snapped)
    ws  = gpd.read_file(watershed_vector)
    if watershed_name is not None:
        output_gpkg = os.path.join(output_dir, f"{watershed_name}_watersheds_with_elev.gpkg")
    else:
        output_gpkg = os.path.join(output_dir, f"watersheds_with_elev.gpkg")
    # 2. turn the DataFrame index (which corresponds to the GPKG FID) into a column named 'fid'
    pts = pts.reset_index().rename(columns={'index': 'fid'})

    # 3. merge the elevation into ws
    ws = ws.merge(
        pts[['fid', 'elevation']],
        left_on='WS_ID',
        right_on='fid',
        how='left'
    ).drop(columns=['fid'])

    # 4. write out
    ws.to_file(output_gpkg, driver="GPKG")
    return pour_points_snapped_gpkg, watershed_vector

def find_intersections(centerline_file, perpendiculars_file, output_file):
    """
    Finds intersection points between centerline and perpendiculars vector files.

    Parameters:
    - centerline_file (str): Path to the centerline vector file (.shp or .gpkg).
    - perpendiculars_file (str): Path to the perpendiculars vector file (.shp or .gpkg).
    - output_file (str): Path for the output points file (.shp or .gpkg).

    Returns:
    - GeoDataFrame containing the intersection points.
    """
    # Validate input file formats
    valid_extensions = ['.shp', '.gpkg']
    center_ext = os.path.splitext(centerline_file)[1].lower()
    perp_ext = os.path.splitext(perpendiculars_file)[1].lower()
    output_ext = os.path.splitext(output_file)[1].lower()

    if center_ext not in valid_extensions:
        raise ValueError(f"Centerline file must be one of {valid_extensions}, got {center_ext}")
    if perp_ext not in valid_extensions:
        raise ValueError(f"Perpendiculars file must be one of {valid_extensions}, got {perp_ext}")
    if output_ext not in valid_extensions:
        raise ValueError(f"Output file must be one of {valid_extensions}, got {output_ext}")

    # Read the centerline and perpendiculars
    center_gdf = gpd.read_file(centerline_file)
    perp_gdf = gpd.read_file(perpendiculars_file)

    # Ensure both GeoDataFrames have the same CRS
    if center_gdf.crs != perp_gdf.crs:
        print("CRS mismatch detected. Reprojecting perpendiculars to match centerline CRS.")
        perp_gdf = perp_gdf.to_crs(center_gdf.crs)

    # Perform spatial join using intersection
    # This can be resource-intensive for large datasets
    intersections = []

    # To optimize, create a spatial index on perpendiculars
    perp_sindex = perp_gdf.sindex

    for idx, center_geom in center_gdf.geometry.items():
        # Potential matches using spatial index
        possible_matches_index = list(perp_sindex.intersection(center_geom.bounds))
        possible_matches = perp_gdf.iloc[possible_matches_index]

        for _, perp_geom in possible_matches.geometry.items():
            if center_geom.intersects(perp_geom):
                intersection = center_geom.intersection(perp_geom)
                if "Point" == intersection.geom_type:
                    intersections.append(intersection)
                elif "MultiPoint" == intersection.geom_type:
                    intersections.extend(intersection.geoms)
                # Handle other geometry types if necessary

    if not intersections:
        print("No intersections found.")
        return None

    # Create a GeoDataFrame from the intersection points
    intersection_gdf = gpd.GeoDataFrame(geometry=intersections, crs=center_gdf.crs)

    # Optionally, remove duplicate points
    intersection_gdf = intersection_gdf.drop_duplicates()

    # Save to the desired output format
    if output_ext == '.shp':
        intersection_gdf.to_file(output_file, driver='ESRI Shapefile')
    elif output_ext == '.gpkg':
        intersection_gdf.to_file(output_file, driver='GPKG')
    else:
        raise ValueError(f"Unsupported output file format: {output_ext}")

    print(f"Intersection points saved to {output_file}")
    return intersection_gdf

def polygonize_raster(raster_path, vector_path, attribute_name='WS_ID'):
    """
    Polygonizes a raster file and saves it as a vector file.

    Parameters:
    - raster_path (str): Path to the input raster file.
    - vector_path (str): Path to the output vector file (.shp or .gpkg).
    - attribute_name (str): Name of the attribute to store raster values.

    Returns:
    - GeoDataFrame of the polygonized raster.
    """
    with rasterio.open(raster_path) as src:
        image = src.read(1)  # Read the first band
        mask = image != src.nodata  # Create a mask for valid data

        print("Starting polygonization of the raster...")
        results = (
            {'properties': {attribute_name: v}, 'geometry': shape(s)}
            for s, v in shapes(image, mask=mask, transform=src.transform)
        )
        geoms = list(results)
        print(f"Extracted {len(geoms)} polygons from the raster.")

    # Create a GeoDataFrame from the shapes
    gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)

    # gdf = replace_dn_with_ws_id(gdf)
    # Save to the desired vector format
    vector_ext = os.path.splitext(vector_path)[1].lower()
    if vector_ext == '.shp':
        gdf.to_file(vector_path, driver='ESRI Shapefile')
    elif vector_ext == '.gpkg':
        gdf.to_file(vector_path, driver='GPKG')
    else:
        raise ValueError(f"Unsupported vector file format: {vector_ext}")

    print(f"Polygonized raster saved to {vector_path}")
    return gdf


from pathlib import Path
from typing import Optional, Literal, Tuple, Dict

import fiona
import geopandas as gpd
import pandas as pd


def aggregate_watersheds(
    *,
    pour_points_gpkg: str,
    watersheds_gpkg: str,
    out_gpkg: str,
    points_layer: Optional[str] = None,
    watersheds_layer: Optional[str] = None,
    join_field: str = "WS_ID",
    # output controls
    out_mode: Literal["single", "per-layer"] = "single",
    out_layer_single: str = "watersheds",
    per_layer_prefix: str = "ws_",
    save_points_with_id: bool = True,
    out_points_layer: str = "pour_points_with_wsid",
) -> Tuple[str, Optional[str]]:
    """
    Build *accumulated* watersheds per pour point by unioning all polygons with
    `join_field >= WS_ID_at_point`, then write results to a GeoPackage.

    Selection rule
    --------------
    For a point with watershed ID T, select polygons where `join_field >= T`,
    dissolve them, and use that union as the point's accumulated watershed.

    Output
    ------
    - out_mode="single": one layer with one feature per unique WS_ID threshold.
    - out_mode="per-layer": one layer per unique WS_ID, named `{per_layer_prefix}{WS_ID}`.
    - Optionally writes the pour points (with attached WS_ID) to `out_points_layer`.
    """
    out_path = Path(out_gpkg)
    pp_path = Path(pour_points_gpkg)
    ws_path = Path(watersheds_gpkg)

    if not pp_path.exists():
        raise FileNotFoundError(f"Pour points GPKG not found: {pp_path}")
    if not ws_path.exists():
        raise FileNotFoundError(f"Watersheds GPKG not found: {ws_path}")

    # Resolve layers
    pp_layers = fiona.listlayers(str(pp_path))
    ws_layers = fiona.listlayers(str(ws_path))
    if points_layer is None:
        if len(pp_layers) != 1:
            raise ValueError(f"Specify points_layer; candidates: {pp_layers}")
        points_layer = pp_layers[0]
    if watersheds_layer is None:
        if len(ws_layers) != 1:
            raise ValueError(f"Specify watersheds_layer; candidates: {ws_layers}")
        watersheds_layer = ws_layers[0]

    # Read
    points = gpd.read_file(pp_path, layer=points_layer)
    polys = gpd.read_file(ws_path, layer=watersheds_layer)

    if points.empty:
        raise ValueError("Pour points layer is empty.")
    if polys.empty:
        raise ValueError("Watersheds layer is empty.")
    if join_field not in polys.columns:
        raise ValueError(
            f"Watersheds layer missing join_field '{join_field}'. "
            f"Columns: {list(polys.columns)}"
        )
    if points.crs is None or polys.crs is None:
        raise ValueError("Points and polygons must have defined CRS.")
    if points.crs != polys.crs:
        points = points.to_crs(polys.crs)

    # Spatial join: attach WS_ID at point location (from polygons)
    pts_join = gpd.sjoin(points, polys[[join_field, "geometry"]], how="left", predicate="within")
    # Normalize to 'WS_ID' column name
    if join_field in pts_join.columns:
        pts_join = pts_join.rename(columns={join_field: "WS_ID"})
    elif f"{join_field}_right" in pts_join.columns:
        pts_join = pts_join.rename(columns={f"{join_field}_right": "WS_ID"})
    else:
        raise ValueError("Failed to locate joined watershed ID on points after spatial join.")

    n_unmatched = int(pts_join["WS_ID"].isna().sum())
    if n_unmatched:
        print(f"Warning: {n_unmatched} pour point(s) did not intersect a watershed polygon.")

    pts_join["WS_ID"] = pts_join["WS_ID"].astype(pd.Int64Dtype())
    # Unique thresholds we need to compute unions for
    thresholds = sorted(pd.unique(pts_join["WS_ID"].dropna().astype(int)))
    if not thresholds:
        raise ValueError("No valid WS_ID values found on pour points.")

    # Prepare polygons: fix validity and cache by >= threshold
    polys_valid = polys.copy()
    polys_valid["__id__"] = polys_valid[join_field]
    polys_valid["geometry"] = polys_valid.geometry.buffer(0)

    # Precompute unions for each threshold (>= selection)
    unions: Dict[int, gpd.GeoDataFrame] = {}
    for t in thresholds:
        sub = polys_valid[polys_valid["__id__"] <= t][["__id__", "geometry"]]
        if sub.empty:
            # Nothing upstream of this threshold; skip
            continue
        # Dissolve all selected polygons into a single feature
        dissolved = sub.dissolve(by=lambda _: 0, as_index=False)  # single-row GeoDataFrame
        dissolved = dissolved.drop(columns="__id__", errors="ignore")
        dissolved["WS_ID"] = int(t)
        unions[t] = dissolved[["WS_ID", "geometry"]]

    if not unions:
        raise ValueError("No accumulated watershed geometries were created.")

    # Combine into one GeoDataFrame
    ws_accum = gpd.GeoDataFrame(
        pd.concat([gdf for gdf in unions.values()], ignore_index=True),
        crs=polys_valid.crs,
    )
    # Final validity touch-up
    ws_accum["geometry"] = ws_accum["geometry"].buffer(0)

    # Write outputs
    if out_path.exists():
        existing = set(fiona.listlayers(str(out_path)))
        if out_mode == "single" and out_layer_single in existing:
            fiona.remove(str(out_path), driver="GPKG", layer=out_layer_single)

    if out_mode == "single":
        ws_accum.to_file(str(out_path), layer=out_layer_single, driver="GPKG")
        print(f"Wrote {len(ws_accum)} accumulated watershed(s) to layer '{out_layer_single}'.")
    else:
        # One layer per unique WS_ID threshold
        for _, row in ws_accum.iterrows():
            layer_name = f"{per_layer_prefix}{int(row['WS_ID'])}"
            sub = gpd.GeoDataFrame([row], geometry="geometry", crs=ws_accum.crs)
            existing = set(fiona.listlayers(str(out_path))) if out_path.exists() else set()
            if layer_name in existing:
                fiona.remove(str(out_path), driver="GPKG", layer=layer_name)
            sub.to_file(str(out_path), layer=layer_name, driver="GPKG")
        print(f"Wrote {len(ws_accum)} per-layer accumulated watershed(s) with prefix '{per_layer_prefix}'.")

    points_gpkg_written = None
    if save_points_with_id:
        if "index_right" in pts_join.columns:
            pts_join = pts_join.drop(columns=["index_right"])
        existing = set(fiona.listlayers(str(out_path))) if out_path.exists() else set()
        if out_points_layer in existing:
            fiona.remove(str(out_path), driver="GPKG", layer=out_points_layer)
        pts_join.to_file(str(out_path), layer=out_points_layer, driver="GPKG")
        points_gpkg_written = str(out_path)
        print(f"Wrote pour points (with WS_ID) to layer '{out_points_layer}'.")

    return str(out_path), points_gpkg_written



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

    gdf.to_file(out_gpkg)
    return str(out_path)

if __name__ == "__main__":
    
    d8_pointer = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240009_TVPI Irrigation Pipe (MSRF)\04_Analysis\Hydrology\Streams\d8_pointer.tif"
    flow_accum = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240009_TVPI Irrigation Pipe (MSRF)\04_Analysis\Hydrology\Streams\flow_accum.tif"
    stream_raster = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240009_TVPI Irrigation Pipe (MSRF)\04_Analysis\Hydrology\Streams\streams_7000k.tif"
    dem = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240009_TVPI Irrigation Pipe (MSRF)\07_GIS\LiDAR\output_USGS10m.tif"
    
    pour_pts = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240009_TVPI Irrigation Pipe (MSRF)\04_Analysis\Hydrology\Streams\TVPI mainstem discharge7-30-25.shp"
    output_dir = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Scratch"


    pour_pts_snapped_gpkg, watersheds_gpkg = get_watersheds(
                d8_pntr=d8_pointer,
                output_dir=output_dir,
                watershed_name=None,
                pour_points=pour_pts,
                #pour_points_snapped=pour_pts_snapped,
                stream_raster=stream_raster,
            )

    aggregated_watersheds = os.path.join(output_dir, "watersheds_dissolved.gpkg")
    aggregate_watersheds(pour_points_gpkg=pour_pts_snapped_gpkg,
                         watersheds_gpkg=watersheds_gpkg,
                         out_gpkg=aggregated_watersheds,
                         join_field='WS_ID',
                         out_mode = 'per-layer'
    )

