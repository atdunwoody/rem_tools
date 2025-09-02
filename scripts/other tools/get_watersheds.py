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

def get_watersheds(d8_pntr, output_dir, watershed_name, pour_points=None, watershed_join_field='DN',
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
        

    else:
        raise ValueError("Must provide stream_raster for pour points.")

    
    pour_points_snapped = os.path.join(output_dir, f"{watershed_name}_pour_points_snapped.shp")
    
    print("Snapping pour points to streams...")
    wbt.jenson_snap_pour_points(
        pour_pts=pour_points,
        streams=stream_raster,
        output=pour_points_snapped,
        snap_dist=50,
    )
    pour_points_snapped_gpkg = os.path.join(output_dir, f"{watershed_name}_pour_points_snapped.gpkg")
    pour_points_snapped_gdf = gpd.read_file(pour_points_snapped)
    pour_points_snapped_gdf.to_file(pour_points_snapped_gpkg, driver='GPKG')
    
    watershed_raster = os.path.join(output_dir, f"{watershed_name}_watersheds.tif")
    watershed_vector = os.path.join(output_dir, f"{watershed_name}_watersheds.gpkg")
    unnested_watersheds = os.path.join(output_dir, f"{watershed_name}_unnested_watersheds.gpkg")

    if not os.path.exists(watershed_raster):
        print("Generating watershed raster...")
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
    pts = gpd.read_file(pour_points)
    ws  = gpd.read_file(watershed_vector)
    output_gpkg = os.path.join(output_dir, f"{watershed_name}_watersheds_with_elev.gpkg")
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

def aggregate_watersheds(pour_points_gpkg_path, wbt_watersheds_gpkg_path, output_gpkg_path, watershed_join_field='WS_ID'):
    """
    Processes pour points and watershed polygons to assign Watershed IDs and saves all watersheds
    as separate polygons within the same layer in the output GeoPackage.
    Additionally, saves the pour points with the assigned Watershed_ID in a new GeoPackage.

    :param pour_points_gpkg_path: Path to the pour points GeoPackage used to create watersheds GeoPackage.
    :param wbt_watersheds_gpkg_path: Path to the watersheds GeoPackage created by get_watersheds.py.
    :param output_gpkg_path: Path to the output GeoPackage for watershed polygons.
    :param watershed_join_field: Field name used to join watersheds to points (default 'DN').
    """
    
    # Load the points GeoPackage
    print(f"Loading points from {pour_points_gpkg_path}...")
    points_layers = fiona.listlayers(pour_points_gpkg_path)
    if not points_layers:
        raise ValueError("No layers found in the points GeoPackage.")
    points_gdf = gpd.read_file(pour_points_gpkg_path, layer=points_layers[0])
    
    # Load the polygons GeoPackage
    print(f"Loading polygons from {wbt_watersheds_gpkg_path}...")
    polygons_layers = fiona.listlayers(wbt_watersheds_gpkg_path)
    if not polygons_layers:
        raise ValueError("No layers found in the polygons GeoPackage.")
    polygons_gdf = gpd.read_file(wbt_watersheds_gpkg_path, layer=polygons_layers[0])
    
    # Check for join field in polygons
    if watershed_join_field not in polygons_gdf.columns:
        print(f"Columns in wbt_watersheds_gpkg_path:\n {polygons_gdf.columns}")
        print(f"Columns in pour_points_gpkg_path:\n {points_gdf.columns}")
        print(f"Join field '{watershed_join_field}' not found in polygons GeoDataFrame.")
        raise ValueError(
            f"The polygons GeoPackage does not contain the specified join_field. \n"
            f"Specify the correct field name in the 'watershed_join_field' parameter."
        )
    
    # Ensure both GeoDataFrames use the same CRS
    if points_gdf.crs != polygons_gdf.crs:
        print("CRS mismatch between points and polygons. Reprojecting points to match polygons CRS.")
        points_gdf = points_gdf.to_crs(polygons_gdf.crs)
    
    # Perform spatial join to assign watershed_join_field to each point
    print("Performing spatial join to assign watershed IDs to points based on watershed polygons...")
    points_gdf = gpd.sjoin(
        points_gdf, 
        polygons_gdf[[watershed_join_field, 'geometry']], 
        how='left', 
        predicate='within'
    )
    
    print(f"Points joined with watershed polygons using '{watershed_join_field}' field.")
    print(f"Columns in the points gdf: {points_gdf.columns}")
    print(f"Columns in the polygons gdf: {polygons_gdf.columns}")
    
    # Rename the watershed_join_field to Watershed_ID for clarity
    points_gdf = points_gdf.rename(columns={watershed_join_field: 'WS_ID'})
    points_gdf = points_gdf.rename(columns={'WS_ID_right': 'WS_ID'})
    print(f"Points gdf renamed to include 'WS_ID' column.")
    print(f"Head of points gdf:\n {points_gdf.head()}")
    
    # Handle points that did not match any polygon
    missing_ids = points_gdf['WS_ID'].isna().sum()
    if missing_ids > 0:
        print(f"Warning: {missing_ids} points did not match any watershed polygon and will have Watershed_ID set to NaN.")
    
    # Initialize a list to store watershed polygons with their IDs
    watershed_list = []
    
    # Iterate over each point with a progress bar
    print("Processing Points and aggregating Watershed Polygons...")
    for idx, point in tqdm(points_gdf.iterrows(), total=points_gdf.shape[0], desc="Processing Points"):
        # Get Watershed_ID from the point
        watershed_id = point['WS_ID']

        if pd.isna(watershed_id):
            # Skip points without a valid Watershed ID
            continue

        try:
            # Ensure watershed_id is integer for comparison
            watershed_id_int = int(watershed_id)
        except ValueError:
            print(f"  Point at index {idx} has a non-integer Watershed_ID value '{watershed_id}'. Skipping.")
            continue

        # Filter polygons where watershed_join_field >= watershed_id_int
        filtered_polygons = polygons_gdf[polygons_gdf[watershed_join_field] >= watershed_id_int].copy()
        if filtered_polygons.empty:
            print(f"  No polygons found for Watershed_ID {watershed_id_int} with {watershed_join_field} >= {watershed_id_int}. Skipping.")
            continue  # Skip if no matching polygons

        # Fix invalid geometries using buffer(0)
        filtered_polygons['geometry'] = filtered_polygons.geometry.buffer(0)
        
        # Combine the filtered polygons into a single geometry
        combined_geometry = filtered_polygons.unary_union
        # Create a new entry with the combined geometry and Watershed ID
        watershed_entry = {
            'WS_ID': watershed_id_int,
            'geometry': combined_geometry
        }
        watershed_list.append(watershed_entry)
    
    if not watershed_list:
        print("No watershed polygons were created. Exiting without creating output GeoPackage.")
        return
    
    # Create a GeoDataFrame from the watershed list
    watershed_gdf = gpd.GeoDataFrame(watershed_list, crs=polygons_gdf.crs)
    
    # Remove any potential invalid geometries
    watershed_gdf['geometry'] = watershed_gdf['geometry'].buffer(0)
    
    # Ensure no duplicate Watershed_IDs
    if watershed_gdf['WS_ID'].duplicated().any():
        print("\nWARNING: Duplicate Watershed_IDs found. Each Watershed_ID should be unique.")
        duplicate_ids = watershed_gdf[watershed_gdf['WS_ID'].duplicated()]['WS_ID'].unique()
        print(f"Duplicate Watershed_IDs: {duplicate_ids}")
        print("Removing duplicate Watershed_IDs...\n")
        watershed_gdf = watershed_gdf.drop_duplicates(subset='WS_ID')
    
    # Save all watersheds to a single layer in the output GeoPackage
    watershed_gdf.to_file(
        output_gpkg_path,
        driver="GPKG"
    )
    
    print(f"All watershed polygons have been processed and saved to {output_gpkg_path}.")
    
    # Define the path for the new pour points GeoPackage
    ws_name = os.path.basename(output_gpkg_path).split('_')[0]
    pour_point_with_id_path = os.path.join(
        os.path.dirname(output_gpkg_path), 
        f'{ws_name}_pour_points_with_wsid.gpkg'
    )
    
    print(f"Saving pour points with Watershed_ID to {pour_point_with_id_path} as layer 'pour_points'...")
    
    # Select relevant columns (including Watershed_ID) for saving
    # Optionally, drop the index_right column from spatial join if present
    if 'index_right' in points_gdf.columns:
        points_gdf = points_gdf.drop(columns=['index_right'])
    
    # Ensure Watershed_ID is properly typed
    points_gdf['WS_ID'] = points_gdf['WS_ID'].astype(pd.Int64Dtype())
    
    # Save the pour points with Watershed_ID to the new GeoPackage
    points_gdf.to_file(
        pour_point_with_id_path,
        layer='pour_points',
        driver='GPKG'
    )
    
    print(f"Pour points with Watershed_ID have been saved to {pour_point_with_id_path} in the 'pour_points' layer.")

import geopandas as gpd
from shapely.ops import unary_union

def combine_features_by_ws_id(
    input_gpkg_path: str,
    output_gpkg_path: str,) -> None:
    """
    Dissolve all features in a GeoPackage by the 'WS_ID' field.

    Parameters
    ----------
    input_gpkg_path : str
        Path to the source GeoPackage.
    output_gpkg_path : str
        Path to the destination GeoPackage.
    input_layer : str, optional
        Name of the layer to read from the input GeoPackage.
        If None, the default (first) layer is used.
    output_layer : str, default "dissolved_ws"
        Name of the layer to create in the output GeoPackage.
    """
    # Read input
 
    gdf = gpd.read_file(input_gpkg_path)

    # Group by WS_ID and union geometries
    dissolved = (
        gdf
        .groupby("WS_ID")["geometry"]
        .apply(unary_union)
        .reset_index()
    )

    # Create a GeoDataFrame (preserving CRS)
    dissolved_gdf = gpd.GeoDataFrame(
        dissolved,
        geometry="geometry",
        crs=gdf.crs
    )

    # Write to output GeoPackage
    dissolved_gdf.to_file(
        output_gpkg_path,
        driver="GPKG"
    )



if __name__ == "__main__":
    
    d8_pointer = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\REM\Streams\d8_pointer.tif"

    stream_raster = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\REM\Streams\streams_5k.tif"
    pour_pts = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\GitHub\BSR_viewer\data\inputs\CRITFC FLIR\CCR_2010_FLIR_CRITFC_corr_UTM.gpkg"
    dem = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\LiDAR\rasters_USGS10m\USGS 10m DEM Clip.tif"
    flow_accum = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\Web App Processing\FLIR Data\flow_accum.tif"
    output_dir = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\Web App Processing"
    

    output_gpkg = os.path.join(output_dir, "CCR_2010_FLIR_CRITFC_corr_with_DA.gpkg")
    # add_DA_from_flow_accum(
    #     points_gpkg=pour_pts,
    #     flow_accum_raster=flow_accum,
    #     out_gpkg=output_gpkg
    # )

    add_point_elevations_from_dem(
        points_gpkg=output_gpkg,
        dem_path=dem,
        out_gpkg=output_gpkg
    )

    # get_watersheds(
    #     d8_pntr=d8_pointer,
    #     output_dir=output_dir,
    #     watershed_name="CCR",
    #     pour_points=pour_pts, 
    #     stream_raster=stream_raster,
    # )

    # wbt_watersheds = os.path.join(output_dir, "CCR_watersheds.gpkg")
    # output_gpkg = os.path.join(output_dir, "CCR_watersheds_dissolved.gpkg")
    
    # combine_features_by_ws_id(
    #     input_gpkg_path=wbt_watersheds,
    #     output_gpkg_path=output_gpkg
    # )


    # perps = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\transects_200m.gpkg"
    # stream_vector = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\Streams Clipped to LiDAR\streams_100k_clipped_to_LiDAR.gpkg"
    # get_watersheds(        
    #     d8_pntr=d8_pointer,
    #     output_dir=output_dir,
    #     watershed_name="GRMW",
    #     stream_vector=stream_vector,
    #     stream_raster=stream_raster,
    #     perpendiculars=perps,
    # )