# dem_processor.py

from whitebox_workflows import WbEnvironment
import whitebox
import os
import geopandas as gpd
import rasterio

def get_flow_accumulation(dem, output_dir, overwrite=False):
    wbt = whitebox.WhiteboxTools()

    # Set the environment
    wbe = WbEnvironment()
    out_dem = os.path.join(output_dir, "dem filled.tif")
    out_pntr = os.path.join(output_dir, "d8 pointer.tif")
    out_accum = os.path.join(output_dir, "flow accumulation.tif")
    wbt.flow_accumulation_full_workflow(
        dem, 
        out_dem, 
        out_pntr, 
        out_accum, 
        out_type="Contributing Area", 
        )

def get_streams(dem, output_dir, threshold=100000, overwrite=False, breach_depressions=True):
    """
    Processes a list of DEM files to extract streams and convert them to GeoPackage format.

    Parameters:
    - dem (str): Path to the DEM file.
    - output_dir (str): Path to the output directory.
    - threshold (int, optional): Threshold value for stream extraction from flow accumulation raster. Default is 100000.
    - output_dir_base (str, optional): Base directory for output. Defaults to the DEM file's directory.
    - overwrite (bool, optional): If True, existing output directories will be overwritten. Default is False.

    Returns:
    - None
    """

    # Initialize WhiteboxTools
    wbt = whitebox.WhiteboxTools()

    # Set the environment
    wbe = WbEnvironment()

    # Create working directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif overwrite:
        # Clear existing directory
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)

    # wbt.set_working_dir(output_dir)

    # Define output filenames

    filled_dem = os.path.join(output_dir, "filled dem.tif")
    d8_pointer = os.path.join(output_dir, "d8 pointer.tif")
    flow_accum = os.path.join(output_dir, "flow accumulation.tif")
    breached_dem = os.path.join(output_dir, "breached dem.tif")
    
    # Skip if the file already exists
    if not os.path.exists(breached_dem) and breach_depressions:
        wbt.breach_depressions_least_cost(dem, breached_dem, 10)
    # Fill depressions in the DEM
    if not os.path.exists(filled_dem) and not breach_depressions:
        wbt.fill_depressions(dem, filled_dem)
    # Calculate flow direction on the filled DEM
    if breach_depressions and not os.path.exists(d8_pointer) and not os.path.exists(flow_accum): 
        print("Calculating flow direction and flow accumulation with breached dem...")
        wbt.d8_pointer(breached_dem, d8_pointer)
        wbt.d8_flow_accumulation(breached_dem, flow_accum)
    elif not os.path.exists(d8_pointer) and not os.path.exists(flow_accum):
        print("Calculating flow direction and flow accumulation with filled dem...")
        wbt.d8_pointer(filled_dem, d8_pointer)
        wbt.d8_flow_accumulation(filled_dem, flow_accum)
    


    streams_raster = os.path.join(output_dir, f"streams_{int(threshold/1000)}k.tif")
    streams_vector = streams_raster.replace(".tif", ".shp")
    streams_gpkg = streams_raster.replace(".tif", ".gpkg")
    if not os.path.exists(streams_raster):
        wbt.extract_streams(flow_accum, streams_raster, threshold)

    # Convert raster streams to vector
    if not os.path.exists(streams_vector):
        wbt.raster_streams_to_vector(streams_raster, d8_pointer, streams_vector)

        # Assign CRS from DEM to the GeoDataFrame and save as GeoPackage
    gdf = gpd.read_file(streams_vector)

    # Load the DEM to get its CRS
    with rasterio.open(dem) as src:
        dem_crs = src.crs

    if dem_crs is None:
        raise ValueError("DEM does not have a CRS. Please provide a DEM with a valid CRS.")
    else:
        print(f"DEM CRS: {dem_crs}")

    # Check if the GeoDataFrame has a CRS
    if gdf.crs is None:
        print("GeoDataFrame does not have a CRS. Assigning CRS from DEM.")
        gdf.set_crs(dem_crs, inplace=True)
    else:
        # Reproject the GeoDataFrame to match the DEM CRS
        gdf = gdf.to_crs(dem_crs)

    # Save the reprojected GeoDataFrame to a GeoPackage
    gdf.to_file(streams_gpkg, driver="GPKG")

    add_max_flow_accum_to_stream(streams_gpkg, flow_accum)
    
    return streams_gpkg, streams_raster, filled_dem, d8_pointer, flow_accum

def add_max_flow_accum_to_stream(streams_gpkg, flow_accum_raster):

    from rasterstats import zonal_stats

    # Load the streams layer
    streams_gdf = gpd.read_file(streams_gpkg)

    # Buffer each stream by 1 meter
    streams_gdf['buffered_geometry'] = streams_gdf.geometry.buffer(1)

    # Calculate the maximum flow accumulation value for each buffered stream
    with rasterio.open(flow_accum_raster) as src:
        stats = zonal_stats(streams_gdf['buffered_geometry'], flow_accum_raster, stats="max", nodata=src.nodata)
        
    # Add the maximum flow accumulation value to the GeoDataFrame
    streams_gdf['flow_accum_max'] = [stat['max'] for stat in stats]

    # Save the updated GeoDataFrame back to the GeoPackage
    streams_gdf = streams_gdf.drop(columns='buffered_geometry')  # Drop the buffered geometry column if not needed
    streams_gdf.to_file(streams_gpkg, driver="GPKG")


if __name__ == "__main__":
    dem = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20250006_Wallowa R Remeander (AP)\07_GIS\Data\AP_WallowaSunriseDEM\WallowaSunriseDEM\GRMW_unclipped_1ft_DEM.tif"
    output_dir = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20250006_Wallowa R Remeander (AP)\07_GIS\Data\REM\Streams"
    threshold = 100000
    get_streams(dem, output_dir, threshold, overwrite=True, breach_depressions=True)

