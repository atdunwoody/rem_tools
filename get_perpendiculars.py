import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge
import numpy as np
import os
from .smooth_stream import interpolate_geopackage

def create_smooth_perpendicular_lines(centerline_path, line_length=60, spacing=5, window=20, output_path=None):

    # Load the centerline from the geopackage
    gdf = gpd.read_file(centerline_path)
    
    try:
        # Dissolve and merge the centerlines into one continuous geometry
        merged_geometry = linemerge(gdf.geometry.unary_union)

    except Exception as e:
        print(f"Error merging centerlines: {e}")
        merged_geometry = gdf.geometry.unary_union
    # Handle MultiLineString appropriately using `geoms`
    if isinstance(merged_geometry, MultiLineString):
        line_parts = merged_geometry.geoms
    else:
        line_parts = [merged_geometry]

    # Initialize an empty list to store perpendicular lines
    perpendiculars = []
    
    # Process each line part
    for line in line_parts:
        length = line.length
        num_samples = int(np.floor(length / spacing))
        for i in range(num_samples + 1):
            # Calculate the point at each interval along the centerline
            point = line.interpolate(i * spacing)
            
            # Get points window meters ahead and behind
            point_back = line.interpolate(max(0, i * spacing - window))
            point_forward = line.interpolate(min(length, i * spacing + window))
            
            # Calculate vectors to these points
            dx_back, dy_back = point.x - point_back.x, point.y - point_back.y
            dx_forward, dy_forward = point_forward.x - point.x, point_forward.y - point.y
            
            # Average the vectors
            dx_avg = (dx_back + dx_forward) / 2
            dy_avg = (dy_back + dy_forward) / 2
            
            # Calculate the perpendicular vector
            len_vector = np.sqrt(dx_avg**2 + dy_avg**2)
            perp_vector = (-dy_avg, dx_avg)
            
            # Normalize and scale the perpendicular vector
            perp_vector = (perp_vector[0] / len_vector * line_length, perp_vector[1] / len_vector * line_length)
            
            # Create the perpendicular line segment
            perp_line = LineString([
                (point.x + perp_vector[0], point.y + perp_vector[1]),
                (point.x - perp_vector[0], point.y - perp_vector[1])
            ])
            
            # Append the perpendicular line to the list
            perpendiculars.append({'geometry': perp_line})
    
    # Convert list to GeoDataFrame
    perpendiculars_gdf = gpd.GeoDataFrame(perpendiculars, crs=gdf.crs)
    
    # Save the perpendicular lines to the output geopackage
    if output_path is not None:
        perpendiculars_gdf.to_file(output_path, driver='GPKG')
    return output_path

def get_perpendiculars_from_stream_CL(input_valley_centerline, output_directory = None, line_length=60, spacing=5, window=100):
    """
    This function takes a valley centerline and creates perpendicular lines at regular intervals along the centerline.
    The perpendicular lines are saved to a GeoPackage file.
    
    input_valley_centerline: str, path to the valley centerline GeoPackage file
    output_directory: str, path to the output directory where the perpendiculars will be saved
    """
    
    if output_directory is None:
        output_directory = os.path.dirname(input_valley_centerline)
    watershed = os.path.basename(input_valley_centerline).split("_")[0]


    output_smoothed_valley = os.path.join(os.path.dirname(input_valley_centerline), f"{watershed}_CL_smoothed.gpkg")
    output_perpendiculars = os.path.join(output_directory, f"{watershed}_smooth_perpendiculars_{spacing}m.gpkg")

    # Call the function
    interpolate_geopackage(
        input_gpkg=input_valley_centerline,
        output_gpkg=output_smoothed_valley,
        interval=window
    )

    perp_lines = create_smooth_perpendicular_lines(output_smoothed_valley, line_length = line_length, 
                                                spacing=spacing, window=window, 
                                                output_path=output_perpendiculars)
    perp_lines.to_file(output_perpendiculars, driver='GPKG')
    
    return output_perpendiculars

def clip_lines_by_poly(input_lines, input_polys, output_gpkg):
    """
    Clips an input set of lines by an input set of polygons in a GeoPackage.

    Parameters:
    input_gpkg (str): Path to the input GeoPackage containing lines and polygons layers.
    lines_layer (str): Name of the layer containing line geometries.
    polygons_layer (str): Name of the layer containing polygon geometries.
    output_gpkg (str): Path to the output GeoPackage where the clipped lines will be stored.
    output_layer_name (str): Name of the layer to store clipped lines (default: 'clipped_lines').

    Returns:
    None
    """
    # Check if output_dir is a directory
    if os.path.isfile(output_gpkg):
        print(f"Output file already exists: {output_gpkg}")
        return output_gpkg
    # Load the line and polygon layers from the input GeoPackage
    lines_gdf = gpd.read_file(input_lines)
    polygons_gdf = gpd.read_file(input_polys)

    # get lines_gdf crs
    crs = lines_gdf.crs
    # set polygons_gdf crs to lines_gdf crs
    polygons_gdf = polygons_gdf.to_crs(crs)
    
    #Buffer the polygons by 5 meters to ensure that the lines are clipped properly
    polygons_gdf['geometry'] = polygons_gdf.buffer(5)
    
    # Perform the clip operation
    clipped_lines_gdf = gpd.overlay(lines_gdf, polygons_gdf, how='intersection')

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_gpkg)
    os.makedirs(output_dir, exist_ok=True)
    # Save the clipped lines to the output GeoPackage
    clipped_lines_gdf.to_file(output_gpkg, driver='GPKG')
    print(f"Clipped lines have been saved to: {output_gpkg}")
    return output_gpkg

def main():

    """
    Uncomment below to get perpendiculars for all centerlines in a directory
    """
    centerline_path = r"Y:\ATD\GIS\ETF\Valley Geometry\Centerlines\Manual Centerlines\MM Manual Centerline.gpkg"
    output_dir = r"Y:\ATD\GIS\ETF\Valley Geometry\Perpendiculars"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    perp_lines = create_smooth_perpendicular_lines(centerline_path, line_length=100, spacing=20, window=10)
    output_path = os.path.join(output_dir, 'MM_perpendiculars_20m.gpkg')
    perp_lines.to_file(output_path, driver='GPKG')

    """
    Uncomment below to get perpendiculars for a single centerline
    """
    # centerline_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\ME_centerline.gpkg"
    # output_path = r"Y:\ATD\GIS\Bennett\Valley Widths\Valley Centerlines\ME_perps_DISSOLVED.gpkg"
    # perp_lines = create_smooth_perpendicular_lines(centerline_path, line_length=100, spacing=100, window=100, output_path=output_path)
    # perp_lines.to_file(output_path, driver='GPKG')
    
if __name__ == '__main__':
    main()
