import rasterio
from rasterio.mask import mask
import geopandas as gpd
import fiona
import os
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

###########################################################
# CONVERT PRISM BIL TO GEOTIFF AND CLIP BY AREA OF INTEREST
###########################################################
def convert_bil_to_geotiff(bil_path: str, geotiff_path: str) -> None:
    """
    Convert an ESRI BIL (.bil + .hdr/.prj) into a GeoTIFF.

    Parameters
    ----------
    bil_path : str
        Path to the input .bil file.
        Ensure the .hdr, .prj (and any .aux.xml) are alongside the .bil.
    geotiff_path : str
        Path for the output GeoTIFF file. Should end with .tif or .tiff.
    """
    # Open the BIL dataset
    with rasterio.open(bil_path) as src:
        profile = src.profile.copy()
        # Update driver and, if desired, compression
        profile.update(driver='GTiff', compress='lzw')

        # Write out as GeoTIFF
        with rasterio.open(geotiff_path, 'w', **profile) as dst:
            dst.write(src.read())

def clip_tif_by_gpkg(tif_path: str,gpkg_path: str,output_tif_path: str,layer: str = None) -> None:
    """
    Clip a GeoTIFF by the polygon(s) in a GeoPackage, reprojecting
    the mask geometries to the raster CRS if needed.

    Parameters
    ----------
    tif_path : str
        Path to the input GeoTIFF file.
    gpkg_path : str
        Path to the input GeoPackage containing the mask polygon(s).
    output_tif_path : str
        Path for the clipped output GeoTIFF.
    layer : str, optional
        Name of the layer within the GeoPackage. If None, the first layer is used.
    """
    # Read mask polygons
    if layer:
        shapes = gpd.read_file(gpkg_path, layer=layer)
    else:
        shapes = gpd.read_file(gpkg_path)

    # Open source raster
    with rasterio.open(tif_path) as src:
        src_crs = src.crs

        # Reproject mask to raster CRS if they differ
        if shapes.crs != src_crs:
            shapes = shapes.to_crs(src_crs)

        geoms = shapes.geometry.values

        # Perform the clipping
        out_image, out_transform = mask(src, geoms, crop=True)
        out_meta = src.meta.copy()

    # Update metadata for the clipped raster
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    # Write clipped raster
    with rasterio.open(output_tif_path, "w", **out_meta) as dst:
        dst.write(out_image)

##############################################################
##############################################################

def rename_field(input_gpkg_path: str, output_gpkg_path: str, layer_name: str = None):
    """
    Rename the 'flow_accum_max' field to 'DA_km2' in all (or a specific) layer of a GeoPackage.

    Parameters
    ----------
    input_gpkg_path : str
        Path to the input GeoPackage.
    output_gpkg_path : str
        Path where the output GeoPackage with renamed field will be written.
    layer_name : str, optional
        Name of the layer to process. If None, all layers will be processed.
    """
    # Determine which layers to process
    layers = [layer_name] if layer_name else fiona.listlayers(input_gpkg_path)

    first_write = True
    for lyr in layers:
        # Read the layer into a GeoDataFrame
        gdf = gpd.read_file(input_gpkg_path, layer=lyr)

        # Rename the field if it exists
        if 'flow_accum_max' in gdf.columns:
            gdf = gdf.rename(columns={'flow_accum_max': 'DA_km2'})
        else:
            print(f"Layer '{lyr}' does not contain a 'flow_accum_max' field; skipping rename for this layer.")

        # Write (or append) to the output GeoPackage
        if first_write:
            # Overwrite or create new file
            gdf.to_file(output_gpkg_path, layer=lyr, driver='GPKG')
            first_write = False
        else:
            # Append additional layers
            gdf.to_file(output_gpkg_path, layer=lyr, driver='GPKG', mode='a')

def print_raster_min_max(folder_path: str, extensions=None):
    """
    For every raster in folder_path (and subfolders), prints:
      <filename> (band <i>): min <min_val>, max <max_val>
    
    Parameters
    ----------
    folder_path : str
        Path to the folder containing rasters.
    extensions : list of str, optional
        Raster file extensions to include (e.g. ['.tif', '.img']). 
        Defaults to common ones.
    """
    if extensions is None:
        extensions = ['.tif', '.img', '.bil', '.vrt']
    
    base = Path(folder_path)
    for fp in base.rglob('*'):
        if fp.suffix.lower() in extensions:
            with rasterio.open(fp) as src:
                for b in src.indexes:
                    arr = src.read(b, masked=True)
                    print(f"{fp.name} (band {b}): min {float(arr.min())}, max {float(arr.max())}")


def cluster_points(
    input_gpkg_path: str,
    output_gpkg_path: str,
    n_clusters: int = 11,
    new_field: str = "cluster_id",):
    """
    Reads points from `input_gpkg_path`/`layer`, clusters them into `n_clusters` groups
    based on XY proximity (KMeans), writes to `output_gpkg_path` with a new integer field.

    Parameters
    ----------
    input_gpkg_path : str
        Path to source GeoPackage.
    layer : str
        Layer name within the GeoPackage.
    output_gpkg_path : str
        Path to output GeoPackage (will be created or overwritten).
    n_clusters : int
        Number of clusters to create.
    new_field : str
        Name of the new integer field to hold cluster IDs (1..n_clusters).
    """
    # 1. Read
    gdf = gpd.read_file(input_gpkg_path)

    # 2. Reproject (if geographic, to a suitable projected CRS)
    if gdf.crs.is_geographic:
        # choose an appropriate local projection; here we use UTM zone of the centroid
        centroid = gdf.unary_union.centroid
        utm_crs = f"+proj=utm +zone={int((centroid.x + 180)//6)+1} +datum=WGS84 +units=m +no_defs"
        gdf = gdf.to_crs(utm_crs)

    # 3. Extract coordinates
    coords = np.vstack([gdf.geometry.x, gdf.geometry.y]).T

    # 4. Cluster
    model = KMeans(n_clusters=n_clusters, random_state=0)
    labels = model.fit_predict(coords)

    # 5. Assign labels 1..n_clusters
    gdf[new_field] = labels + 1

    # 6. Write out
    gdf.to_file(output_gpkg_path, driver="GPKG")


if __name__ == "__main__":
    input_gpkg = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM\Streams\Streams_1.0k\streams_1k.gpkg"
    output_gpkg = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM\test_REM\streams_100k_clip.gpkg"

    cluster_points(
        input_gpkg_path = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM\min_elev_points_100m.gpkg",
        output_gpkg_path=r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM\min_elev_points_100m_clustered.gpkg",
        n_clusters=11
    )