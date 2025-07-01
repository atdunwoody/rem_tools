import rasterio

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

import rasterio
from rasterio.mask import mask
import geopandas as gpd

def clip_tif_by_gpkg(
    tif_path: str,
    gpkg_path: str,
    output_tif_path: str,
    layer: str = None
) -> None:
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


if __name__ == "__main__":
    input_tif = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM\PRISM_ppt_30yr_normal_4kmM4_annual.tif"
    mask_gpkg = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\Basin Mask.gpkg"
    output_tif = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\PRISM_annual_clipped.tif"

    clip_tif_by_gpkg(input_tif, mask_gpkg, output_tif)
    print(f"Clipped raster written to {output_tif}")


