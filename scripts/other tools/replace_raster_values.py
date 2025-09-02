import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

def replace_raster(base_raster_path: str,
                   replacement_raster_path: str,
                   output_raster_path: str
                   ) -> None:
    """
    Produce an output raster by substituting base raster values
    with replacement raster values wherever valid.

    Parameters
    ----------
    base_raster_path : str
        Path to the base raster (e.g. GeoTIFF).
    replacement_raster_path : str
        Path to the replacement raster.
    output_raster_path : str
        Path where the output raster will be written.
    """
    # Open base and replacement
    with rasterio.open(base_raster_path) as base_src, \
         rasterio.open(replacement_raster_path) as rep_src:

        # Read base data
        base_data = base_src.read(1)
        profile = base_src.profile

        # Prepare an array to receive the reprojected replacement
        rep_reproj = np.full(base_data.shape, rep_src.nodata, dtype=base_src.dtypes[0])

        # Reproject replacement into base’s grid
        reproject(
            source=rasterio.band(rep_src, 1),
            destination=rep_reproj,
            src_transform=rep_src.transform,
            src_crs=rep_src.crs,
            dst_transform=base_src.transform,
            dst_crs=base_src.crs,
            resampling=Resampling.bilinear
        )

        # Build mask of valid replacement pixels
        if rep_src.nodata is not None:
            valid_mask = rep_reproj != rep_src.nodata
        else:
            valid_mask = ~np.isnan(rep_reproj)

        # Substitute values
        out_data = base_data.copy()
        out_data[valid_mask] = rep_reproj[valid_mask]

        # Write output
        with rasterio.open(output_raster_path, 'w', **profile) as dst:
            dst.write(out_data, 1)


base_raster_path = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\REM\REM_bathy-WSE interpolated_radius_corrected.tif"
replacement_raster_path = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\REM\Voronoi Method\20250725\REM_bathy-_WSE_rand_points_valid.tif"
output_raster_path = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\REM\Voronoi Method\combined corrected REM\min_points_interpolated_radius_WSE_corrected_v2.tif"
replace_raster(base_raster_path,
                replacement_raster_path,
                output_raster_path
               )