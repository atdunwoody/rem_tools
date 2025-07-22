
import geopandas as gpd
from rasterstats import zonal_stats


def add_BF_to_streams_Legg(streams_gpkg_path: str, precip_raster_path: str = None) -> None:
    """
    Reads stream features from a GeoPackage, computes mean annual precipitation (cm)
    for each feature based on intersecting PRISM raster data, adds a new precip field,
    then computes bankfull width (m) and depth (m) using Legg & Olson 2015.

    Parameters
    ----------
    streams_gpkg_path : str
        Path to the input streams GeoPackage.
    precip_raster_path : str
        Path to the input annual precipitation raster (PRISM .tif in mm).
    output_gpkg_path : str
        Path where the output GeoPackage (with new fields) will be written.
    layer : str, optional
        Name of the layer to read from the input GeoPackage. If None, the default
        (first) layer will be used.
    """
    # constants
    KM2_TO_MI2 = 0.386102          # km² → mi²
    CM_TO_IN = 1.0 / 2.54          # cm → in
    FT_TO_M = 0.3048               # ft → m

    # 1. Read streams layer
    streams = gpd.read_file(streams_gpkg_path)

    # after reading:
    streams = streams[streams.geometry.notnull()]
    streams = streams[streams.is_valid]

    
    # # 2. Compute per-feature mean precipitation (mm)
    # stats = zonal_stats(
    #     streams,
    #     precip_raster_path,
    #     stats=['mean'],
    #     geojson_out=False,
    #     all_touched=True
    # )
    # mean_vals_mm = [s['mean'] for s in stats]

    # Uncomment if you want to develop watershed based appraoch
    # 3. Convert from mm to cm and add field
    # streams['ann_precip_cm'] = [
    #     mv / 10.0 if mv is not None else None
    #     for mv in mean_vals_mm
    # ]

    # 4. Compute drainage area in mi² and precip in inches
    streams['DA_mi2'] = streams['DA_km2'] * KM2_TO_MI2
    streams['ann_precip_in'] = 72.17 * CM_TO_IN # 72.17 is the average annual precipitation in cm for the GRMW basin
    streams['ann_precip_cm'] = 72.17 # 72.17 is the average annual precipitation in cm for the GRMW basin

    # 5. Bankfull width (m) based on Legg & Olson 2015:
    #    width_ft = 1.16 * 0.91 * (DA_mi2^0.381) * (precip_in^0.634)
    #    convert ft → m
    streams['BF_width_Legg_m'] = (
        FT_TO_M *
        1.16 * 0.91 *
        (streams['DA_mi2'] ** 0.381) *
        (streams['ann_precip_in'] ** 0.634)
    )

    # 6. Bankfull depth (m) based on Legg & Olson 2015:
    #    depth = 0.0939 * (DA_km2^0.233) * (precip_cm^0.264)
    streams['BF_depth_Legg_m'] = (
        0.0939 *
        (streams['DA_km2'] ** 0.233) *
        (streams['ann_precip_cm'] ** 0.264)
    )

    # 7. Write to new GeoPackage (overwrites if exists)
    streams.to_file(streams_gpkg_path, driver='GPKG')
    return streams_gpkg_path

def add_BF_to_streams_Jackson(streams_gpkg_path: str) -> None:
    """
    Reads stream features from a GeoPackage, computes bankfull width (m) and depth (m)
    using Jackson 2016.

    Parameters
    ----------
    streams_gpkg_path : str
        Path to the input streams GeoPackage.
    """
    # 1. Read streams layer
    streams = gpd.read_file(streams_gpkg_path)
    km2_to_mi2 = 0.386102  # km² to mi²
    ft_to_m = 0.3048  # ft to m
    # 2. Compute bankfull width (m) based on Jackson 2016:
    #    width_m = ft_to_m * 9.40 * (DA * m2_to_mi2) ** 0.42
    streams['BF_width_Jackson_m'] = ft_to_m * 9.40 * ((streams['DA_km2'] * km2_to_mi2) ** 0.42)

    # 3. Compute bankfull depth (m) based on Jackson 2016:
    #    depth_ft = 0.61 * DA_mi^2 **0.33
    streams['BF_depth_Jackson_m'] = ft_to_m * 0.61 * ((streams['DA_km2'] * km2_to_mi2) ** 0.33)

    # 4. Write to new GeoPackage (overwrites if exists)
    streams.to_file(streams_gpkg_path, driver='GPKG')
    return streams_gpkg_path


streams_gpkg = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\REM\Bankfull Regression\streams_100k_clipped_to_LiDAR.gpkg"
add_BF_to_streams_Legg(streams_gpkg, precip_raster_path=None)
add_BF_to_streams_Jackson(streams_gpkg)