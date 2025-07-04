# dem_processor.py

from whitebox_workflows import WbEnvironment
import whitebox
import os
import geopandas as gpd
import rasterio
from shapely.geometry import LineString, MultiLineString
from typing import Optional, Tuple
from rasterstats import zonal_stats

def get_streams(dem: str, output_dir: str, threshold: int = 100000, overwrite: bool = False,
                breach_depressions: bool = True, thin_n: int = 10,
                create_thinned: bool = True, precip_raster: Optional[str] = None):
    """
    Processes a DEM to extract streams, save them to a GeoPackage, and—
    optionally—create a thinned centerline layer.

    Returns:
      streams_gpkg, streams_raster, filled_dem, d8_pointer,
      flow_accum, thinned_gpkg (or None if create_thinned=False)
    """
    # --- Whitebox setup ------------------------------------------------------
    wbt = whitebox.WhiteboxTools()
    wbe = WbEnvironment()

    # --- Prepare output directory -------------------------------------------
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    elif overwrite:
        for fn in os.listdir(output_dir):
            fp = os.path.join(output_dir, fn)
            if os.path.isfile(fp) or os.path.islink(fp):
                os.unlink(fp)
            else:
                import shutil
                shutil.rmtree(fp)

    # --- Intermediate filenames ---------------------------------------------
    filled_dem    = os.path.join(output_dir, "filled_dem.tif")
    d8_pointer    = os.path.join(output_dir, "d8_pointer.tif")
    flow_accum    = os.path.join(output_dir, "flow_accum.tif")
    breached_dem  = os.path.join(output_dir, "breached_dem.tif")

    # --- Breach / fill -------------------------------------------------------
    if breach_depressions and not os.path.exists(breached_dem):
        wbt.breach_depressions_least_cost(dem, breached_dem, 10)
    if not breach_depressions and not os.path.exists(filled_dem):
        wbt.fill_depressions(dem, filled_dem)

    # --- Flow direction & accumulation --------------------------------------
    src_dem = breached_dem if breach_depressions else filled_dem
    if not os.path.exists(d8_pointer) or not os.path.exists(flow_accum):
        wbt.d8_pointer(src_dem, d8_pointer)
        wbt.d8_flow_accumulation(src_dem, flow_accum)

    # --- Extract streams -----------------------------------------------------
    streams_raster = os.path.join(output_dir, f"streams_{int(threshold/1000)}k.tif")
    if not os.path.exists(streams_raster):
        wbt.extract_streams(flow_accum, streams_raster, threshold)

    # --- Raster → Shapefile → GeoPackage ------------------------------------
    streams_shp   = streams_raster.replace(".tif", ".shp")
    streams_gpkg  = streams_raster.replace(".tif", ".gpkg")
    streams_layer = os.path.splitext(os.path.basename(streams_gpkg))[0]

    if not os.path.exists(streams_shp):
        wbt.raster_streams_to_vector(streams_raster, d8_pointer, streams_shp)

    # read vector, ensure proper CRS
    gdf = gpd.read_file(streams_shp)
    with rasterio.open(dem) as src:
        dem_crs = src.crs
    if dem_crs is None:
        raise ValueError("DEM has no CRS.")
    if gdf.crs is None:
        gdf.set_crs(dem_crs, inplace=True)
    else:
        gdf = gdf.to_crs(dem_crs)

    # save to GeoPackage
    gdf.to_file(streams_gpkg, layer=streams_layer, driver="GPKG")

    # --- Thin centerline if requested ---------------------------------------
    thinned_gpkg = None
    if create_thinned:
        thinned_gpkg = streams_gpkg.replace(".gpkg", "_thinned.gpkg")
        thin_centerline(
            input_gpkg=streams_gpkg,
            layer_name=streams_layer,
            output_gpkg=thinned_gpkg,
            output_layer=f"{streams_layer}_thinned",
            n=thin_n
        )

    add_DA_to_stream(streams_gpkg, flow_accum)
    add_BF_dims_to_stream(streams_gpkg, precip_raster)
    return streams_gpkg

def add_DA_to_stream(streams_gpkg, flow_accum_raster, unit: str = "m") -> None:
    """
    Adds a column to the streams GeoPackage with the drainage area (DA) in km²
    by calculating the maximum flow accumulation value within a 1-meter buffer
    around each stream.
    Args:
        streams_gpkg (str): Path to the GeoPackage containing the streams layer.
        flow_accum_raster (str): Path to the flow accumulation raster.
        unit (str): Unit of the flow accumulation raster. Either "m" for meters or "ft" for feet.
    """
    from rasterstats import zonal_stats

    # Load the streams layer
    streams_gdf = gpd.read_file(streams_gpkg)

    # Buffer each stream by 1 meter
    streams_gdf['buffered_geometry'] = streams_gdf.geometry.buffer(1)

    # Calculate the maximum flow accumulation value for each buffered stream
    with rasterio.open(flow_accum_raster) as src:
        stats = zonal_stats(streams_gdf['buffered_geometry'], flow_accum_raster, stats="max", nodata=src.nodata)

    if unit.lower() == "ft":
        # Convert from square feet to square kilometers
        # 1 square foot = 0.092903 square meters, and 1 square kilometer = 1,000,000 square meters
        conversion_factor = (0.092903 / 1e6)
    else:
        # Assume unit is meters, so conversion factor is 1 (1 m² = 1 m²)
        conversion_factor = 1e6
    # Add the maximum flow accumulation value to the GeoDataFrame
    streams_gdf['DA_km2'] = [stat['max'] for stat in stats] / conversion_factor  # Convert from m² to km²

    # Save the updated GeoDataFrame back to the GeoPackage
    streams_gdf = streams_gdf.drop(columns='buffered_geometry')  # Drop the buffered geometry column if not needed
    streams_gdf.to_file(streams_gpkg, driver="GPKG")

def add_BF_dims_to_stream(streams_gpkg_path: str, precip_raster_path: str) -> None:
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

    
    # 2. Compute per-feature mean precipitation (mm)
    stats = zonal_stats(
        streams,
        precip_raster_path,
        stats=['mean'],
        geojson_out=False,
        all_touched=True
    )
    mean_vals_mm = [s['mean'] for s in stats]

    # 3. Convert from mm to cm and add field
    streams['ann_precip_cm'] = [
        mv / 10.0 if mv is not None else None
        for mv in mean_vals_mm
    ]

    # 4. Compute drainage area in mi² and precip in inches
    streams['DA_mi2'] = streams['DA_km2'] * KM2_TO_MI2
    streams['ann_precip_in'] = streams['ann_precip_cm'] * CM_TO_IN

    # 5. Bankfull width (m) based on Legg & Olson 2015:
    #    width_ft = 1.16 * 0.91 * (DA_mi2^0.381) * (precip_in^0.634)
    #    convert ft → m
    streams['bank_width_m'] = (
        FT_TO_M *
        1.16 * 0.91 *
        (streams['DA_mi2'] ** 0.381) *
        (streams['ann_precip_in'] ** 0.634)
    )

    # 6. Bankfull depth (m) based on Legg & Olson 2015:
    #    depth = 0.0939 * (DA_km2^0.233) * (precip_cm^0.264)
    streams['bank_depth_m'] = (
        0.0939 *
        (streams['DA_km2'] ** 0.233) *
        (streams['ann_precip_cm'] ** 0.264)
    )

    # 7. Write to new GeoPackage (overwrites if exists)
    streams.to_file(streams_gpkg_path, driver='GPKG')
    return streams_gpkg_path

def thin_centerline( input_gpkg: str, layer_name: str, output_gpkg: str,
    output_layer: Optional[str] = None, n: int = 10) -> None:
    """
    Keeps every nᵗʰ vertex in each LineString (or part of a MultiLineString)
    from the specified layer in input_gpkg, writing to output_gpkg.
    """
    gdf = gpd.read_file(input_gpkg, layer=layer_name)

    def _thin(geom):
        if geom is None or geom.is_empty:
            return geom
        if isinstance(geom, LineString):
            coords = list(geom.coords)
            pts = coords[::n]
            if coords[-1] not in pts:
                pts.append(coords[-1])
            return LineString(pts)
        if isinstance(geom, MultiLineString):
            parts = []
            for part in geom.geoms:
                coords = list(part.coords)
                pts = coords[::n]
                if coords[-1] not in pts:
                    pts.append(coords[-1])
                parts.append(LineString(pts))
            return MultiLineString(parts)
        return geom

    gdf["geometry"] = gdf.geometry.apply(_thin)
    out_layer = output_layer or layer_name
    gdf.to_file(output_gpkg, layer=out_layer, driver="GPKG")

if __name__ == "__main__":
    
   dem = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\LiDAR\grmw_rasters\bathymetry.tif"
   output_dir = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\Streams_Bathymetry"
   threshold = 100000  # 100k m²
   get_streams(
       dem=dem,
       output_dir=output_dir,
       threshold=threshold,
       overwrite=True,
       breach_depressions=True,
       thin_n=10,
       create_thinned=False,
       precip_raster=None  # Optional, can be set to a PRISM raster path
   )
