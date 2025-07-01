# dem_processor.py

from whitebox_workflows import WbEnvironment
import whitebox
import os
import geopandas as gpd
import rasterio
from shapely.geometry import LineString, MultiLineString
from typing import Optional, Tuple

def get_streams(
    dem: str,
    output_dir: str,
    threshold: int = 100000,
    overwrite: bool = False,
    breach_depressions: bool = True,
    thin_n: int = 10,
    create_thinned: bool = True
) -> Tuple[str, str, str, str, str, Optional[str]]:
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

    add_max_flow_accum_to_stream(streams_gpkg, flow_accum)
    
    return streams_gpkg, streams_raster, filled_dem, d8_pointer, flow_accum, thinned_gpkg


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


def thin_centerline(
    input_gpkg: str,
    layer_name: str,
    output_gpkg: str,
    output_layer: Optional[str] = None,
    n: int = 10
) -> None:
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
    dem = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\LiDAR\rasters_USGS10m\USGS 10m DEM Clip.tif"
    output_dir = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Projects\20240007_Atlas Process (GRMW)\07_GIS\Data\LiDAR\Streams"
    thresholds = [1000, 10000, 100000]
    for threshold in thresholds:
        sub_out_dir = os.path.join(output_dir, f"Streams_{threshold/1000}k")
        get_streams(dem, sub_out_dir, threshold=threshold, overwrite=False, breach_depressions=True, thin_n=10)

