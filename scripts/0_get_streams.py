# dem_processor.py

from whitebox_workflows import WbEnvironment
import whitebox
import os
import geopandas as gpd
import rasterio
from shapely.geometry import LineString, MultiLineString
from typing import Optional, Tuple
from rasterstats import zonal_stats
import geopandas as gpd
import fiona


def get_streams(dem: str, output_dir: str, threshold: int = 100000, overwrite: bool = False,
                breach_depressions: bool = True, thin_n: int = 10,
                create_thinned: bool = True, precip_raster: Optional[str] = None
):
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

    # --- Intermediate filenames ---------------------------------------------
    filled_dem    = os.path.join(output_dir, "filled_dem.tif")
    d8_pointer    = os.path.join(output_dir, "d8_pointer.tif")
    flow_accum    = os.path.join(output_dir, "flow_accum.tif")
    breached_dem  = os.path.join(output_dir, "breached_dem.tif")
    if overwrite:
        os.remove(filled_dem) if os.path.exists(filled_dem) else None
        os.remove(d8_pointer) if os.path.exists(d8_pointer) else None
        os.remove(flow_accum) if os.path.exists(flow_accum) else None
        os.remove(breached_dem) if os.path.exists(breached_dem) else None
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
        print(f"Thinning centerline by keeping every {thin_n}ᵗʰ vertex...")
        thinned_gpkg = streams_gpkg.replace(".gpkg", "_thinned.gpkg")
        thin_centerline(
            input_gpkg=streams_gpkg,
            layer_name=streams_layer,
            output_gpkg=thinned_gpkg,
            output_layer=f"{streams_layer}_thinned",
            n=thin_n
        )

    print("Adding drainage area and bankfull dimensions...")
    add_DA_to_stream(streams_gpkg, flow_accum)
    add_BF_to_streams_Legg(streams_gpkg, precip_raster)
    add_BF_to_streams_Castro(streams_gpkg)
    add_BF_to_streams_Beechie(streams_gpkg)
    
    print(f"[✔] Streams extracted to: {streams_gpkg}")
    return streams_gpkg


def add_DA_to_stream(
    streams_gpkg: str,
    flow_accum_raster: str,
    unit: str = "m") -> None:
    """
    Adds a 'DA_km2' field to the streams layer in `streams_gpkg`,
    by taking the maximum flow-accumulation value within a 1 m buffer
    and converting to km².
    Assumes the flow-accumulation raster has a cell size of 10 m.
    """
    # 1. load
    streams = gpd.read_file(streams_gpkg)

    # 2. buffer
    streams['buffer'] = streams.geometry.buffer(0.1)

    # 3. zonal stats
    with rasterio.open(flow_accum_raster) as src:
        nodata = src.nodata
        cell_size = src.res[0]  # assuming square cells

    stats = zonal_stats(
        streams['buffer'],
        flow_accum_raster,
        # pick the 99th percentile to avoid outliers
        stats=['median'],
        nodata=nodata
    )

    # 4. compute conversion factor: km²-per-cell-unit
    if unit.lower() == "ft":
        # 10 meter cell size, so multiply by 100 to get m²
        # 1 ft² → 0.092903 m², then → km² by ×1e-6
        conv = 0.092903 * 1e-6 * cell_size**2
    else:
        # assume metres: 1 m² → 1e-6 km²
        conv = 1e-6 * cell_size**2

    # 5. element‑wise multiply
    streams['DA_km2'] = [
        (s['median'] or 0) * conv
        for s in stats
    ]

    # 6. cleanup & save
    streams = streams.drop(columns='buffer')
    streams.to_file(streams_gpkg, driver="GPKG")

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

    ###########################################################
    # Uncomment if you want to develop watershed based approach
    ###########################################################
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

def add_BF_to_streams_Castro(streams_gpkg_path: str) -> None:
    """
    Reads stream features from a GeoPackage, computes bankfull width (m) and depth (m)
    using Castro & Jackson 2001.

    Parameters
    ----------
    streams_gpkg_path : str
        Path to the input streams GeoPackage.
    """
    # 1. Read streams layer
    streams = gpd.read_file(streams_gpkg_path)
    km2_to_mi2 = 0.386102  # km² to mi²
    ft_to_m = 0.3048  # ft to m
    # 2. Compute bankfull width (m) based on Castro & Jackson 2001:
    #    width_m = ft_to_m * 9.40 * (DA * m2_to_mi2) ** 0.42
    streams['BF_width_Castro_m'] = ft_to_m * 9.40 * ((streams['DA_km2'] * km2_to_mi2) ** 0.42)

    # 3. Compute bankfull depth (m) based on Castro & Jackson 2001:
    #    depth_ft = 0.61 * DA_mi^2 **0.33
    streams['BF_depth_Castro_m'] = ft_to_m * 0.61 * ((streams['DA_km2'] * km2_to_mi2) ** 0.33)

    # 4. Write to new GeoPackage (overwrites if exists)
    streams.to_file(streams_gpkg_path, driver='GPKG')
    return streams_gpkg_path

def add_BF_to_streams_Beechie(streams_gpkg_path: str) -> None:
    """
    Reads stream features from a GeoPackage, computes bankfull width (m) and depth (m)
    using Beechie and IMAKI 2013.

    Parameters
    ----------
    streams_gpkg_path : str
        Path to the input streams GeoPackage.
    """
    # 1. Read streams layer
    streams = gpd.read_file(streams_gpkg_path)
    # 2. Compute bankfull width (m) based on Beechie and IMAKI 2013:
    P_cm_yr = 72.17  # average annual precipitation in cm for the GRMW basin based on PRISM
    streams['BF_width_Beechie_m'] = 0.177 * ((streams['DA_km2']) ** 0.397) * P_cm_yr ** 0.453
    streams['BF_depth_Beechie_m'] = streams['BF_width_Beechie_m'] * streams['BF_depth_Castro_m'] / streams['BF_width_Castro_m']

    # 4. Write to new GeoPackage (overwrites if exists)
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

def threshold_lines_by_length(
    input_gpkg: str,
    output_gpkg: str,
    threshold: float = 1200.0
) -> None:
    """
    Read the first layer of `input_gpkg`, keep only LineString/MultiLineString
    features longer than `threshold` (in CRS units, typically metres), and
    write them to `output_gpkg` in the same layer name.
    """
    # get first layer name
    layers = fiona.listlayers(input_gpkg)
    if not layers:
        raise ValueError(f"No layers found in {input_gpkg!r}")
    layer = layers[0]

    # load features
    gdf = gpd.read_file(input_gpkg, layer=layer)

    # select only line geometries
    is_line = gdf.geometry.type.isin(["LineString", "MultiLineString"])
    lines = gdf[is_line].copy()

    # compute length (in CRS units)
    lines["__length"] = lines.geometry.length

    # filter by threshold
    filtered = lines[lines["__length"] > threshold].drop(columns="__length")

    # write back, preserving layer name
    filtered.to_file(output_gpkg, layer=layer, driver="GPKG")



if __name__ == "__main__":
   dem = r"C:\L\OneDrive - Lichen\Documents\Projects\Atlas\FDAT\DEM_30m.tif"
   output_dir = r"C:\L\OneDrive - Lichen\Documents\Projects\Atlas\FDAT\Streams_30m"
   threshold = 1000  # ft²
   get_streams(
       dem=dem,
       output_dir=output_dir,
       threshold=threshold,
       overwrite=False,
       breach_depressions=False,
       create_thinned=False,
       precip_raster=None  # Optional, can be set to a PRISM raster path
   )

