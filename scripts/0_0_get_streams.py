# dem_processor.py

import math
import os
import warnings
from typing import Optional

import fiona
import geopandas as gpd
import rasterio
import whitebox
from rasterstats import zonal_stats
from shapely.geometry import LineString, MultiLineString
from whitebox_workflows import WbEnvironment


def _get_raster_area_metadata(raster_path: str) -> dict:
    """
    Returns raster CRS and pixel area information needed to convert
    cell counts to physical area.

    Returns
    -------
    dict with keys:
        crs
        pixel_width
        pixel_height
        pixel_area_native
        pixel_area_m2
        linear_units
    """
    with rasterio.open(raster_path) as src:
        crs = src.crs
        xres, yres = src.res
        pixel_width = abs(xres)
        pixel_height = abs(yres)

    if crs is None:
        raise ValueError(f"Raster has no CRS: {raster_path}")

    if not crs.is_projected:
        raise ValueError(
            f"Raster is not projected: {raster_path} ({crs}). "
            "Drainage area and threshold conversions based on pixel area are not reliable "
            "for geographic CRS. Reproject the DEM/raster to a projected CRS first."
        )

    try:
        linear_units = crs.linear_units.lower() if crs.linear_units else None
    except Exception:
        linear_units = None

    pixel_area_native = pixel_width * pixel_height

    if linear_units in {"metre", "meter", "metres", "meters", "m"}:
        pixel_area_m2 = pixel_area_native
    elif linear_units in {"foot", "feet", "us survey foot", "foot_us", "ft"}:
        pixel_area_m2 = pixel_area_native * 0.09290304
    else:
        raise ValueError(
            f"Unrecognized projected CRS linear units '{linear_units}' for raster: {raster_path}"
        )

    return {
        "crs": crs,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "pixel_area_native": pixel_area_native,
        "pixel_area_m2": pixel_area_m2,
        "linear_units": linear_units,
    }


def km2_to_cell_threshold(flow_accum_raster: str, threshold_km2: float) -> int:
    """
    Convert a drainage-area threshold in km² to a flow-accumulation threshold
    in number of contributing cells.
    """
    meta = _get_raster_area_metadata(flow_accum_raster)
    threshold_m2 = threshold_km2 * 1_000_000.0
    threshold_cells = math.ceil(threshold_m2 / meta["pixel_area_m2"])
    return int(threshold_cells)


def get_streams(
    dem: str,
    output_dir: str,
    threshold_km2: float = 1.0,
    overwrite: bool = False,
    breach_depressions: bool = True,
    thin_n: int = 10,
    create_thinned: bool = True,
    precip_raster: Optional[str] = None,
):
    """
    Process a DEM to extract streams, save them to a GeoPackage, and optionally
    create a thinned centerline layer.

    Parameters
    ----------
    dem : str
        Input DEM path.
    output_dir : str
        Output directory.
    threshold_km2 : float, default 1.0
        Stream initiation threshold expressed as drainage area in square kilometers.
    overwrite : bool, default False
        Whether to overwrite intermediate outputs.
    breach_depressions : bool, default True
        If True, breach depressions. If False, fill depressions.
    thin_n : int, default 10
        Keep every nth vertex if create_thinned is True.
    create_thinned : bool, default True
        Whether to create a thinned centerline layer.
    precip_raster : Optional[str], default None
        Optional precipitation raster for future use.

    Returns
    -------
    str
        Path to stream GeoPackage.
    """
    # --- Whitebox setup ------------------------------------------------------
    wbt = whitebox.WhiteboxTools()
    _ = WbEnvironment()

    # --- Validate DEM CRS/projection early ----------------------------------
    try:
        dem_meta = _get_raster_area_metadata(dem)
    except ValueError as e:
        warnings.warn(str(e))
        raise

    # --- Prepare output directory -------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # --- Intermediate filenames ---------------------------------------------
    filled_dem = os.path.join(output_dir, "filled_dem.tif")
    d8_pointer = os.path.join(output_dir, "d8_pointer.tif")
    flow_accum = os.path.join(output_dir, "flow_accum.tif")
    breached_dem = os.path.join(output_dir, "breached_dem.tif")

    if overwrite:
        for path in [filled_dem, d8_pointer, flow_accum, breached_dem]:
            if os.path.exists(path):
                os.remove(path)

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

    # --- Convert threshold from km² to contributing cells -------------------
    try:
        threshold_cells = km2_to_cell_threshold(flow_accum, threshold_km2)
    except ValueError as e:
        warnings.warn(str(e))
        raise

    threshold_label = str(threshold_km2).replace(".", "p")
    streams_raster = os.path.join(output_dir, f"streams_{threshold_label}km2.tif")

    # --- Extract streams -----------------------------------------------------
    if not os.path.exists(streams_raster):
        print(
            f"Extracting streams using threshold = {threshold_km2} km² "
            f"({threshold_cells} contributing cells)"
        )
        wbt.extract_streams(flow_accum, streams_raster, threshold_cells)

    # --- Raster -> Shapefile -> GeoPackage ----------------------------------
    streams_shp = streams_raster.replace(".tif", ".shp")
    streams_gpkg = streams_raster.replace(".tif", ".gpkg")
    streams_layer = os.path.splitext(os.path.basename(streams_gpkg))[0]

    if not os.path.exists(streams_shp):
        wbt.raster_streams_to_vector(streams_raster, d8_pointer, streams_shp)

    gdf = gpd.read_file(streams_shp)
    dem_crs = dem_meta["crs"]

    if gdf.crs is None:
        gdf = gdf.set_crs(dem_crs)
    else:
        gdf = gdf.to_crs(dem_crs)

    gdf.to_file(streams_gpkg, layer=streams_layer, driver="GPKG")

    # --- Thin centerline if requested ---------------------------------------
    thinned_gpkg = None
    if create_thinned:
        print(f"Thinning centerline by keeping every {thin_n}th vertex...")
        thinned_gpkg = streams_gpkg.replace(".gpkg", "_thinned.gpkg")
        thin_centerline(
            input_gpkg=streams_gpkg,
            layer_name=streams_layer,
            output_gpkg=thinned_gpkg,
            output_layer=f"{streams_layer}_thinned",
            n=thin_n,
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
    da_field: str = "DA_km2",
    buffer_cells: float = 0.75,
) -> None:
    """
    Adds drainage area in km² to stream features in `streams_gpkg` using a
    flow-accumulation raster.

    Assumes the flow accumulation raster stores upslope contributing area as
    number of cells.

    Notes
    -----
    - Uses the maximum flow-accumulation value within a small buffer around
      each stream feature.
    - Pixel area is calculated automatically from the raster resolution and CRS.
    - Warns and exits if the raster is not projected.
    """
    streams = gpd.read_file(streams_gpkg)

    if streams.empty:
        warnings.warn(f"No stream features found in {streams_gpkg}. Skipping DA calculation.")
        return

    try:
        meta = _get_raster_area_metadata(flow_accum_raster)
    except ValueError as e:
        warnings.warn(str(e))
        return

    raster_crs = meta["crs"]
    nodata = None
    with rasterio.open(flow_accum_raster) as src:
        nodata = src.nodata

    if streams.crs is None:
        warnings.warn(
            f"Streams layer has no CRS in {streams_gpkg}. Assuming raster CRS: {raster_crs}."
        )
        streams = streams.set_crs(raster_crs)
    elif streams.crs != raster_crs:
        streams = streams.to_crs(raster_crs)

    buffer_dist = max(meta["pixel_width"], meta["pixel_height"]) * buffer_cells
    buffers = streams.geometry.buffer(buffer_dist)

    stats = zonal_stats(
        buffers,
        flow_accum_raster,
        stats=["max"],
        nodata=nodata,
        all_touched=True,
    )

    streams[da_field] = [
        (s["max"] * meta["pixel_area_m2"] * 1e-6) if s.get("max") is not None else None
        for s in stats
    ]

    streams.to_file(streams_gpkg, driver="GPKG")


def add_BF_to_streams_Legg(streams_gpkg_path: str, precip_raster_path: str = None) -> None:
    """
    Reads stream features from a GeoPackage, computes mean annual precipitation (cm)
    for each feature based on intersecting PRISM raster data, adds a new precip field,
    then computes bankfull width (m) and depth (m) using Legg & Olson 2015.
    """
    KM2_TO_MI2 = 0.386102
    CM_TO_IN = 1.0 / 2.54
    FT_TO_M = 0.3048

    streams = gpd.read_file(streams_gpkg_path)
    streams = streams[streams.geometry.notnull()]
    streams = streams[streams.is_valid]

    # Placeholder basin-average precipitation. Replace with raster-based approach if desired.
    streams["DA_mi2"] = streams["DA_km2"] * KM2_TO_MI2
    streams["ann_precip_in"] = 72.17 * CM_TO_IN
    streams["ann_precip_cm"] = 72.17

    streams["BF_width_Legg_m"] = (
        FT_TO_M
        * 1.16
        * 0.91
        * (streams["DA_mi2"] ** 0.381)
        * (streams["ann_precip_in"] ** 0.634)
    )

    streams["BF_depth_Legg_m"] = (
        0.0939
        * (streams["DA_km2"] ** 0.233)
        * (streams["ann_precip_cm"] ** 0.264)
    )

    streams.to_file(streams_gpkg_path, driver="GPKG")
    return streams_gpkg_path


def add_BF_to_streams_Castro(streams_gpkg_path: str) -> None:
    """
    Reads stream features from a GeoPackage, computes bankfull width (m) and depth (m)
    using Castro & Jackson 2001.
    """
    streams = gpd.read_file(streams_gpkg_path)
    km2_to_mi2 = 0.386102
    ft_to_m = 0.3048

    streams["BF_width_Castro_m"] = ft_to_m * 9.40 * ((streams["DA_km2"] * km2_to_mi2) ** 0.42)
    streams["BF_depth_Castro_m"] = ft_to_m * 0.61 * ((streams["DA_km2"] * km2_to_mi2) ** 0.33)

    streams.to_file(streams_gpkg_path, driver="GPKG")
    return streams_gpkg_path


def add_BF_to_streams_Beechie(streams_gpkg_path: str) -> None:
    """
    Reads stream features from a GeoPackage, computes bankfull width (m) and depth (m)
    using Beechie and Imaki 2013.
    """
    streams = gpd.read_file(streams_gpkg_path)
    P_cm_yr = 72.17

    streams["BF_width_Beechie_m"] = 0.177 * (streams["DA_km2"] ** 0.397) * (P_cm_yr ** 0.453)
    streams["BF_depth_Beechie_m"] = (
        streams["BF_width_Beechie_m"]
        * streams["BF_depth_Castro_m"]
        / streams["BF_width_Castro_m"]
    )

    streams.to_file(streams_gpkg_path, driver="GPKG")
    return streams_gpkg_path


def thin_centerline(
    input_gpkg: str,
    layer_name: str,
    output_gpkg: str,
    output_layer: Optional[str] = None,
    n: int = 10,
) -> None:
    """
    Keeps every nth vertex in each LineString (or part of a MultiLineString)
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
    threshold: float = 1200.0,
) -> None:
    """
    Read the first layer of `input_gpkg`, keep only LineString/MultiLineString
    features longer than `threshold` (in CRS units), and write them to `output_gpkg`.
    """
    layers = fiona.listlayers(input_gpkg)
    if not layers:
        raise ValueError(f"No layers found in {input_gpkg!r}")
    layer = layers[0]

    gdf = gpd.read_file(input_gpkg, layer=layer)

    is_line = gdf.geometry.type.isin(["LineString", "MultiLineString"])
    lines = gdf[is_line].copy()

    lines["__length"] = lines.geometry.length
    filtered = lines[lines["__length"] > threshold].drop(columns="__length")

    filtered.to_file(output_gpkg, layer=layer, driver="GPKG")


if __name__ == "__main__":
    dem = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\CTUIR Hidaway Creek\REM\Streams\USGS10m.tif"
    output_dir = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\CTUIR Hidaway Creek\REM\Streams"
    threshold_km2 = 3  # stream initiation threshold in square kilometers

    get_streams(
        dem=dem,
        output_dir=output_dir,
        threshold_km2=threshold_km2,
        overwrite=False,
        breach_depressions=True,
        create_thinned=False,
        precip_raster=None,
    )