# 0_get_streams.py

import math
import os
import warnings
from typing import Optional

import fiona
import geopandas as gpd
import rasterio
import whitebox
from rasterstats import zonal_stats
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
    print(
        f"Raster metadata for area calculations:\n"
        f"  Raster: {raster_path}\n"
        f"  CRS: {crs}\n"
        f"  Pixel size: {pixel_width:.2f} x {pixel_height:.2f} {linear_units}\n"
        f"  Pixel area: {pixel_area_native:.2f} {linear_units}² = {pixel_area_m2:.4f} m²"
    )
    return {
        "crs": crs,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "pixel_area_native": pixel_area_native,
        "pixel_area_m2": pixel_area_m2,
        "linear_units": linear_units,
    }


def _normalize_linear_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    unit = unit.lower().strip()
    if unit in {"metre", "meter", "metres", "meters", "m"}:
        return "meters"
    if unit in {"foot", "feet", "us survey foot", "foot_us", "ft"}:
        return "feet"
    return None


def _convert_length_units(value: float, from_units: str, to_units: str) -> float:
    """
    Convert a length value between feet and meters.
    """
    from_units = _normalize_linear_unit(from_units)
    to_units = _normalize_linear_unit(to_units)

    if from_units is None or to_units is None:
        raise ValueError(f"Unsupported unit conversion: {from_units} -> {to_units}")

    if from_units == to_units:
        return value

    if from_units == "feet" and to_units == "meters":
        return value * 0.3048

    if from_units == "meters" and to_units == "feet":
        return value / 0.3048

    raise ValueError(f"Unsupported unit conversion: {from_units} -> {to_units}")


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


def km2_to_cell_threshold(flow_accum_raster: str, threshold_km2: float) -> int:
    """
    Convert a drainage-area threshold in km² to a flow-accumulation threshold
    in number of contributing cells.
    """
    meta = _get_raster_area_metadata(flow_accum_raster)
    threshold_m2 = threshold_km2 * 1_000_000.0
    threshold_cells = math.ceil(threshold_m2 / meta["pixel_area_m2"])
    return int(threshold_cells)


def add_DA_to_stream(
    streams_gpkg: str,
    flow_accum_raster: str,
    buffer_cells: float = 0.75,
) -> None:
    """
    Adds drainage area in km² to stream features in `streams_gpkg` using a
    flow-accumulation raster.

    Assumes the flow accumulation raster stores upslope contributing area as
    number of cells.
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

    streams["DA_km2"] = [
        (s["max"] * meta["pixel_area_m2"] * 1e-6) if s.get("max") is not None else None
        for s in stats
    ]
    streams["DA_mi2"] = streams["DA_km2"] * 0.386102

    streams.to_file(streams_gpkg, driver="GPKG")


def add_precip_to_streams(
    streams_gpkg_path: str,
    precip_raster_path: str,
    precip_field_mm: str = "ann_precip_mm_yr",
    buffer_cells: float = 0.75,
) -> str:
    """
    Sample a precipitation raster in mm/yr and write mean precipitation to each
    stream feature.
    """
    streams = gpd.read_file(streams_gpkg_path)

    if streams.empty:
        warnings.warn(f"No stream features found in {streams_gpkg_path}. Skipping precip calc.")
        return streams_gpkg_path

    try:
        meta = _get_raster_area_metadata(precip_raster_path)
    except ValueError as e:
        warnings.warn(str(e))
        raise

    raster_crs = meta["crs"]
    with rasterio.open(precip_raster_path) as src:
        nodata = src.nodata

    if streams.crs is None:
        warnings.warn(
            f"Streams layer has no CRS in {streams_gpkg_path}. Assuming raster CRS: {raster_crs}."
        )
        streams = streams.set_crs(raster_crs)
    elif streams.crs != raster_crs:
        streams = streams.to_crs(raster_crs)

    buffer_dist = max(meta["pixel_width"], meta["pixel_height"]) * buffer_cells
    buffers = streams.geometry.buffer(buffer_dist)

    stats = zonal_stats(
        buffers,
        precip_raster_path,
        stats=["mean"],
        nodata=nodata,
        all_touched=True,
    )

    streams[precip_field_mm] = [s["mean"] if s.get("mean") is not None else None for s in stats]
    streams["ann_precip_in_yr"] = streams[precip_field_mm] / 25.4

    streams.to_file(streams_gpkg_path, driver="GPKG")
    return streams_gpkg_path


def add_temp_to_streams(
    streams_gpkg_path: str,
    temperature_raster_path: str,
    temp_field_f: str = "ann_temp_F",
    buffer_cells: float = 0.75,
) -> str:
    """
    Sample a PRISM temperature raster in deg_C and write mean temperature to each
    stream feature in deg_F.
    """
    streams = gpd.read_file(streams_gpkg_path)

    if streams.empty:
        warnings.warn(f"No stream features found in {streams_gpkg_path}. Skipping temp calc.")
        return streams_gpkg_path

    try:
        meta = _get_raster_area_metadata(temperature_raster_path)
    except ValueError as e:
        warnings.warn(str(e))
        raise

    raster_crs = meta["crs"]
    with rasterio.open(temperature_raster_path) as src:
        nodata = src.nodata

    if streams.crs is None:
        warnings.warn(
            f"Streams layer has no CRS in {streams_gpkg_path}. Assuming raster CRS: {raster_crs}."
        )
        streams = streams.set_crs(raster_crs)
    elif streams.crs != raster_crs:
        streams = streams.to_crs(raster_crs)

    buffer_dist = max(meta["pixel_width"], meta["pixel_height"]) * buffer_cells
    buffers = streams.geometry.buffer(buffer_dist)

    stats = zonal_stats(
        buffers,
        temperature_raster_path,
        stats=["mean"],
        nodata=nodata,
        all_touched=True,
    )

    streams[temp_field_f] = [s["mean"] if s.get("mean") is not None else None for s in stats]

    streams["ann_temp_F"] = streams[temp_field_f] * 1.8 + 32
    streams.to_file(streams_gpkg_path, driver="GPKG")
    return streams_gpkg_path


def add_slope_to_streams(
    streams_gpkg_path: str,
    dem_path: str,
    dem_vertical_units: str,
    buffer_cells: float = 0.75,
) -> str:
    """
    Add slope attributes to stream features using:
        slope = (max elevation on line - min elevation on line) / line length

    Parameters
    ----------
    streams_gpkg_path : str
        Path to stream GeoPackage.
    dem_path : str
        DEM path used to sample elevation.
    dem_vertical_units : str
        Vertical units of the DEM elevations. Must be 'feet' or 'meters'.
    buffer_cells : float
        Buffer distance around line features expressed in raster cell widths.
        A small buffer is used so narrow lines reliably intersect DEM cells.

    Writes
    ------
    elev_min
        Minimum DEM elevation along buffered stream line, in original DEM vertical units.
    elev_max
        Maximum DEM elevation along buffered stream line, in original DEM vertical units.
    elev_min_ft
        Minimum DEM elevation along buffered stream line, in feet.
    elev_max_ft
        Maximum DEM elevation along buffered stream line, in feet.
    line_len
        Stream length in DEM horizontal CRS units.
    slope
        Unitless slope ratio, with rise converted to match horizontal units.
    slope_pct
        Slope expressed as percent.
    """
    streams = gpd.read_file(streams_gpkg_path)

    if streams.empty:
        warnings.warn(f"No stream features found in {streams_gpkg_path}. Skipping slope calc.")
        return streams_gpkg_path

    try:
        meta = _get_raster_area_metadata(dem_path)
    except ValueError as e:
        warnings.warn(str(e))
        raise

    dem_vertical_units = _normalize_linear_unit(dem_vertical_units)
    horizontal_units = _normalize_linear_unit(meta["linear_units"])

    if dem_vertical_units not in {"feet", "meters"}:
        raise ValueError("dem_vertical_units must be 'feet' or 'meters'")

    if horizontal_units not in {"feet", "meters"}:
        raise ValueError(
            f"Unsupported DEM horizontal CRS units for slope calculation: {meta['linear_units']}"
        )

    raster_crs = meta["crs"]
    with rasterio.open(dem_path) as src:
        nodata = src.nodata

    if streams.crs is None:
        warnings.warn(
            f"Streams layer has no CRS in {streams_gpkg_path}. Assuming DEM CRS: {raster_crs}."
        )
        streams = streams.set_crs(raster_crs)
    elif streams.crs != raster_crs:
        streams = streams.to_crs(raster_crs)

    streams = streams[streams.geometry.notnull()].copy()
    streams = streams[streams.is_valid].copy()

    buffer_dist = max(meta["pixel_width"], meta["pixel_height"]) * buffer_cells
    sample_geoms = streams.geometry.buffer(buffer_dist)

    stats = zonal_stats(
        sample_geoms,
        dem_path,
        stats=["min", "max"],
        nodata=nodata,
        all_touched=True,
    )

    streams["elev_min"] = [s["min"] if s.get("min") is not None else None for s in stats]
    streams["elev_max"] = [s["max"] if s.get("max") is not None else None for s in stats]

    streams["elev_min_ft"] = streams["elev_min"].apply(
        lambda x: _convert_length_units(x, dem_vertical_units, "feet") if x is not None else None
    )
    streams["elev_max_ft"] = streams["elev_max"].apply(
        lambda x: _convert_length_units(x, dem_vertical_units, "feet") if x is not None else None
    )

    streams["line_len"] = streams.geometry.length

    relief_native = streams["elev_max"] - streams["elev_min"]

    valid = (
        streams["elev_min"].notna()
        & streams["elev_max"].notna()
        & streams["line_len"].notna()
        & (streams["line_len"] > 0)
    )

    streams["slope"] = None

    relief_horizontal_units = relief_native.copy()
    relief_horizontal_units.loc[valid] = relief_native.loc[valid].apply(
        lambda x: _convert_length_units(x, dem_vertical_units, horizontal_units)
    )

    streams.loc[valid, "slope"] = (
        relief_horizontal_units.loc[valid] / streams.loc[valid, "line_len"]
    )

    streams.to_file(streams_gpkg_path, driver="GPKG")
    return streams_gpkg_path

def add_MAF_to_streams(
    streams_gpkg_path: str,
    precip_raster_path: Optional[str] = None,
    temperature_raster_path: Optional[str] = None,
) -> str:
    """
    Compute mean annual flow (MAF) and MAF-based channel dimensions.

    Vogel et al. 1999 regression used here:
        MAF_m3s = exp(-10.18) *
                  (DA_km2 ** 1.00269) *
                  (P_mm_yr ** 1.86412) *
                  ((T_F * 10.0) ** -1.1579)

    where:
        DA_km2 = drainage area in km²
        P_mm_yr = annual precipitation in mm/yr
        T_F = mean annual temperature in deg_F

    Legg & Olson 2015 use the Vogel 1999 MAF to regress for bankfull width and depth:
        MAF_width_Legg_ft = 1.04 * 4.09 * (MAF_cfs ** 0.478)
        MAF_depth_Legg_ft = 1.09 * 0.23 * (MAF_cfs ** 0.370)

    Notes
    -----
    - This function only computes MAF if `temperature_raster_path` is provided.
    - If `precip_raster_path` is provided, it will also sample precipitation.
    - Requires DA_km2 to already exist.
    """
    if temperature_raster_path is None:
        return streams_gpkg_path

    if precip_raster_path is not None:
        add_precip_to_streams(streams_gpkg_path, precip_raster_path)

    add_temp_to_streams(streams_gpkg_path, temperature_raster_path)

    streams = gpd.read_file(streams_gpkg_path)
    streams = streams[streams.geometry.notnull()]
    streams = streams[streams.is_valid].copy()

    required_fields = ["DA_km2", "ann_precip_mm_yr", "ann_temp_F"]
    missing = [field for field in required_fields if field not in streams.columns]
    if missing:
        warnings.warn(
            f"Cannot compute MAF. Missing required field(s): {', '.join(missing)}"
        )
        return streams_gpkg_path

    FT3S_PER_M3S = 35.3147
    FT_TO_M = 0.3048

    valid = (
        streams["DA_km2"].notna()
        & streams["ann_precip_mm_yr"].notna()
        & streams["ann_temp_F"].notna()
        & (streams["DA_km2"] > 0)
        & (streams["ann_precip_mm_yr"] > 0)
        & (streams["ann_temp_F"] > 0)
    )

    streams["MAF_cfs"] = None
    streams["MAF_width_Legg_ft"] = None
    streams["MAF_depth_Legg_ft"] = None

    maf_m3s = (
        math.exp(-10.18)
        * (streams.loc[valid, "DA_km2"] ** 1.00269)
        * (streams.loc[valid, "ann_precip_mm_yr"] ** 1.86412)
        * ((streams.loc[valid, "ann_temp_F"] * 10.0) ** -1.1579)
    )

    maf_cfs = maf_m3s * FT3S_PER_M3S

    maf_width_ft = 1.04 * 4.09 * (maf_cfs ** 0.478)
    maf_depth_ft = 1.09 * 0.23 * (maf_cfs ** 0.370)

    streams.loc[valid, "MAF_cfs"] = maf_cfs
    streams.loc[valid, "MAF_width_Legg_ft"] = maf_width_ft
    streams.loc[valid, "MAF_depth_Legg_ft"] = maf_depth_ft

    streams.to_file(streams_gpkg_path, driver="GPKG")
    return streams_gpkg_path


def add_BF_to_streams_Legg(
    streams_gpkg_path: str,
    precip_raster_path: Optional[str] = None,
    precip_fallback_mm_yr: Optional[float] = None,
) -> str:
    """
    Reads stream features from a GeoPackage, computes mean annual precipitation
    for each feature, and then computes bankfull width using Legg & Olson 2015.

    Precipitation handling
    ----------------------
    - If precip_raster_path is provided, it is sampled as mm/yr.
    - Otherwise, precip_fallback_mm_yr is used as a constant value.
    - If neither is provided, no Legg calculation is performed.
    """
    FT_TO_M = 0.3048

    if precip_raster_path is not None:
        add_precip_to_streams(streams_gpkg_path, precip_raster_path)

    streams = gpd.read_file(streams_gpkg_path)
    streams = streams[streams.geometry.notnull()]
    streams = streams[streams.is_valid].copy()

    if precip_raster_path is None:
        if precip_fallback_mm_yr is None:
            return streams_gpkg_path
        streams["ann_precip_mm_yr"] = precip_fallback_mm_yr
        streams["ann_precip_in_yr"] = precip_fallback_mm_yr / 25.4

    streams["BF_width_Legg_m"] = (
        FT_TO_M
        * 1.16
        * 0.91
        * (streams["DA_mi2"] ** 0.381)
        * (streams["ann_precip_in_yr"] ** 0.634)
    )

    streams.to_file(streams_gpkg_path, driver="GPKG")
    return streams_gpkg_path


def add_BF_to_streams_Beechie(streams_gpkg_path: str) -> str:
    """
    Reads stream features from a GeoPackage, computes bankfull width (m) and depth (m)
    using Beechie and Imaki 2014, section 3.3.

    Bankfull width = 0.177 * (DA_km2^0.397) * (P_cm_yr^0.453)
    """
    streams = gpd.read_file(streams_gpkg_path)

    if "ann_precip_cm_yr" not in streams.columns:
        return streams_gpkg_path

    if "BF_depth_Castro_m" not in streams.columns or "BF_width_Castro_m" not in streams.columns:
        return streams_gpkg_path

    streams["BF_width_Beechie_m"] = 0.177 * (streams["DA_km2"] ** 0.397) * ((streams["ann_precip_mm_yr"] / 10) ** 0.453)
    streams["BF_depth_Beechie_m"] = (
        streams["BF_width_Beechie_m"]
        * streams["BF_depth_Castro_m"]
        / streams["BF_width_Castro_m"]
    )

    streams.to_file(streams_gpkg_path, driver="GPKG")
    return streams_gpkg_path


def add_BF_to_streams_Castro(
    streams_gpkg_path: str,
    ecoregion: int = 0
) -> str:
    """
    Reads stream features from a GeoPackage, computes bankfull width (m) and depth (m)
    using Castro & Jackson 2001.

    Castro & Jackson Ecoregions:
    All Pacific Northwest (default) - 0
    Pacific Maritime Mountains - 1
    West Interior Basin and Range - 2
    Western Cordillera - 3
    """
    streams = gpd.read_file(streams_gpkg_path)
    km2_to_mi2 = 0.386102
    ft_to_m = 0.3048

    if ecoregion == 0:
        streams["BF_width_Castro_m"] = ft_to_m * 11.80 * ((streams["DA_km2"] * km2_to_mi2) ** 0.38)
        streams["BF_depth_Castro_m"] = ft_to_m * 1.13 * ((streams["DA_km2"] * km2_to_mi2) ** 0.24)

    if ecoregion == 1:
        streams["BF_width_Castro_m"] = ft_to_m * 12.39 * ((streams["DA_km2"] * km2_to_mi2) ** 0.43)
        streams["BF_depth_Castro_m"] = ft_to_m * 0.66 * ((streams["DA_km2"] * km2_to_mi2) ** 0.39)

    if ecoregion == 2:
        streams["BF_width_Castro_m"] = ft_to_m * 3.27 * ((streams["DA_km2"] * km2_to_mi2) ** 0.51)
        streams["BF_depth_Castro_m"] = ft_to_m * 0.79 * ((streams["DA_km2"] * km2_to_mi2) ** 0.24)

    if ecoregion == 3:
        streams["BF_width_Castro_m"] = ft_to_m * 9.40 * ((streams["DA_km2"] * km2_to_mi2) ** 0.42)
        streams["BF_depth_Castro_m"] = ft_to_m * 0.61 * ((streams["DA_km2"] * km2_to_mi2) ** 0.33)

    streams.to_file(streams_gpkg_path, driver="GPKG")
    return streams_gpkg_path


def get_streams(
    dem: str,
    output_dir: str,
    threshold_km2: float = 1.0,
    overwrite: bool = False,
    breach_depressions: bool = True,
    precip_raster: Optional[str] = None,
    precip_fallback_mm_yr: Optional[float] = None,
    castro_ecoregion: int = 0,
    temperature_raster: Optional[str] = None,
    dem_vertical_units: str = "meters",
):
    """
    Process a DEM to extract streams, save them to a GeoPackage, and optionally
    add drainage area, precipitation, MAF, bankfull dimensions, and line slope.

    Parameters
    ----------
    dem_vertical_units : str
        Vertical units of the DEM elevations. Must be 'feet' or 'meters'.
    """
    wbt = whitebox.WhiteboxTools()
    _ = WbEnvironment()

    try:
        dem_meta = _get_raster_area_metadata(dem)
    except ValueError as e:
        warnings.warn(str(e))
        raise

    os.makedirs(output_dir, exist_ok=True)

    filled_dem = os.path.join(output_dir, "filled_dem.tif")
    d8_pointer = os.path.join(output_dir, "d8_pointer.tif")
    flow_accum = os.path.join(output_dir, "flow_accum.tif")
    breached_dem = os.path.join(output_dir, "breached_dem.tif")

    if overwrite:
        for path in [filled_dem, d8_pointer, flow_accum, breached_dem]:
            if os.path.exists(path):
                os.remove(path)

    if breach_depressions and not os.path.exists(breached_dem):
        wbt.breach_depressions_least_cost(dem, breached_dem, 10)

    if not breach_depressions and not os.path.exists(filled_dem):
        wbt.fill_depressions(dem, filled_dem)

    src_dem = breached_dem if breach_depressions else filled_dem
    if not os.path.exists(d8_pointer) or not os.path.exists(flow_accum):
        wbt.d8_pointer(src_dem, d8_pointer)
        wbt.d8_flow_accumulation(src_dem, flow_accum)

    try:
        threshold_cells = km2_to_cell_threshold(flow_accum, threshold_km2)
    except ValueError as e:
        warnings.warn(str(e))
        raise

    threshold_label = str(threshold_km2).replace(".", "p")
    streams_raster = os.path.join(output_dir, f"streams_{threshold_label}km2.tif")

    if not os.path.exists(streams_raster):
        print(
            f"Extracting streams using threshold = {threshold_km2} km² "
            f"({threshold_cells} contributing cells)"
        )
        wbt.extract_streams(flow_accum, streams_raster, threshold_cells)

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

    print("Adding drainage area, slope, and bankfull dimensions...")
    add_DA_to_stream(streams_gpkg, flow_accum)
    add_slope_to_streams(
        streams_gpkg_path=streams_gpkg,
        dem_path=dem,
        dem_vertical_units=dem_vertical_units,
    )

    if temperature_raster is not None:
        add_MAF_to_streams(
            streams_gpkg_path=streams_gpkg,
            precip_raster_path=precip_raster,
            temperature_raster_path=temperature_raster,
        )

    add_BF_to_streams_Castro(streams_gpkg, ecoregion=castro_ecoregion)
    add_BF_to_streams_Beechie(streams_gpkg)
    add_BF_to_streams_Legg(
        streams_gpkg_path=streams_gpkg,
        precip_raster_path=precip_raster,
        precip_fallback_mm_yr=precip_fallback_mm_yr,
    )

    print(f"[✔] Streams extracted to: {streams_gpkg}")
    return streams_gpkg


if __name__ == "__main__":
    dem = r"C:\L\Lichen\Lichen - Documents\Projects\20240001.4_Tucan 5-15 (CTUIR)\07_GIS\Wenaha\DEMs\USGS_10m_EPSG2927.tif"
    output_dir = r"C:\L\Lichen\Lichen - Documents\Projects\20240001.4_Tucan 5-15 (CTUIR)\07_GIS\Wenaha\streams_10m"
    precip_raster = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\PRISM\prism_ppt_30yr_avg_mmyr.tif"  # PRISM mm/yr
    temperature_raster = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\PRISM\prism_tmean_30yr_avg_degC.tif"  # PRISM deg_C
    threshold_km2 = 5

    # User must specify DEM vertical units: "feet" or "meters"
    dem_vertical_units = "meters"

    # Ecoregions:
    # 0 = All Pacific Northwest
    # 1 = Pacific Maritime Mountains
    # 2 = West Interior Basin and Range
    # 3 = Western Cordillera
    castro_ecoregion = 2

    get_streams(
        dem=dem,
        output_dir=output_dir,
        threshold_km2=threshold_km2,
        overwrite=False,
        breach_depressions=True,
        precip_raster=precip_raster,
        precip_fallback_mm_yr=None,
        castro_ecoregion=castro_ecoregion,
        temperature_raster=temperature_raster,
        dem_vertical_units=dem_vertical_units,
    )