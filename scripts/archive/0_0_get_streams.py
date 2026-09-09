# get_streams.py

import math
import os
import warnings
from typing import Optional, Sequence

import fiona
import geopandas as gpd
import numpy as np
import rasterio
import whitebox
from rasterstats import zonal_stats
from shapely.geometry import LineString, MultiLineString
from whitebox_workflows import WbEnvironment


M2_TO_SQMI = 3.861021585424458e-7
SQMI_TO_KM2 = 2.589988110336
IN_TO_CM = 2.54
MM_TO_IN = 1.0 / 25.4
FT_TO_M = 0.3048


# -----------------------------------------------------------------------------
# Raster CRS and unit helpers
# -----------------------------------------------------------------------------

def _unit_name(unit: Optional[str]) -> str:
    return (unit or "").strip().lower().replace("_", " ").replace("-", " ")


def _crs_xy_units_to_m(crs) -> float:
    """
    Return the conversion factor from the raster/vector CRS horizontal units to meters.

    Example
    -------
    - projected CRS in meters: returns 1.0
    - projected CRS in feet: returns about 0.3048
    - projected CRS in US survey feet: returns about 0.3048006096
    """
    if crs is None:
        raise ValueError("CRS is None. Cannot determine horizontal units.")

    if not crs.is_projected:
        raise ValueError(
            f"CRS is not projected: {crs}. Reproject to a projected CRS before "
            "calculating drainage area or stream slope."
        )

    # Rasterio CRS usually exposes a reliable linear_units_factor for projected CRS.
    try:
        factor = crs.linear_units_factor
        if isinstance(factor, Sequence) and not isinstance(factor, str) and len(factor) >= 2:
            factor_value = float(factor[1])
        else:
            factor_value = float(factor)

        if np.isfinite(factor_value) and factor_value > 0:
            return factor_value
    except Exception:
        pass

    # Conservative fallback for common unit names.
    try:
        units = _unit_name(crs.linear_units)
    except Exception:
        units = ""

    if units in {"metre", "meter", "metres", "meters", "m"}:
        return 1.0
    if units in {"foot", "feet", "ft", "international foot"}:
        return 0.3048
    if units in {"us survey foot", "us survey feet", "survey foot", "foot us"}:
        return 1200.0 / 3937.0

    raise ValueError(
        f"Could not determine projected CRS horizontal unit conversion to meters. "
        f"CRS={crs}, linear_units={units!r}."
    )


def _z_units_to_xy_units_factor(z_units: str, xy_units_to_m: float) -> float:
    """
    Return a multiplier that converts DEM z values to the CRS horizontal units.

    Slope is dimensionless only when vertical and horizontal distances are in the
    same units. If the DEM has meter elevations and a foot-based CRS, this returns
    about 3.28084 so elevations are converted to feet before fitting slope.

    Accepted z_units:
    - "same_as_xy", "xy", "crs": no conversion
    - "meter", "metre", "m"
    - "foot", "feet", "ft"
    - "us_survey_foot", "us survey foot", "survey foot"
    """
    units = _unit_name(z_units)

    if units in {"same as xy", "same_as_xy", "xy", "crs", "horizontal"}:
        return 1.0

    if units in {"metre", "meter", "metres", "meters", "m"}:
        z_units_to_m = 1.0
    elif units in {"foot", "feet", "ft", "international foot"}:
        z_units_to_m = 0.3048
    elif units in {"us survey foot", "us survey feet", "survey foot", "foot us"}:
        z_units_to_m = 1200.0 / 3937.0
    else:
        raise ValueError(
            f"Unsupported dem_z_units={z_units!r}. Use 'same_as_xy', 'meter', "
            "'foot', or 'us_survey_foot'."
        )

    return z_units_to_m / xy_units_to_m


def _get_raster_area_metadata(raster_path: str) -> dict:
    """
    Returns raster CRS, horizontal-unit conversion, and pixel area information.
    """
    with rasterio.open(raster_path) as src:
        crs = src.crs
        xres, yres = src.res
        pixel_width = abs(float(xres))
        pixel_height = abs(float(yres))

    if crs is None:
        raise ValueError(f"Raster has no CRS: {raster_path}")

    xy_units_to_m = _crs_xy_units_to_m(crs)
    pixel_area_native = pixel_width * pixel_height
    pixel_area_m2 = pixel_area_native * (xy_units_to_m ** 2)

    try:
        linear_units = crs.linear_units
    except Exception:
        linear_units = None

    return {
        "crs": crs,
        "linear_units": linear_units,
        "xy_units_to_m": xy_units_to_m,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "pixel_area_native": pixel_area_native,
        "pixel_area_m2": pixel_area_m2,
    }


def km2_to_cell_threshold(reference_raster: str, threshold_km2: float) -> int:
    """
    Convert a drainage-area threshold in km² to a contributing-cell threshold.
    """
    meta = _get_raster_area_metadata(reference_raster)
    threshold_m2 = threshold_km2 * 1_000_000.0
    return int(math.ceil(threshold_m2 / meta["pixel_area_m2"]))


def _write_gpkg(gdf: gpd.GeoDataFrame, gpkg_path: str, layer: Optional[str] = None) -> None:
    """Write a GeoDataFrame to a GeoPackage, preserving the intended layer."""
    if layer is None:
        existing_layers = fiona.listlayers(gpkg_path) if os.path.exists(gpkg_path) else []
        layer = existing_layers[0] if existing_layers else os.path.splitext(os.path.basename(gpkg_path))[0]

    gdf.to_file(gpkg_path, layer=layer, driver="GPKG")


def _read_gpkg(gpkg_path: str, layer: Optional[str] = None) -> tuple[gpd.GeoDataFrame, str]:
    """Read a GeoPackage and return both the GeoDataFrame and layer name."""
    if layer is None:
        layers = fiona.listlayers(gpkg_path)
        if not layers:
            raise ValueError(f"No layers found in {gpkg_path!r}")
        layer = layers[0]

    return gpd.read_file(gpkg_path, layer=layer), layer


# -----------------------------------------------------------------------------
# Raster value assignment
# -----------------------------------------------------------------------------

def add_raster_mean_to_streams(
    streams_gpkg_path: str,
    raster_path: str,
    output_field: str,
    all_touched: bool = True,
    layer: Optional[str] = None,
) -> None:
    """
    Adds the mean raster value intersecting each stream feature.

    The stream layer keeps its original CRS. A temporary geometry copy is
    reprojected to the raster CRS before sampling.
    """
    if raster_path is None:
        warnings.warn(f"No raster path provided for {output_field}. Skipping.")
        return

    streams, layer = _read_gpkg(streams_gpkg_path, layer)
    streams = streams[streams.geometry.notnull()].copy()
    streams = streams[streams.is_valid].copy()

    if streams.empty:
        warnings.warn(f"No valid stream features found. Skipping {output_field}.")
        return

    if streams.crs is None:
        raise ValueError(
            f"Streams layer has no CRS in {streams_gpkg_path}. Set the stream CRS first."
        )

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata

    if raster_crs is None:
        raise ValueError(f"Raster has no CRS: {raster_path}")

    streams_for_stats = streams.to_crs(raster_crs) if streams.crs != raster_crs else streams.copy()

    stats = zonal_stats(
        vectors=streams_for_stats.geometry,
        raster=raster_path,
        stats=["mean"],
        nodata=nodata,
        all_touched=all_touched,
    )

    streams[output_field] = [s.get("mean") if s.get("mean") is not None else np.nan for s in stats]
    _write_gpkg(streams, streams_gpkg_path, layer)


def add_PRISM_to_streams(
    streams_gpkg_path: str,
    ppt_raster_path: str,
    tmean_raster_path: str,
    layer: Optional[str] = None,
) -> None:
    """
    Adds PRISM 30-year average precipitation and mean temperature to streams.

    Stored output fields:
    - ann_precip_in: annual precipitation in inches
    - tmean_degC: mean annual temperature in degrees C
    """
    add_raster_mean_to_streams(
        streams_gpkg_path=streams_gpkg_path,
        raster_path=ppt_raster_path,
        output_field="__ppt_mm_yr",
        layer=layer,
    )

    add_raster_mean_to_streams(
        streams_gpkg_path=streams_gpkg_path,
        raster_path=tmean_raster_path,
        output_field="tmean_degC",
        layer=layer,
    )

    streams, layer = _read_gpkg(streams_gpkg_path, layer)

    if "__ppt_mm_yr" in streams.columns:
        streams["ann_precip_in"] = streams["__ppt_mm_yr"] * MM_TO_IN
        streams = streams.drop(columns="__ppt_mm_yr")

    _write_gpkg(streams, streams_gpkg_path, layer)


# -----------------------------------------------------------------------------
# Drainage area and slope
# -----------------------------------------------------------------------------

def add_DA_to_stream(
    streams_gpkg: str,
    flow_accum_cells_raster: str,
    da_field: str = "DA_sqmi",
    da_km2_field: Optional[str] = None,
    buffer_cells: float = 0.75,
    layer: Optional[str] = None,
) -> None:
    """
    Adds drainage area to each stream feature from a cell-count flow-accumulation raster.

    Requirements
    ------------
    `flow_accum_cells_raster` must store D8 flow accumulation as number of upslope
    contributing cells. The script creates that raster using Whitebox with
    out_type="cells".

    Method
    ------
    1. Verify the raster has a projected CRS.
    2. Calculate pixel area in m² using raster resolution and CRS horizontal units.
    3. Buffer each stream by a fraction of a cell.
    4. Use the maximum contributing-cell count intersecting that buffer.
    5. Convert cells to square miles and, optionally, km².
    """
    streams, layer = _read_gpkg(streams_gpkg, layer)

    if streams.empty:
        warnings.warn(f"No stream features found in {streams_gpkg}. Skipping drainage area.")
        return

    meta = _get_raster_area_metadata(flow_accum_cells_raster)
    raster_crs = meta["crs"]

    with rasterio.open(flow_accum_cells_raster) as src:
        nodata = src.nodata

    if streams.crs is None:
        warnings.warn(
            f"Streams layer has no CRS in {streams_gpkg}. Assuming flow-accumulation CRS: {raster_crs}."
        )
        streams = streams.set_crs(raster_crs)
    elif streams.crs != raster_crs:
        streams = streams.to_crs(raster_crs)

    buffer_dist = max(meta["pixel_width"], meta["pixel_height"]) * buffer_cells
    buffers = streams.geometry.buffer(buffer_dist)

    stats = zonal_stats(
        buffers,
        flow_accum_cells_raster,
        # 99th percentile
        stats = ["percentile_99"],
        nodata=nodata,
        all_touched=True,
    )

    flow_cells = np.array(
        [s.get("percentile_99") if s.get("percentile_99") is not None else np.nan for s in stats],
        dtype=float,
    )

    streams[da_field] = flow_cells * meta["pixel_area_m2"] * M2_TO_SQMI

    if da_km2_field is not None:
        streams[da_km2_field] = flow_cells * meta["pixel_area_m2"] / 1_000_000.0

    _write_gpkg(streams, streams_gpkg, layer)


def add_dem_slope_to_streams(
    streams_gpkg: str,
    dem_raster: str,
    slope_field: str = "slope_m_m",
    slope_pct_field: Optional[str] = "slope_pct",
    sample_spacing: Optional[float] = None,
    min_samples: int = 3,
    dem_z_units: str = "same_as_xy",
    layer: Optional[str] = None,
) -> None:
    """
    Adds DEM-derived longitudinal slope to each stream feature.

    Method
    ------
    For each LineString or MultiLineString feature:
    1. Reproject stream geometry to the DEM CRS.
    2. Confirm the DEM CRS is projected and determine horizontal units.
    3. Convert sampled DEM elevations to the DEM horizontal units.
    4. Sample elevations along the stream at regular spacing.
    5. Fit elevation versus streamwise distance using least squares.
    6. Store the absolute slope as dimensionless rise/run.

    Parameters
    ----------
    dem_z_units : str, default "same_as_xy"
        Vertical units of DEM values. Use "same_as_xy" when the DEM elevations
        are in the same units as the DEM horizontal CRS. Use "meter", "foot", or
        "us_survey_foot" when they differ.
    """
    streams, layer = _read_gpkg(streams_gpkg, layer)

    if streams.empty:
        warnings.warn(f"No stream features found in {streams_gpkg}. Skipping DEM slope.")
        return

    if streams.crs is None:
        raise ValueError(f"Streams layer has no CRS in {streams_gpkg}. Set the stream CRS first.")

    dem_meta = _get_raster_area_metadata(dem_raster)
    dem_crs = dem_meta["crs"]
    z_to_xy = _z_units_to_xy_units_factor(dem_z_units, dem_meta["xy_units_to_m"])

    with rasterio.open(dem_raster) as src:
        nodata = src.nodata
        default_spacing = min(dem_meta["pixel_width"], dem_meta["pixel_height"])
        spacing = sample_spacing or default_spacing

        if spacing <= 0:
            raise ValueError("sample_spacing must be positive.")

        streams_for_slope = streams.to_crs(dem_crs) if streams.crs != dem_crs else streams.copy()

        def _sample_line_profile(line: LineString, start_offset: float = 0.0):
            if line is None or line.is_empty or line.length <= 0:
                return [], []

            distances = np.arange(0.0, line.length, spacing).tolist()
            if not distances or distances[-1] < line.length:
                distances.append(line.length)

            points = [line.interpolate(d) for d in distances]
            coords = [(pt.x, pt.y) for pt in points]
            sampled = list(src.sample(coords))

            valid_distances = []
            valid_elevations = []

            for d, value_array in zip(distances, sampled):
                if value_array is None or len(value_array) == 0:
                    continue

                z = float(value_array[0])

                if nodata is not None and np.isclose(z, nodata):
                    continue
                if not np.isfinite(z):
                    continue

                valid_distances.append(start_offset + d)
                valid_elevations.append(z * z_to_xy)

            return valid_distances, valid_elevations

        def _profile_slope(geom):
            if geom is None or geom.is_empty:
                return np.nan

            all_distances = []
            all_elevations = []
            offset = 0.0

            if isinstance(geom, LineString):
                distances, elevations = _sample_line_profile(geom, start_offset=0.0)
                all_distances.extend(distances)
                all_elevations.extend(elevations)

            elif isinstance(geom, MultiLineString):
                # This assumes the stored part order is hydraulically meaningful.
                # If not, dissolve/order the network before using this slope field.
                for part in geom.geoms:
                    distances, elevations = _sample_line_profile(part, start_offset=offset)
                    all_distances.extend(distances)
                    all_elevations.extend(elevations)
                    offset += part.length
            else:
                return np.nan

            if len(all_distances) < min_samples:
                return np.nan

            x = np.asarray(all_distances, dtype=float)
            z = np.asarray(all_elevations, dtype=float)
            valid = np.isfinite(x) & np.isfinite(z)

            if valid.sum() < min_samples:
                return np.nan

            x = x[valid]
            z = z[valid]

            if np.nanmax(x) == np.nanmin(x):
                return np.nan

            # Fit z = a * distance + b. Units are consistent because z has been
            # converted to DEM horizontal CRS units.
            a, _b = np.polyfit(x, z, 1)
            return abs(float(a))

        streams[slope_field] = streams_for_slope.geometry.apply(_profile_slope)

    if slope_pct_field is not None:
        streams[slope_pct_field] = streams[slope_field] * 100.0

    _write_gpkg(streams, streams_gpkg, layer)


# -----------------------------------------------------------------------------
# Bankfull equations
# -----------------------------------------------------------------------------

def _get_precip_in(streams: gpd.GeoDataFrame, fallback_precip_in: float = 72.17 / 2.54):
    """
    Returns annual precipitation in inches.

    The fallback is 72.17 cm converted to inches, consistent with the previous
    default precipitation value.
    """
    if "ann_precip_in" not in streams.columns:
        warnings.warn(
            "ann_precip_in not found. "
            f"Using fallback precipitation value of {fallback_precip_in:.2f} inches."
        )
        return fallback_precip_in

    return streams["ann_precip_in"]


def add_BF_to_streams_Legg(streams_gpkg_path: str, layer: Optional[str] = None) -> str:
    streams, layer = _read_gpkg(streams_gpkg_path, layer)
    streams = streams[streams.geometry.notnull()].copy()
    streams = streams[streams.is_valid].copy()

    if "DA_sqmi" not in streams.columns:
        raise ValueError("DA_sqmi field is required for Legg bankfull equations.")

    da_sqmi = streams["DA_sqmi"]
    da_km2 = da_sqmi * SQMI_TO_KM2
    precip_in = _get_precip_in(streams)
    precip_cm = precip_in * IN_TO_CM

    streams["BF_width_Legg_m"] = FT_TO_M * 1.16 * 0.91 * (da_sqmi ** 0.381) * (precip_in ** 0.634)
    streams["BF_depth_Legg_m"] = 0.0939 * (da_km2 ** 0.233) * (precip_cm ** 0.264)

    _write_gpkg(streams, streams_gpkg_path, layer)
    return streams_gpkg_path


def add_BF_to_streams_Castro(streams_gpkg_path: str, layer: Optional[str] = None) -> str:
    streams, layer = _read_gpkg(streams_gpkg_path, layer)

    if "DA_sqmi" not in streams.columns:
        raise ValueError("DA_sqmi field is required for Castro bankfull equations.")

    da_sqmi = streams["DA_sqmi"]
    streams["BF_width_Castro_m"] = FT_TO_M * 9.40 * (da_sqmi ** 0.42)
    streams["BF_depth_Castro_m"] = FT_TO_M * 0.61 * (da_sqmi ** 0.33)

    _write_gpkg(streams, streams_gpkg_path, layer)
    return streams_gpkg_path


def add_BF_to_streams_Beechie(streams_gpkg_path: str, layer: Optional[str] = None) -> str:
    streams, layer = _read_gpkg(streams_gpkg_path, layer)

    if "DA_sqmi" not in streams.columns:
        raise ValueError("DA_sqmi field is required for Beechie bankfull equations.")

    if "BF_width_Castro_m" not in streams.columns or "BF_depth_Castro_m" not in streams.columns:
        raise ValueError(
            "BF_width_Castro_m and BF_depth_Castro_m are required before computing "
            "Beechie-scaled depth. Run add_BF_to_streams_Castro first."
        )

    da_km2 = streams["DA_sqmi"] * SQMI_TO_KM2
    precip_in = _get_precip_in(streams)
    precip_cm = precip_in * IN_TO_CM

    streams["BF_width_Beechie_m"] = 0.177 * (da_km2 ** 0.397) * (precip_cm ** 0.453)

    # This is not a direct Beechie depth equation. It applies the Castro
    # depth:width ratio to the Beechie width estimate.
    streams["BF_depth_Beechie_scaled_m"] = (
        streams["BF_width_Beechie_m"]
        * streams["BF_depth_Castro_m"]
        / streams["BF_width_Castro_m"]
    )

    _write_gpkg(streams, streams_gpkg_path, layer)
    return streams_gpkg_path


# -----------------------------------------------------------------------------
# Line thinning and filtering utilities
# -----------------------------------------------------------------------------

def thin_centerline(
    input_gpkg: str,
    layer_name: str,
    output_gpkg: str,
    output_layer: Optional[str] = None,
    n: int = 2,
) -> None:
    """Keeps every nth vertex in each LineString or MultiLineString."""
    gdf = gpd.read_file(input_gpkg, layer=layer_name)

    if n < 1:
        raise ValueError("n must be >= 1")

    def _thin(geom):
        if geom is None or geom.is_empty:
            return geom

        if isinstance(geom, LineString):
            coords = list(geom.coords)
            if len(coords) <= 2:
                return geom
            pts = coords[::n]
            if coords[-1] not in pts:
                pts.append(coords[-1])
            if len(pts) < 2:
                return geom
            return LineString(pts)

        if isinstance(geom, MultiLineString):
            parts = []
            for part in geom.geoms:
                coords = list(part.coords)
                if len(coords) <= 2:
                    parts.append(part)
                    continue
                pts = coords[::n]
                if coords[-1] not in pts:
                    pts.append(coords[-1])
                if len(pts) >= 2:
                    parts.append(LineString(pts))
            return MultiLineString(parts) if parts else geom

        return geom

    gdf["geometry"] = gdf.geometry.apply(_thin)
    _write_gpkg(gdf, output_gpkg, output_layer or layer_name)


def threshold_lines_by_length(
    input_gpkg: str,
    output_gpkg: str,
    threshold: float = 1200.0,
) -> None:
    """
    Read the first layer of `input_gpkg`, keep only line features longer than
    `threshold` in CRS units, and write them to `output_gpkg`.
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
    _write_gpkg(filtered, output_gpkg, layer)


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------

def get_streams(
    dem: str,
    output_dir: str,
    threshold_km2: float = 1.0,
    overwrite: bool = False,
    breach_depressions: bool = True,
    thin_n: int = 10,
    create_thinned: bool = True,
    precip_raster: Optional[str] = None,
    temp_raster: Optional[str] = None,
    dem_z_units: str = "same_as_xy",
):
    """
    Process a DEM to extract streams, save them to a GeoPackage, optionally create
    a thinned centerline layer, and add drainage area, PRISM climate values, slope,
    and bankfull dimensions.

    Important unit behavior
    -----------------------
    - Flow accumulation is explicitly generated as contributing cell count.
    - Drainage area is calculated as cells * raster pixel area, with raster CRS
      horizontal units converted to meters.
    - Slope is calculated from DEM samples after converting elevations to DEM
      horizontal CRS units.
    - Set dem_z_units="meter" or "foot" if DEM elevations differ from the DEM
      horizontal CRS units.
    """
    wbt = whitebox.WhiteboxTools()
    _ = WbEnvironment()

    dem_meta = _get_raster_area_metadata(dem)

    os.makedirs(output_dir, exist_ok=True)

    filled_dem = os.path.join(output_dir, "filled_dem.tif")
    breached_dem = os.path.join(output_dir, "breached_dem.tif")
    d8_pointer = os.path.join(output_dir, "d8_pointer.tif")

    # Use an explicit filename so older non-cell accumulation rasters are not reused.
    flow_accum_cells = os.path.join(output_dir, "flow_accum_cells.tif")

    if overwrite:
        for path in [filled_dem, breached_dem, d8_pointer, flow_accum_cells]:
            if os.path.exists(path):
                os.remove(path)

    if breach_depressions and not os.path.exists(breached_dem):
        wbt.breach_depressions_least_cost(dem, breached_dem, 10)

    if not breach_depressions and not os.path.exists(filled_dem):
        wbt.fill_depressions(dem, filled_dem)

    src_dem = breached_dem if breach_depressions else filled_dem

    if not os.path.exists(d8_pointer):
        wbt.d8_pointer(src_dem, d8_pointer)

    if not os.path.exists(flow_accum_cells):
        print("Creating D8 flow accumulation as contributing cell count...")
        wbt.d8_flow_accumulation(src_dem, flow_accum_cells, out_type="cells")

    threshold_cells = km2_to_cell_threshold(dem, threshold_km2)
    threshold_label = str(threshold_km2).replace(".", "p")
    streams_raster = os.path.join(output_dir, f"streams_{threshold_label}km2.tif")

    if overwrite and os.path.exists(streams_raster):
        os.remove(streams_raster)

    if not os.path.exists(streams_raster):
        print(
            f"Extracting streams using threshold = {threshold_km2} km² "
            f"({threshold_cells} contributing cells)"
        )
        wbt.extract_streams(flow_accum_cells, streams_raster, threshold_cells)

    streams_shp = streams_raster.replace(".tif", ".shp")
    streams_gpkg = streams_raster.replace(".tif", ".gpkg")
    streams_layer = os.path.splitext(os.path.basename(streams_gpkg))[0]

    if overwrite:
        for path in [streams_shp, streams_gpkg]:
            if os.path.exists(path):
                if path.endswith(".gpkg"):
                    os.remove(path)
                # Whitebox may manage shapefile sidecars; leave sidecars alone here.

    if not os.path.exists(streams_shp):
        wbt.raster_streams_to_vector(streams_raster, d8_pointer, streams_shp)

    gdf = gpd.read_file(streams_shp)
    if gdf.crs is None:
        gdf = gdf.set_crs(dem_meta["crs"])
    else:
        gdf = gdf.to_crs(dem_meta["crs"])

    _write_gpkg(gdf, streams_gpkg, streams_layer)

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

    print(
        "Adding drainage area from cell-count flow accumulation "
        f"using pixel area = {dem_meta['pixel_area_m2']:.6f} m² "
        f"({dem_meta['linear_units']} CRS units)."
    )
    
    #add a unique ID to each stream feature for easier debugging and reference
    streams, layer = _read_gpkg(streams_gpkg, streams_layer)
    streams["stream_id"] = range(1, len(streams) + 1)
    _write_gpkg(streams, streams_gpkg, streams_layer)
    
    add_DA_to_stream(
        streams_gpkg=streams_gpkg,
        flow_accum_cells_raster=flow_accum_cells,
        da_field="DA_sqmi",
        da_km2_field="DA_km2",
        layer=streams_layer,
    )

    print(
        "Adding DEM-derived longitudinal slope with DEM z units "
        f"interpreted as {dem_z_units!r}."
    )
    add_dem_slope_to_streams(
        streams_gpkg=streams_gpkg,
        dem_raster=dem,
        slope_field="slope_ft_ft",
        slope_pct_field="slope_pct",
        dem_z_units=dem_z_units,
        layer=streams_layer,
    )

    if precip_raster is not None and temp_raster is not None:
        print("Adding PRISM precipitation in inches and temperature in degrees C...")
        add_PRISM_to_streams(
            streams_gpkg_path=streams_gpkg,
            ppt_raster_path=precip_raster,
            tmean_raster_path=temp_raster,
            layer=streams_layer,
        )
    else:
        warnings.warn(
            "PRISM precipitation and/or temperature raster not provided. Bankfull "
            "equations that use precipitation will fall back to the default value."
        )

    print("Adding bankfull dimensions...")
    add_BF_to_streams_Legg(streams_gpkg, layer=streams_layer)
    add_BF_to_streams_Castro(streams_gpkg, layer=streams_layer)
    add_BF_to_streams_Beechie(streams_gpkg, layer=streams_layer)

    print(f"[✔] Streams extracted to: {streams_gpkg}")
    return streams_gpkg


if __name__ == "__main__":
    dem = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\0_Data_in\0_Stream_Intermittency\rasters_USGS10m\USGS_10m_DEM.tif"
    threshold_km2 = 1
    output_dir = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\1_Analysis\Stream Intermittency\USGS 10m Streams"

    get_streams(
        dem=dem,
        output_dir=output_dir,
        threshold_km2=threshold_km2,
        overwrite=False,
        breach_depressions=True,
        create_thinned=True,
        precip_raster=r"C:\L\Lichen\Lichen - Documents\Library\GIS\PRISM\prism_ppt_30yr_avg_mmyr.tif",
        temp_raster=r"C:\L\Lichen\Lichen - Documents\Library\GIS\PRISM\prism_tmean_30yr_avg_degC.tif",
        # Keep this as "same_as_xy" if the DEM elevations are in the same units
        # as the projected CRS. Change to "meter", "foot", or "us_survey_foot"
        # if the vertical units differ from the horizontal CRS units.
        dem_z_units="same_as_xy",
    )
