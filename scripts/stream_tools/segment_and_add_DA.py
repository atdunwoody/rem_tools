#!/usr/bin/env python3
"""
segment_and_add_DA.py

Dissolve stream lines into a single network geometry, re-segment the network
using input segmentation lines, and add:
- DA_km2
- length_ft
- slope_ftft
- optional Legg bankfull width/depth fields if a precip raster is supplied

Inputs are set directly in the script.
"""

import os
import warnings

import fiona
import geopandas as gpd
import numpy as np
import rasterio
from rasterstats import zonal_stats
from shapely.geometry import GeometryCollection, LineString, MultiLineString
from shapely.ops import linemerge, split, unary_union


def _get_first_layer(gpkg_path: str) -> str:
    layers = fiona.listlayers(gpkg_path)
    if not layers:
        raise ValueError(f"No layers found in {gpkg_path!r}")
    return layers[0]


def _normalize_length_unit(unit_value: str) -> str:
    if unit_value is None:
        raise ValueError("Unit value is None.")

    u = str(unit_value).strip().lower()
    if u in {"m", "meter", "meters", "metre", "metres"}:
        return "m"
    if u in {"ft", "foot", "feet", "us survey foot", "foot_us"}:
        return "ft"

    raise ValueError(
        f"Unrecognized unit {unit_value!r}. Use one of: 'm', 'meter', 'meters', "
        f"'metre', 'metres', 'ft', 'foot', 'feet'."
    )


def _get_raster_area_metadata(raster_path: str) -> dict:
    """
    Returns raster CRS and pixel area information needed to convert
    cell counts to physical area.
    """
    with rasterio.open(raster_path) as src:
        crs = src.crs
        xres, yres = src.res
        pixel_width = abs(xres)
        pixel_height = abs(yres)
        nodata = src.nodata

    if crs is None:
        raise ValueError(f"Raster has no CRS: {raster_path}")

    if not crs.is_projected:
        raise ValueError(
            f"Raster is not projected: {raster_path} ({crs}). "
            "Drainage area and slope calculations are not reliable for geographic CRS. "
            "Reproject first."
        )

    try:
        linear_units_raw = crs.linear_units.lower() if crs.linear_units else None
    except Exception:
        linear_units_raw = None

    linear_units = _normalize_length_unit(linear_units_raw)
    pixel_area_native = pixel_width * pixel_height

    if linear_units == "m":
        pixel_area_m2 = pixel_area_native
        length_to_ft = 3.280839895
    elif linear_units == "ft":
        pixel_area_m2 = pixel_area_native * 0.09290304
        length_to_ft = 1.0
    else:
        raise ValueError(
            f"Unrecognized projected CRS linear units '{linear_units_raw}' for raster: {raster_path}"
        )

    return {
        "crs": crs,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "pixel_area_m2": pixel_area_m2,
        "nodata": nodata,
        "linear_units": linear_units,
        "length_to_ft": length_to_ft,
    }


def _flatten_to_lines(geom):
    if geom is None or geom.is_empty:
        return []

    if isinstance(geom, LineString):
        return [geom]

    if isinstance(geom, MultiLineString):
        return list(geom.geoms)

    if isinstance(geom, GeometryCollection):
        lines = []
        for g in geom.geoms:
            lines.extend(_flatten_to_lines(g))
        return lines

    return []


def _explode_multilines(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    gdf = gdf.explode(index_parts=False).reset_index(drop=True)
    return gdf


def dissolve_streams_to_single_geometry(streams: gpd.GeoDataFrame):
    geom = unary_union(streams.geometry.values)
    try:
        geom = linemerge(geom)
    except Exception:
        pass
    return geom


def segment_streams_by_lines(stream_geom, segmentation_lines_gdf: gpd.GeoDataFrame) -> list:
    splitter = unary_union(segmentation_lines_gdf.geometry.values)

    try:
        split_result = split(stream_geom, splitter)
        pieces = _flatten_to_lines(split_result)
    except Exception:
        warnings.warn(
            "Split operation failed. Attempting fallback by noding/unioning stream and segmentation lines."
        )
        combined = unary_union([stream_geom, splitter])
        try:
            combined = linemerge(combined)
        except Exception:
            pass
        pieces = _flatten_to_lines(combined)

    pieces = [g for g in pieces if g is not None and not g.is_empty and g.length > 0]
    return pieces


def add_da_to_segments(
    segments_gdf: gpd.GeoDataFrame,
    flow_accum_raster: str,
    da_field: str = "DA_km2",
    buffer_cells: float = 0.75,
) -> gpd.GeoDataFrame:
    """
    Add drainage area to segmented streams using maximum flow accumulation value
    within a small buffer around each segment.
    """
    meta = _get_raster_area_metadata(flow_accum_raster)
    raster_crs = meta["crs"]

    if segments_gdf.crs is None:
        warnings.warn(f"Segments layer has no CRS. Assuming raster CRS: {raster_crs}.")
        segments_gdf = segments_gdf.set_crs(raster_crs)
    elif segments_gdf.crs != raster_crs:
        segments_gdf = segments_gdf.to_crs(raster_crs)

    buffer_dist = max(meta["pixel_width"], meta["pixel_height"]) * buffer_cells
    buffers = segments_gdf.geometry.buffer(buffer_dist)

    stats = zonal_stats(
        buffers,
        flow_accum_raster,
        stats=["max"],
        nodata=meta["nodata"],
        all_touched=True,
    )

    segments_gdf[da_field] = [
        (s["max"] * meta["pixel_area_m2"] * 1e-6) if s.get("max") is not None else None
        for s in stats
    ]

    return segments_gdf


def add_length_ft(
    segments_gdf: gpd.GeoDataFrame,
    reference_raster: str,
    field_name: str = "length_ft",
) -> gpd.GeoDataFrame:
    """
    Add segment length in feet using CRS linear units from the reference raster.
    """
    meta = _get_raster_area_metadata(reference_raster)
    raster_crs = meta["crs"]

    if segments_gdf.crs is None:
        segments_gdf = segments_gdf.set_crs(raster_crs)
    elif segments_gdf.crs != raster_crs:
        segments_gdf = segments_gdf.to_crs(raster_crs)

    segments_gdf[field_name] = segments_gdf.geometry.length * meta["length_to_ft"]
    return segments_gdf


def _sample_dem_value(dem_dataset, x, y):
    """
    Sample one DEM value and convert nodata to np.nan.
    """
    val = next(dem_dataset.sample([(x, y)]))[0]
    if dem_dataset.nodata is not None and np.isclose(val, dem_dataset.nodata):
        return np.nan
    return float(val)


def _get_line_endpoints(line):
    coords = list(line.coords)
    return coords[0], coords[-1]


def add_slope_ftft(
    segments_gdf: gpd.GeoDataFrame,
    dem_raster: str,
    dem_vertical_units: str,
    field_name: str = "slope_ftft",
) -> gpd.GeoDataFrame:
    """
    Add slope as rise/run in ft/ft using DEM elevations sampled at segment endpoints.

    Parameters
    ----------
    dem_vertical_units : str
        Vertical units of DEM elevation values. Must be 'ft' or 'm'.
    """
    meta = _get_raster_area_metadata(dem_raster)
    raster_crs = meta["crs"]
    dem_vertical_units = _normalize_length_unit(dem_vertical_units)

    if segments_gdf.crs is None:
        warnings.warn(f"Segments CRS missing. Assuming DEM CRS: {raster_crs}.")
        segments_gdf = segments_gdf.set_crs(raster_crs)
    elif segments_gdf.crs != raster_crs:
        segments_gdf = segments_gdf.to_crs(raster_crs)

    if dem_vertical_units == "ft":
        z_to_ft = 1.0
    elif dem_vertical_units == "m":
        z_to_ft = 3.280839895
    else:
        raise ValueError("dem_vertical_units must be 'ft' or 'm'.")

    slopes = []

    with rasterio.open(dem_raster) as src:
        for geom in segments_gdf.geometry:
            if geom is None or geom.is_empty:
                slopes.append(None)
                continue

            if isinstance(geom, MultiLineString):
                parts = list(geom.geoms)
                geom_use = max(parts, key=lambda g: g.length)
            else:
                geom_use = geom

            start_xy, end_xy = _get_line_endpoints(geom_use)
            z1 = _sample_dem_value(src, *start_xy)
            z2 = _sample_dem_value(src, *end_xy)

            if np.isnan(z1) or np.isnan(z2):
                slopes.append(None)
                continue

            run_ft = geom_use.length * meta["length_to_ft"]
            if run_ft == 0:
                slopes.append(None)
                continue

            rise_ft = abs(z2 - z1) * z_to_ft
            slopes.append(rise_ft / run_ft)

    segments_gdf[field_name] = slopes
    return segments_gdf


def add_precip_to_segments(
    segments_gdf: gpd.GeoDataFrame,
    precip_raster: str,
    precip_units: str,
    field_name_cm: str = "ann_precip_cm",
    field_name_in: str = "ann_precip_in",
    stats_buffer_cells: float = 0.75,
) -> gpd.GeoDataFrame:
    """
    Add mean annual precipitation to segments from a raster.

    Parameters
    ----------
    precip_units : str
        Units of precip raster values. Must be 'cm', 'mm', 'in', or 'm'.
    """
    valid_units = {"cm", "mm", "in", "m"}
    precip_units = str(precip_units).strip().lower()
    if precip_units not in valid_units:
        raise ValueError(f"precip_units must be one of {sorted(valid_units)}")

    meta = _get_raster_area_metadata(precip_raster)
    raster_crs = meta["crs"]

    if segments_gdf.crs is None:
        warnings.warn(f"Segments CRS missing. Assuming precip raster CRS: {raster_crs}.")
        segments_gdf = segments_gdf.set_crs(raster_crs)
    elif segments_gdf.crs != raster_crs:
        segments_gdf = segments_gdf.to_crs(raster_crs)

    buffer_dist = max(meta["pixel_width"], meta["pixel_height"]) * stats_buffer_cells
    buffers = segments_gdf.geometry.buffer(buffer_dist)

    stats = zonal_stats(
        buffers,
        precip_raster,
        stats=["mean"],
        nodata=meta["nodata"],
        all_touched=True,
    )

    precip_vals = [s.get("mean") if s.get("mean") is not None else None for s in stats]

    if precip_units == "cm":
        precip_cm = precip_vals
    elif precip_units == "mm":
        precip_cm = [v / 10.0 if v is not None else None for v in precip_vals]
    elif precip_units == "m":
        precip_cm = [v * 100.0 if v is not None else None for v in precip_vals]
    elif precip_units == "in":
        precip_cm = [v * 2.54 if v is not None else None for v in precip_vals]
    else:
        raise ValueError(f"Unsupported precip_units: {precip_units}")

    segments_gdf[field_name_cm] = precip_cm
    segments_gdf[field_name_in] = [
        (v / 2.54) if v is not None else None for v in precip_cm
    ]

    return segments_gdf


def add_BF_to_streams_Legg(
    segments_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Add Legg bankfull width/depth fields based on DA_km2 and segment precipitation fields.

    Requires:
    - DA_km2
    - ann_precip_cm
    - ann_precip_in

    Adds:
    - DA_mi2
    - BF_width_Legg_m
    - BF_depth_Legg_m
    """
    required_fields = ["DA_km2", "ann_precip_cm", "ann_precip_in"]
    missing = [f for f in required_fields if f not in segments_gdf.columns]
    if missing:
        raise ValueError(f"Missing required fields for Legg BF calculation: {missing}")

    KM2_TO_MI2 = 0.386102
    FT_TO_M = 0.3048

    segments_gdf["DA_mi2"] = segments_gdf["DA_km2"] * KM2_TO_MI2

    segments_gdf["BF_width_Legg_m"] = (
        FT_TO_M
        * 1.16
        * 0.91
        * (segments_gdf["DA_mi2"] ** 0.381)
        * (segments_gdf["ann_precip_in"] ** 0.634)
    )

    segments_gdf["BF_depth_Legg_m"] = (
        0.0939
        * (segments_gdf["DA_km2"] ** 0.233)
        * (segments_gdf["ann_precip_cm"] ** 0.264)
    )

    return segments_gdf


def segment_and_add_metrics(
    stream_gpkg: str,
    segmentation_gpkg: str,
    flow_accum_raster: str,
    dem_raster: str,
    dem_vertical_units: str,
    output_gpkg: str,
    output_layer: str = "segmented_streams",
    min_length: float = 0.0,
    precip_raster: str | None = None,
    precip_units: str | None = None,
) -> str:
    """
    Main workflow:
    1. Read streams and segmentation lines
    2. Reproject to DEM/raster CRS if needed
    3. Dissolve streams to a single geometry
    4. Split by segmentation lines
    5. Add DA_km2, length_ft, slope_ftft
    6. Optionally add precipitation and Legg BF fields if a precip raster is supplied
    7. Write output gpkg
    """
    dem_vertical_units = _normalize_length_unit(dem_vertical_units)

    if precip_raster is not None and precip_units is None:
        raise ValueError("If precip_raster is supplied, precip_units must also be supplied.")

    stream_layer = _get_first_layer(stream_gpkg)
    seg_layer = _get_first_layer(segmentation_gpkg)

    streams = gpd.read_file(stream_gpkg, layer=stream_layer)
    seg_lines = gpd.read_file(segmentation_gpkg, layer=seg_layer)

    streams = streams[streams.geometry.notnull() & ~streams.geometry.is_empty].copy()
    seg_lines = seg_lines[seg_lines.geometry.notnull() & ~seg_lines.geometry.is_empty].copy()

    streams = streams[streams.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    seg_lines = seg_lines[seg_lines.geometry.type.isin(["LineString", "MultiLineString"])].copy()

    if streams.empty:
        raise ValueError("No line geometries found in stream GeoPackage.")
    if seg_lines.empty:
        raise ValueError("No line geometries found in segmentation GeoPackage.")

    dem_meta = _get_raster_area_metadata(dem_raster)
    raster_crs = dem_meta["crs"]

    if streams.crs is None:
        warnings.warn(f"Streams CRS missing. Assuming DEM CRS: {raster_crs}.")
        streams = streams.set_crs(raster_crs)
    elif streams.crs != raster_crs:
        streams = streams.to_crs(raster_crs)

    if seg_lines.crs is None:
        warnings.warn(f"Segmentation CRS missing. Assuming DEM CRS: {raster_crs}.")
        seg_lines = seg_lines.set_crs(raster_crs)
    elif seg_lines.crs != raster_crs:
        seg_lines = seg_lines.to_crs(raster_crs)

    streams = _explode_multilines(streams)
    seg_lines = _explode_multilines(seg_lines)

    dissolved_stream = dissolve_streams_to_single_geometry(streams)
    segmented_geoms = segment_streams_by_lines(dissolved_stream, seg_lines)

    if not segmented_geoms:
        raise ValueError("No output segments were created.")

    segments = gpd.GeoDataFrame(
        {"segment_id": range(1, len(segmented_geoms) + 1)},
        geometry=segmented_geoms,
        crs=raster_crs,
    )

    if min_length > 0:
        segments = segments[segments.geometry.length > min_length].copy()
        segments = segments.reset_index(drop=True)
        segments["segment_id"] = range(1, len(segments) + 1)

    segments = add_da_to_segments(segments, flow_accum_raster, da_field="DA_km2")
    segments = add_length_ft(segments, dem_raster, field_name="length_ft")
    segments = add_slope_ftft(
        segments,
        dem_raster,
        dem_vertical_units=dem_vertical_units,
        field_name="slope_ftft",
    )

    if precip_raster is not None:
        segments = add_precip_to_segments(
            segments,
            precip_raster=precip_raster,
            precip_units=precip_units,
            field_name_cm="ann_precip_cm",
            field_name_in="ann_precip_in",
        )
        segments = add_BF_to_streams_Legg(segments)

    out_dir = os.path.dirname(output_gpkg)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    segments.to_file(output_gpkg, layer=output_layer, driver="GPKG")
    return output_gpkg


if __name__ == "__main__":
    flow_accum_raster = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\DEMs\Trout-Bear Cr\Full length analysis\streams\10m full watershed\flow_accum.tif"
    stream_gpkg = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\DEMs\Trout-Bear Cr\Full length analysis\streams\10m full watershed\streams_10km2.gpkg"
    segmentation_gpkg = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\Klickitat\TroutBearCreeks_RestorationOpportunities\Reach breaks.gpkg"
    dem_raster = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\DEMs\Trout-Bear Cr\Full length analysis\TB DEM merged 1,5ft.tif"

    # REQUIRED: explicitly specify DEM vertical units: "ft" or "m"
    dem_vertical_units = "ft"

    # OPTIONAL: only needed if you want Legg BF fields
    precip_raster = None
    precip_units = None
    # precip_raster = r"C:\path\to\annual_precip.tif"
    # precip_units = "mm"

    output_gpkg = r"C:\L\Lichen\Lichen - Documents\Projects\20250008_Geomorph Cons (YKFP)\07_GIS\DEMs\Trout-Bear Cr\Full length analysis\streams\10m full watershed\streams_10km2_segmented_metrics.gpkg"
    output_layer = "segmented_streams"
    min_length = 0.0

    out = segment_and_add_metrics(
        stream_gpkg=stream_gpkg,
        segmentation_gpkg=segmentation_gpkg,
        flow_accum_raster=flow_accum_raster,
        dem_raster=dem_raster,
        dem_vertical_units=dem_vertical_units,
        output_gpkg=output_gpkg,
        output_layer=output_layer,
        min_length=min_length,
        precip_raster=precip_raster,
        precip_units=precip_units,
    )

    print(f"[OK] Wrote segmented streams with metrics to: {out}")