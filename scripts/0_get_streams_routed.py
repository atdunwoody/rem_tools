


# get_streams.py

import math
import os
import sqlite3
import warnings
from typing import Optional, Sequence

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import whitebox
from rasterstats import zonal_stats
from shapely.geometry import LineString, MultiLineString
from shapely.ops import substring
from whitebox_workflows import WbEnvironment


dem = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\CFC Silver Creek\Field Data\LiDAR\USGS DEM 10m.tif"
threshold_km2 = 1
output_dir = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\CFC Silver Creek\Field Data\LiDAR\Streams"

# Optional: Values are in stream CRS units. Set both to None to skip segmentation.
segment_interval = 300.0
min_segment_length = 250.0

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
    """
    Write a GeoDataFrame to a GeoPackage, preserving the intended layer.
    """
    if layer is None:
        existing_layers = fiona.listlayers(gpkg_path) if os.path.exists(gpkg_path) else []
        layer = existing_layers[0] if existing_layers else os.path.splitext(os.path.basename(gpkg_path))[0]

    gdf.to_file(gpkg_path, layer=layer, driver="GPKG")


def _read_gpkg(gpkg_path: str, layer: Optional[str] = None) -> tuple[gpd.GeoDataFrame, str]:
    """
    Read a GeoPackage and return both the GeoDataFrame and layer name.
    """
    if layer is None:
        layers = fiona.listlayers(gpkg_path)
        if not layers:
            raise ValueError(f"No layers found in {gpkg_path!r}")
        layer = layers[0]

    return gpd.read_file(gpkg_path, layer=layer), layer


# -----------------------------------------------------------------------------
# Stream routing helpers
# -----------------------------------------------------------------------------

WHITEBOX_D8_OFFSETS = {
    # Whitebox native pointer convention. Rows increase downward.
    # Pointer grid:
    #   64 128   1
    #   32   0   2
    #   16   8   4
    1: (-1, 1),
    2: (0, 1),
    4: (1, 1),
    8: (1, 0),
    16: (1, -1),
    32: (0, -1),
    64: (-1, -1),
    128: (-1, 0),
}

ESRI_D8_OFFSETS = {
    # ESRI pointer convention. Rows increase downward.
    1: (0, 1),
    2: (1, 1),
    4: (1, 0),
    8: (1, -1),
    16: (0, -1),
    32: (-1, -1),
    64: (-1, 0),
    128: (-1, 1),
}


def _ids_to_text(ids) -> str:
    """
    Convert a sequence of numeric IDs to a pipe-delimited text field.

    GeoPackage vector fields should not store Python lists directly. A text
    field is more portable and remains easy to split later.
    """
    if ids is None:
        return ""

    if isinstance(ids, str):
        return ids

    try:
        values = [int(v) for v in ids if pd.notna(v)]
    except TypeError:
        if pd.isna(ids):
            return ""
        values = [int(ids)]

    return "|".join(str(v) for v in sorted(set(values)))


def _as_int_or_none(value):
    if value is None or pd.isna(value):
        return None
    return int(value)


def _write_table_to_gpkg(df: pd.DataFrame, gpkg_path: str, table_name: str) -> None:
    """
    Write a non-spatial table to an existing GeoPackage.

    The table is also registered in gpkg_contents as an attribute table when
    that metadata table exists.
    """
    table = df.copy()

    with sqlite3.connect(gpkg_path) as con:
        table.to_sql(table_name, con, if_exists="replace", index=False)

        try:
            con.execute("DELETE FROM gpkg_contents WHERE table_name = ?", (table_name,))
            con.execute(
                """
                INSERT INTO gpkg_contents (
                    table_name, data_type, identifier, description, last_change,
                    min_x, min_y, max_x, max_y, srs_id
                )
                VALUES (
                    ?, 'attributes', ?, '',
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    NULL, NULL, NULL, NULL, NULL
                )
                """,
                (table_name, table_name),
            )
        except sqlite3.OperationalError:
            # Non-standard or incomplete GeoPackage. The table itself was still written.
            pass


def _routing_table_for_gpkg(routing: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare a routing table for GeoPackage storage by converting list fields to text.
    """
    table = routing.copy()

    for field in ["upstream_link_ids", "all_upstream_link_ids"]:
        if field in table.columns:
            table[field] = table[field].apply(_ids_to_text)

    return table


def build_stream_link_routing(
    stream_links_raster: str,
    d8_pointer_raster: str,
    flow_accum_cells_raster: Optional[str] = None,
    esri_pntr: bool = False,
) -> pd.DataFrame:
    """
    Build immediate and cumulative upstream routing for Whitebox stream links.

    Parameters
    ----------
    stream_links_raster
        Raster from WhiteboxTools StreamLinkIdentifier.
    d8_pointer_raster
        D8 pointer raster used to create the stream network.
    flow_accum_cells_raster
        Optional D8 flow accumulation raster in contributing cells.
    esri_pntr
        Set True only when the D8 pointer was created using the ESRI pointer convention.

    Returns
    -------
    pandas.DataFrame
        One row per stream link, with direct downstream, direct upstream, and all
        upstream link IDs.
    """
    offsets = ESRI_D8_OFFSETS if esri_pntr else WHITEBOX_D8_OFFSETS

    with rasterio.open(stream_links_raster) as link_src:
        links = link_src.read(1)
        link_nodata = link_src.nodata
        link_transform = link_src.transform
        link_crs = link_src.crs

    with rasterio.open(d8_pointer_raster) as ptr_src:
        pntr = ptr_src.read(1)
        ptr_nodata = ptr_src.nodata

        if ptr_src.shape != links.shape:
            raise ValueError("Stream-link raster and D8 pointer raster shapes do not match.")

        if ptr_src.transform != link_transform:
            raise ValueError("Stream-link raster and D8 pointer raster transforms do not match.")

        if ptr_src.crs != link_crs:
            raise ValueError("Stream-link raster and D8 pointer raster CRS values do not match.")

    if flow_accum_cells_raster is not None:
        with rasterio.open(flow_accum_cells_raster) as acc_src:
            acc = acc_src.read(1)
            acc_meta = _get_raster_area_metadata(flow_accum_cells_raster)

            if acc_src.shape != links.shape:
                raise ValueError("Flow accumulation raster shape does not match stream-link raster.")

            if acc_src.transform != link_transform:
                raise ValueError("Flow accumulation raster transform does not match stream-link raster.")
    else:
        acc = None
        acc_meta = None

    valid_links = np.isfinite(links)

    if link_nodata is not None:
        valid_links &= links != link_nodata

    valid_links &= links > 0

    link_ids = np.unique(links[valid_links]).astype(int)

    downstream_candidates = {int(link_id): [] for link_id in link_ids}
    outlet_cells = {int(link_id): [] for link_id in link_ids}

    nrows, ncols = links.shape
    rows, cols = np.where(valid_links)

    for r, c in zip(rows, cols):
        link_id = int(links[r, c])
        p = pntr[r, c]

        if not np.isfinite(p):
            continue

        # A D8 value of 0 is a valid no-lower-neighbor/outlet code in Whitebox.
        # Treat it as an outlet even if a modified pointer raster reports 0 as nodata.
        if ptr_nodata is not None and p == ptr_nodata and int(p) != 0:
            continue

        p = int(p)

        if p not in offsets:
            outlet_cells[link_id].append((r, c))
            continue

        dr, dc = offsets[p]
        rr = r + dr
        cc = c + dc

        if rr < 0 or rr >= nrows or cc < 0 or cc >= ncols:
            outlet_cells[link_id].append((r, c))
            continue

        downstream_value = links[rr, cc]
        downstream_valid = np.isfinite(downstream_value)

        if link_nodata is not None:
            downstream_valid &= downstream_value != link_nodata

        downstream_valid &= downstream_value > 0

        if downstream_valid:
            downstream_link_id = int(downstream_value)

            if downstream_link_id != link_id:
                outlet_cells[link_id].append((r, c))
                downstream_candidates[link_id].append(downstream_link_id)
        else:
            outlet_cells[link_id].append((r, c))

    records = []

    for link_id in link_ids:
        link_id = int(link_id)
        candidates = downstream_candidates[link_id]
        unique_candidates = sorted(set(candidates))

        if len(unique_candidates) == 0:
            downstream_link_id = None
            flag = "outlet_or_no_downstream_stream_cell"
        elif len(unique_candidates) == 1:
            downstream_link_id = unique_candidates[0]
            flag = ""
        else:
            downstream_link_id = max(set(candidates), key=candidates.count)
            flag = f"multiple_downstream_candidates:{_ids_to_text(unique_candidates)}"

        outlets = outlet_cells[link_id]

        if acc is not None and outlets:
            outlet_row, outlet_col = max(outlets, key=lambda rc: acc[rc[0], rc[1]])
            outlet_flow_accum_cells = float(acc[outlet_row, outlet_col])
        elif outlets:
            outlet_row, outlet_col = outlets[0]
            outlet_flow_accum_cells = np.nan
        else:
            outlet_row = None
            outlet_col = None
            outlet_flow_accum_cells = np.nan
            flag = "no_outlet_cell_found" if flag == "" else f"{flag};no_outlet_cell_found"

        if outlet_row is not None and outlet_col is not None:
            outlet_x, outlet_y = rasterio.transform.xy(
                link_transform,
                outlet_row,
                outlet_col,
                offset="center",
            )
        else:
            outlet_x = np.nan
            outlet_y = np.nan

        if acc_meta is not None and np.isfinite(outlet_flow_accum_cells):
            outlet_da_sqmi = outlet_flow_accum_cells * acc_meta["pixel_area_m2"] * M2_TO_SQMI
            outlet_da_km2 = outlet_flow_accum_cells * acc_meta["pixel_area_m2"] / 1_000_000.0
        else:
            outlet_da_sqmi = np.nan
            outlet_da_km2 = np.nan

        records.append(
            {
                "link_id": link_id,
                "downstream_link_id": downstream_link_id,
                "outlet_row": outlet_row,
                "outlet_col": outlet_col,
                "outlet_x": outlet_x,
                "outlet_y": outlet_y,
                "outlet_flow_accum_cells": outlet_flow_accum_cells,
                "outlet_DA_sqmi": outlet_da_sqmi,
                "outlet_DA_km2": outlet_da_km2,
                "routing_flag": flag,
            }
        )

    routing = pd.DataFrame(records)

    upstream_lookup = (
        routing.dropna(subset=["downstream_link_id"])
        .assign(downstream_link_id=lambda x: x["downstream_link_id"].astype(int))
        .groupby("downstream_link_id")["link_id"]
        .apply(lambda s: sorted(map(int, s)))
        .to_dict()
    )

    routing["upstream_link_ids"] = routing["link_id"].map(upstream_lookup).apply(
        lambda x: x if isinstance(x, list) else []
    )
    routing["n_upstream_links"] = routing["upstream_link_ids"].apply(len)

    def _all_upstream(link_id: int) -> list[int]:
        found = set()
        stack = list(upstream_lookup.get(int(link_id), []))

        while stack:
            candidate = int(stack.pop())

            if candidate in found:
                continue

            found.add(candidate)
            stack.extend(upstream_lookup.get(candidate, []))

        return sorted(found)

    routing["all_upstream_link_ids"] = routing["link_id"].apply(_all_upstream)
    routing["n_all_upstream_links"] = routing["all_upstream_link_ids"].apply(len)

    return routing


def add_stream_link_routing_to_streams(
    streams_gpkg: str,
    stream_links_raster: str,
    routing: pd.DataFrame,
    layer: Optional[str] = None,
    buffer_cells: float = 0.75,
) -> None:
    """
    Add link IDs and link-level routing fields to a stream vector layer.

    Link IDs are assigned from the stream-link raster using the modal raster value
    within a small stream buffer.
    """
    streams, layer = _read_gpkg(streams_gpkg, layer)

    if streams.empty:
        warnings.warn(f"No stream features found in {streams_gpkg}. Skipping stream routing fields.")
        return

    if streams.crs is None:
        raise ValueError(f"Streams layer has no CRS in {streams_gpkg}. Set the stream CRS first.")

    meta = _get_raster_area_metadata(stream_links_raster)
    raster_crs = meta["crs"]

    with rasterio.open(stream_links_raster) as src:
        nodata = src.nodata

    streams_for_stats = streams.to_crs(raster_crs) if streams.crs != raster_crs else streams.copy()
    buffer_dist = max(meta["pixel_width"], meta["pixel_height"]) * buffer_cells
    buffers = streams_for_stats.geometry.buffer(buffer_dist)

    stats = zonal_stats(
        buffers,
        stream_links_raster,
        categorical=True,
        nodata=nodata,
        all_touched=True,
    )

    link_ids = []

    for stat in stats:
        counts = {}

        for value, count in stat.items():
            if value is None:
                continue

            try:
                link_value = int(float(value))
            except (TypeError, ValueError):
                continue

            if link_value <= 0:
                continue

            counts[link_value] = counts.get(link_value, 0) + int(count)

        if not counts:
            link_ids.append(pd.NA)
            continue

        modal_link_id, _modal_count = max(counts.items(), key=lambda item: item[1])
        link_ids.append(modal_link_id)

    streams["link_id"] = pd.Series(link_ids, dtype="Int64")

    # Keep the vector layers compact. Full upstream link lists and link-outlet
    # drainage-area fields are retained in the non-spatial link-routing table.
    join_fields = [
        "link_id",
        "downstream_link_id",
        "n_upstream_links",
        "n_all_upstream_links",
        "routing_flag",
    ]

    routing_for_join = routing[join_fields].copy()
    routing_for_join = routing_for_join.rename(
        columns={
            "downstream_link_id": "ds_link_id",
            "n_upstream_links": "n_us_links",
            "n_all_upstream_links": "n_all_us_l",
            "routing_flag": "link_route_flag",
        }
    )

    streams = streams.merge(routing_for_join, on="link_id", how="left")
    _write_gpkg(streams, streams_gpkg, layer)


def add_segment_routing_to_streams(
    streams_gpkg: str,
    layer: Optional[str] = None,
    segment_uid_field: str = "segment_uid",
    link_id_field: str = "link_id",
    downstream_link_field: str = "ds_link_id",
    segment_order_field: str = "segment_id",
    da_field: str = "DA_sqmi",
) -> pd.DataFrame:
    """
    Add regular-segment routing fields to a segmented stream layer.

    The within-link segment order is inferred from drainage area when available.
    The downstream-most segment in each link is connected to the upstream-most
    segment in the downstream link.
    """
    streams, layer = _read_gpkg(streams_gpkg, layer)

    required = [segment_uid_field, link_id_field, segment_order_field]
    missing = [field for field in required if field not in streams.columns]

    if missing:
        raise ValueError(
            f"Segment routing requires fields {required}; missing {missing}."
        )

    streams["__route_group"] = streams[link_id_field].apply(_as_int_or_none)
    streams["__downstream_group"] = (
        streams[downstream_link_field].apply(_as_int_or_none)
        if downstream_link_field in streams.columns
        else None
    )

    downstream_by_segment = {}
    downstream_tail_by_group = {}
    headwater_segment_by_group = {}
    flags_by_segment = {int(uid): "" for uid in streams[segment_uid_field]}

    for group_id, group in streams.dropna(subset=["__route_group"]).groupby("__route_group"):
        group = group.copy()
        group["__seg_order"] = group[segment_order_field].astype(float)
        ordered = group.sort_values("__seg_order")
        order_flag = ""

        if da_field in ordered.columns and ordered[da_field].notna().sum() >= 2:
            first_da = float(ordered[da_field].dropna().iloc[0])
            last_da = float(ordered[da_field].dropna().iloc[-1])

            if last_da < first_da:
                ordered = ordered.iloc[::-1].copy()
                order_flag = "segment_order_reversed_by_DA"
        else:
            order_flag = "segment_order_assumed"

        ordered_uids = [int(v) for v in ordered[segment_uid_field].tolist()]

        if not ordered_uids:
            continue

        headwater_segment_by_group[int(group_id)] = ordered_uids[0]
        downstream_tail_by_group[int(group_id)] = ordered_uids[-1]

        if order_flag:
            for uid in ordered_uids:
                flags_by_segment[uid] = order_flag

        for current_uid, downstream_uid in zip(ordered_uids[:-1], ordered_uids[1:]):
            downstream_by_segment[current_uid] = downstream_uid

    for group_id, tail_uid in downstream_tail_by_group.items():
        group_rows = streams[streams["__route_group"] == group_id]

        if group_rows.empty:
            continue

        downstream_group = _as_int_or_none(group_rows["__downstream_group"].iloc[0])

        if downstream_group is None:
            flags_by_segment[tail_uid] = (
                "outlet_segment"
                if flags_by_segment[tail_uid] == ""
                else f"{flags_by_segment[tail_uid]};outlet_segment"
            )
            continue

        downstream_head_uid = headwater_segment_by_group.get(downstream_group)

        if downstream_head_uid is None:
            flags_by_segment[tail_uid] = (
                f"downstream_link_missing:{downstream_group}"
                if flags_by_segment[tail_uid] == ""
                else f"{flags_by_segment[tail_uid]};downstream_link_missing:{downstream_group}"
            )
            continue

        downstream_by_segment[tail_uid] = downstream_head_uid

    upstream_lookup = {}

    for segment_uid, downstream_uid in downstream_by_segment.items():
        upstream_lookup.setdefault(int(downstream_uid), []).append(int(segment_uid))

    def _all_upstream_segments(segment_uid: int) -> list[int]:
        found = set()
        stack = list(upstream_lookup.get(int(segment_uid), []))

        while stack:
            candidate = int(stack.pop())

            if candidate in found:
                continue

            found.add(candidate)
            stack.extend(upstream_lookup.get(candidate, []))

        return sorted(found)

    route_records = []

    for _, row in streams.iterrows():
        segment_uid = int(row[segment_uid_field])
        downstream_uid = downstream_by_segment.get(segment_uid)
        upstream_uids = sorted(upstream_lookup.get(segment_uid, []))
        all_upstream_uids = _all_upstream_segments(segment_uid)

        route_records.append(
            {
                segment_uid_field: segment_uid,
                "ds_seg_uid": downstream_uid,
                "us_seg_uids": _ids_to_text(upstream_uids),
                "all_us_seg_uids": _ids_to_text(all_upstream_uids),
                "n_us_segs": len(upstream_uids),
                "n_all_us_s": len(all_upstream_uids),
                "seg_route_flag": flags_by_segment.get(segment_uid, ""),
            }
        )

    route_table = pd.DataFrame(route_records)
    streams = streams.drop(columns=["__route_group", "__downstream_group"], errors="ignore")
    streams = streams.merge(route_table, on=segment_uid_field, how="left")

    _write_gpkg(streams, streams_gpkg, layer)
    return route_table


def simplify_stream_layer_fields(
    streams_gpkg: str,
    layer: Optional[str] = None,
) -> None:
    """
    Keep a compact, analysis-ready set of output fields on the final stream layer.

    Full link-routing details remain in the non-spatial link-routing table. This
    function keeps segment-level routing, compact link context, drainage area,
    slope, PRISM fields, and bankfull regression fields when present.
    """
    streams, layer = _read_gpkg(streams_gpkg, layer)

    preferred_order = [
        # Source and segmentation identifiers.
        "stream_id",
        "source_stream_id",
        "segment_uid",
        "segment_id",
        "segment_len",
        # Compact link-level routing context.
        "link_id",
        "ds_link_id",
        "n_us_links",
        "n_all_us_l",
        "link_route_flag",
        # Segment-level routing.
        "ds_seg_uid",
        "us_seg_uids",
        "all_us_seg_uids",
        "n_us_segs",
        "n_all_us_s",
        "seg_route_flag",
        # Analysis attributes.
        "DA_sqmi",
        "DA_km2",
        "slope_ft_ft",
        "slope_pct",
        "ann_precip_in",
        "tmean_degC",
        # Bankfull regression outputs.
        "BF_width_Legg_m",
        "BF_depth_Legg_m",
        "BF_width_Castro_m",
        "BF_depth_Castro_m",
        "BF_width_Beechie_m",
        "BF_depth_Beechie_scaled_m",
    ]

    keep = [field for field in preferred_order if field in streams.columns]

    # Preserve any future bankfull fields without requiring this helper to be
    # updated each time a regression is added.
    for field in streams.columns:
        if field == streams.geometry.name:
            continue

        if field.startswith("BF_") and field not in keep:
            keep.append(field)

    keep.append(streams.geometry.name)
    streams = streams[keep].copy()

    _write_gpkg(streams, streams_gpkg, layer)


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

    streams[output_field] = [
        s.get("mean") if s.get("mean") is not None else np.nan
        for s in stats
    ]

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
# Stream segmentation
# -----------------------------------------------------------------------------

def _segment_linestring(
    line: LineString,
    segment_interval: float,
    min_segment_length: Optional[float] = None,
) -> list[LineString]:
    """
    Segment a LineString at a regular interval.

    If the final segment would be shorter than min_segment_length, it is merged
    with the previous segment for the same source feature.

    segment_interval and min_segment_length are in CRS units.
    """
    if line is None or line.is_empty or line.length <= 0:
        return []

    if segment_interval <= 0:
        raise ValueError("segment_interval must be positive.")

    if min_segment_length is not None and min_segment_length < 0:
        raise ValueError("min_segment_length must be >= 0.")

    length = float(line.length)

    if length <= segment_interval:
        return [line]

    breakpoints = list(np.arange(0.0, length, segment_interval))
    if breakpoints[-1] < length:
        breakpoints.append(length)

    segments = []

    for start, end in zip(breakpoints[:-1], breakpoints[1:]):
        if end <= start:
            continue

        seg = substring(line, start, end)

        if seg is None or seg.is_empty:
            continue

        if isinstance(seg, MultiLineString):
            parts = [part for part in seg.geoms if part.length > 0]
            if not parts:
                continue
            seg = max(parts, key=lambda g: g.length)

        if not isinstance(seg, LineString) or seg.length <= 0:
            continue

        segments.append(seg)

    if not segments:
        return [line]

    if (
        min_segment_length is not None
        and len(segments) > 1
        and segments[-1].length < min_segment_length
    ):
        # Rebuild the last two segments as one segment from the original line.
        merged_start = max(0.0, length - segments[-2].length - segments[-1].length)
        merged = substring(line, merged_start, length)

        if isinstance(merged, MultiLineString):
            parts = [part for part in merged.geoms if part.length > 0]
            if parts:
                merged = max(parts, key=lambda g: g.length)

        if isinstance(merged, LineString) and merged.length > 0:
            segments = segments[:-2] + [merged]

    return segments


def segment_stream_network(
    input_gpkg: str,
    output_gpkg: str,
    segment_interval: float,
    min_segment_length: Optional[float] = None,
    input_layer: Optional[str] = None,
    output_layer: Optional[str] = None,
    source_id_field: str = "stream_id",
) -> str:
    """
    Segment each stream feature into regular-length segments.

    Attributes are copied from the source feature. The output adds:
    - source_stream_id: original stream_id or source feature index
    - segment_id: 1-based segment number within the source feature
    - segment_uid: globally unique 1-based segment ID
    - segment_len: segment length in CRS units
    """
    streams, input_layer = _read_gpkg(input_gpkg, input_layer)

    if streams.empty:
        raise ValueError(f"No stream features found in {input_gpkg!r}.")

    if streams.crs is None:
        raise ValueError("Stream layer has no CRS. Segment length must be in CRS units.")

    if output_layer is None:
        output_layer = f"{input_layer}_segmented"

    rows = []

    for source_idx, row in streams.iterrows():
        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        source_stream_id = (
            row[source_id_field]
            if source_id_field in streams.columns
            else source_idx
        )

        source_attrs = row.drop(labels="geometry").to_dict()
        feature_segments = []

        if isinstance(geom, LineString):
            feature_segments.extend(
                _segment_linestring(
                    geom,
                    segment_interval=segment_interval,
                    min_segment_length=min_segment_length,
                )
            )

        elif isinstance(geom, MultiLineString):
            # Multipart features are segmented part-by-part, but all parts retain
            # the same source_stream_id.
            for part in geom.geoms:
                feature_segments.extend(
                    _segment_linestring(
                        part,
                        segment_interval=segment_interval,
                        min_segment_length=min_segment_length,
                    )
                )
        else:
            continue

        for seg_num, seg_geom in enumerate(feature_segments, start=1):
            out_row = dict(source_attrs)
            out_row["source_stream_id"] = source_stream_id
            out_row["segment_id"] = seg_num
            out_row["segment_len"] = float(seg_geom.length)
            out_row["geometry"] = seg_geom
            rows.append(out_row)

    if not rows:
        raise ValueError("Segmentation produced no valid stream segments.")

    segmented = gpd.GeoDataFrame(rows, geometry="geometry", crs=streams.crs)
    segmented["segment_uid"] = np.arange(1, len(segmented) + 1)
    _write_gpkg(segmented, output_gpkg, output_layer)

    return output_layer


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
    contributing cells.

    Method
    ------
    1. Verify the raster has a projected CRS.
    2. Calculate pixel area in m² using raster resolution and CRS horizontal units.
    3. Buffer each stream by a fraction of a cell.
    4. Use the 99th percentile contributing-cell count intersecting that buffer.
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
        stats=["percentile_99"],
        nodata=nodata,
        all_touched=True,
    )

    flow_cells = np.array(
        [
            s.get("percentile_99") if s.get("percentile_99") is not None else np.nan
            for s in stats
        ],
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
    """
    Keeps every nth vertex in each LineString or MultiLineString.
    """
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
    segment_interval: Optional[float] = None,
    min_segment_length: Optional[float] = None,
    add_routing: bool = True,
    d8_esri_pntr: bool = False,
    link_buffer_cells: float = 0.75,
    simplify_output_fields: bool = True,
):
    """
    Process a DEM to extract streams, save them to a GeoPackage, optionally add
    link-level routing, optionally create a thinned centerline layer, optionally
    segment the stream network, and add drainage area, PRISM climate values,
    slope, and bankfull dimensions.

    If add_routing is True, the script creates a Whitebox stream-link raster,
    builds a downstream/upstream routing table from the D8 pointer, writes that
    routing table to the output GeoPackage, and joins link-level routing fields
    to the stream vector layer.

    If segment_interval is provided, the script writes a segmented layer to the
    same GeoPackage and calculates drainage area, slope, PRISM, and bankfull
    fields on that segmented layer. If add_routing is True, it also adds
    segment-level routing fields to the segmented layer.

    If simplify_output_fields is True, the final analysis layer is pruned to a
    compact field set. Full link-routing details remain in the link-routing
    table.

    Important unit behavior
    -----------------------
    - Flow accumulation is explicitly generated as contributing cell count.
    - Drainage area is calculated as cells * raster pixel area, with raster CRS
      horizontal units converted to meters.
    - Slope is calculated from DEM samples after converting elevations to DEM
      horizontal CRS units.
    - segment_interval and min_segment_length are in stream CRS units.
    - Set dem_z_units="meter" or "foot" if DEM elevations differ from the DEM
      horizontal CRS units.
    - d8_esri_pntr should remain False unless the D8 pointer raster was created
      using the ESRI pointer convention.
    """
    wbt = whitebox.WhiteboxTools()
    _ = WbEnvironment()

    dem_meta = _get_raster_area_metadata(dem)

    os.makedirs(output_dir, exist_ok=True)

    filled_dem = os.path.join(output_dir, "filled_dem.tif")
    breached_dem = os.path.join(output_dir, "breached_dem.tif")
    d8_pointer = os.path.join(output_dir, "d8_pointer.tif")

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
        wbt.d8_pointer(src_dem, d8_pointer, esri_pntr=d8_esri_pntr)

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

    stream_links_raster = os.path.join(output_dir, f"stream_links_{threshold_label}km2.tif")

    if overwrite and os.path.exists(stream_links_raster):
        os.remove(stream_links_raster)

    if add_routing and not os.path.exists(stream_links_raster):
        print("Creating Whitebox stream-link raster for routing...")
        wbt.stream_link_identifier(
            d8_pointer,
            streams_raster,
            stream_links_raster,
            esri_pntr=d8_esri_pntr,
            zero_background=True,
        )

    streams_shp = streams_raster.replace(".tif", ".shp")
    streams_gpkg = streams_raster.replace(".tif", ".gpkg")
    streams_layer = os.path.splitext(os.path.basename(streams_gpkg))[0]

    if overwrite:
        if os.path.exists(streams_gpkg):
            os.remove(streams_gpkg)

        # Remove shapefile sidecars, if present.
        shp_base = os.path.splitext(streams_shp)[0]
        for ext in [".shp", ".shx", ".dbf", ".prj", ".cpg"]:
            sidecar = shp_base + ext
            if os.path.exists(sidecar):
                os.remove(sidecar)

    if not os.path.exists(streams_shp):
        wbt.raster_streams_to_vector(
            streams_raster,
            d8_pointer,
            streams_shp,
            esri_pntr=d8_esri_pntr,
        )

    gdf = gpd.read_file(streams_shp)

    if gdf.crs is None:
        gdf = gdf.set_crs(dem_meta["crs"])
    else:
        gdf = gdf.to_crs(dem_meta["crs"])

    _write_gpkg(gdf, streams_gpkg, streams_layer)

    streams, layer = _read_gpkg(streams_gpkg, streams_layer)
    streams["stream_id"] = range(1, len(streams) + 1)
    _write_gpkg(streams, streams_gpkg, streams_layer)

    link_routing = None

    if add_routing:
        print("Building stream-link routing table from D8 pointer...")
        link_routing = build_stream_link_routing(
            stream_links_raster=stream_links_raster,
            d8_pointer_raster=d8_pointer,
            flow_accum_cells_raster=flow_accum_cells,
            esri_pntr=d8_esri_pntr,
        )

        link_routing_layer = f"{streams_layer}_link_routing"
        _write_table_to_gpkg(
            _routing_table_for_gpkg(link_routing),
            streams_gpkg,
            link_routing_layer,
        )

        add_stream_link_routing_to_streams(
            streams_gpkg=streams_gpkg,
            stream_links_raster=stream_links_raster,
            routing=link_routing,
            layer=streams_layer,
            buffer_cells=link_buffer_cells,
        )

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

    analysis_layer = streams_layer

    if segment_interval is not None:
        print(
            f"Segmenting stream network at {segment_interval} CRS units "
            f"with minimum segment length = {min_segment_length}."
        )

        analysis_layer = f"{streams_layer}_segmented"

        segment_stream_network(
            input_gpkg=streams_gpkg,
            output_gpkg=streams_gpkg,
            segment_interval=segment_interval,
            min_segment_length=min_segment_length,
            input_layer=streams_layer,
            output_layer=analysis_layer,
            source_id_field="stream_id",
        )

    print(
        "Adding drainage area from cell-count flow accumulation "
        f"using pixel area = {dem_meta['pixel_area_m2']:.6f} m² "
        f"({dem_meta['linear_units']} CRS units)."
    )
    
    add_DA_to_stream(
        streams_gpkg=streams_gpkg,
        flow_accum_cells_raster=flow_accum_cells,
        da_field="DA_sqmi",
        da_km2_field="DA_km2",
        layer=analysis_layer,
    )

    if add_routing and segment_interval is not None:
        print("Adding segment-level routing fields to segmented stream layer...")
        segment_routing = add_segment_routing_to_streams(
            streams_gpkg=streams_gpkg,
            layer=analysis_layer,
            da_field="DA_sqmi",
        )
        _write_table_to_gpkg(
            segment_routing,
            streams_gpkg,
            f"{analysis_layer}_routing",
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
        layer=analysis_layer,
    )

    if precip_raster is not None and temp_raster is not None:
        print("Adding PRISM precipitation in inches and temperature in degrees C...")

        add_PRISM_to_streams(
            streams_gpkg_path=streams_gpkg,
            ppt_raster_path=precip_raster,
            tmean_raster_path=temp_raster,
            layer=analysis_layer,
        )
    else:
        warnings.warn(
            "PRISM precipitation and/or temperature raster not provided. Bankfull "
            "equations that use precipitation will fall back to the default value."
        )

    print("Adding bankfull dimensions...")

    add_BF_to_streams_Legg(streams_gpkg, layer=analysis_layer)
    add_BF_to_streams_Castro(streams_gpkg, layer=analysis_layer)
    add_BF_to_streams_Beechie(streams_gpkg, layer=analysis_layer)

    if simplify_output_fields:
        print("Simplifying final analysis-layer attributes...")
        simplify_stream_layer_fields(streams_gpkg, layer=analysis_layer)

    print(f"[✔] Streams extracted to: {streams_gpkg}")
    print(f"[✔] Analysis layer: {analysis_layer}")

    if add_routing:
        print(f"[✔] Link routing table: {streams_layer}_link_routing")

        if segment_interval is not None:
            print(f"[✔] Segment routing table: {analysis_layer}_routing")

    return streams_gpkg


if __name__ == "__main__":

    
    get_streams(
        dem=dem,
        output_dir=output_dir,
        threshold_km2=threshold_km2,
        overwrite=False,
        breach_depressions=True,
        create_thinned=False,
        precip_raster=r"C:\L\Lichen\Lichen - Documents\Library\GIS\PRISM\prism_ppt_30yr_avg_mmyr.tif",
        temp_raster=r"C:\L\Lichen\Lichen - Documents\Library\GIS\PRISM\prism_tmean_30yr_avg_degC.tif",
        dem_z_units="same_as_xy",
        # Optional. Values are in stream CRS units.
        # Set both to None to calculate stats on the unsegmented stream layer.
        segment_interval=segment_interval,
        min_segment_length=min_segment_length,
        simplify_output_fields=True,
    )