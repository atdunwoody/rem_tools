"""Create river-mile station points along a stream centerline.

Nearby centerline segment endpoints are snapped together before the route is
merged. The supplied start point is then snapped to the nearest endpoint of the
merged centerline. Stations begin at RM 0.0 and continue at the requested
interval.

Examples
--------
Run with the project-specific defaults defined below:

    python create_river_mile_stationing.py

Use other inputs:

    python create_river_mile_stationing.py streams.gpkg start.gpkg -o stations.gpkg

Select a route from a centerline layer containing several streams:

    python create_river_mile_stationing.py streams.gpkg start.gpkg \
        --where "GNIS_NAME = 'Grande Ronde River'"
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from pyproj import CRS
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union


DEFAULT_STREAMS = Path(
    r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\CFC Silver Creek\Field Data\LiDAR\Silver Creek Centerline.gpkg"
)
DEFAULT_START_POINT = Path(
    r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\CFC Silver Creek\Field Data\LiDAR\Streams\confluence point.gpkg"
)
DEFAULT_OUTPUT = DEFAULT_STREAMS.with_name("river_mile_stations.gpkg")

METERS_PER_MILE = 1609.344
OUTPUT_LAYER = "river_mile_stations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create regularly spaced river-mile stations from a point and a "
            "stream centerline. Nearby segment endpoints are snapped together "
            "before stationing."
        )
    )
    parser.add_argument(
        "streams",
        nargs="?",
        type=Path,
        default=DEFAULT_STREAMS,
        help=f"Stream centerline dataset (default: {DEFAULT_STREAMS})",
    )
    parser.add_argument(
        "start_point",
        nargs="?",
        type=Path,
        default=DEFAULT_START_POINT,
        help=f"Start-point dataset (default: {DEFAULT_START_POINT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=f"Output GeoPackage (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--stream-layer",
        help="Stream layer name. If omitted, GeoPandas reads the first layer.",
    )
    parser.add_argument(
        "--point-layer",
        help="Start-point layer name. If omitted, GeoPandas reads the first layer.",
    )
    parser.add_argument(
        "--where",
        help=(
            "Optional SQL-style attribute filter applied to the stream layer, "
            "for example: GNIS_NAME = 'Grande Ronde River'."
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Station spacing in miles (default: 0.1).",
    )
    parser.add_argument(
        "--label-interval",
        type=float,
        default=0.5,
        help="Interval in miles at which the label field is populated (default: 0.5).",
    )
    parser.add_argument(
        "--snap-tolerance",
        type=float,
        default=10.0,
        help=(
            "Maximum endpoint-to-endpoint gap to snap, in meters "
            "(default: 10.0)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output GeoPackage if it already exists.",
    )
    return parser.parse_args()


def read_inputs(args: argparse.Namespace) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    stream_kwargs: dict[str, object] = {}
    point_kwargs: dict[str, object] = {}
    if args.stream_layer:
        stream_kwargs["layer"] = args.stream_layer
    if args.point_layer:
        point_kwargs["layer"] = args.point_layer
    if args.where:
        stream_kwargs["where"] = args.where

    streams = gpd.read_file(args.streams, **stream_kwargs)
    start = gpd.read_file(args.start_point, **point_kwargs)
    return streams, start


def validate_inputs(streams: gpd.GeoDataFrame, start: gpd.GeoDataFrame) -> None:
    if streams.empty:
        raise ValueError("The stream selection contains no features.")
    if start.empty:
        raise ValueError("The start-point dataset contains no features.")
    if streams.crs is None:
        raise ValueError("The stream centerline has no defined CRS.")
    if start.crs is None:
        raise ValueError("The start point has no defined CRS.")
    if len(start) != 1:
        raise ValueError(
            f"The start-point dataset must contain exactly one feature; found {len(start)}."
        )
    if start.geometry.iloc[0] is None or start.geometry.iloc[0].is_empty:
        raise ValueError("The start-point geometry is empty.")
    if start.geometry.iloc[0].geom_type != "Point":
        raise TypeError(
            "The start-point feature must be a Point; found "
            f"{start.geometry.iloc[0].geom_type}."
        )

    allowed = {"LineString", "MultiLineString"}
    invalid = sorted(set(streams.geometry.geom_type.dropna()) - allowed)
    if invalid:
        raise TypeError(
            "The stream dataset may contain only line geometries; found "
            + ", ".join(invalid)
            + "."
        )


def local_utm_crs(gdf: gpd.GeoDataFrame) -> CRS:
    """Return a local UTM CRS for a layer whose CRS is geographic."""
    centroid = unary_union(
        [
            geometry
            for geometry in gdf.to_crs(4326).geometry
            if geometry is not None and not geometry.is_empty
        ]
    ).centroid
    zone = int(math.floor((centroid.x + 180.0) / 6.0) + 1)
    zone = min(max(zone, 1), 60)
    epsg = 32600 + zone if centroid.y >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def working_crs(streams: gpd.GeoDataFrame) -> CRS:
    """Return a projected CRS suitable for measuring centerline distance."""
    crs = CRS.from_user_input(streams.crs)
    if crs.is_geographic:
        return local_utm_crs(streams)
    if not crs.is_projected:
        raise ValueError(
            "The stream CRS is neither projected nor geographic and cannot be used "
            "for distance measurement."
        )
    return crs


def line_parts(geometry: LineString | MultiLineString) -> list[LineString]:
    """Return the LineString parts of a line geometry."""
    if isinstance(geometry, LineString):
        return [geometry]
    if isinstance(geometry, MultiLineString):
        return list(geometry.geoms)
    raise TypeError(f"Expected line geometry; found {geometry.geom_type}.")


def closest_endpoint_pair(
    first: LineString, second: LineString
) -> tuple[float, int, int]:
    """Return distance and endpoint indices for the closest endpoint pair."""
    first_endpoints = (Point(first.coords[0]), Point(first.coords[-1]))
    second_endpoints = (Point(second.coords[0]), Point(second.coords[-1]))

    candidates = [
        (a.distance(b), first_index, second_index)
        for first_index, a in enumerate(first_endpoints)
        for second_index, b in enumerate(second_endpoints)
    ]
    return min(candidates, key=lambda item: item[0])


def stitch_lines(
    first: LineString,
    second: LineString,
    first_endpoint: int,
    second_endpoint: int,
) -> LineString:
    """Snap two selected endpoints to their midpoint and concatenate the lines."""
    first_coords = [tuple(coord[:2]) for coord in first.coords]
    second_coords = [tuple(coord[:2]) for coord in second.coords]

    # Orient the first line toward the join and the second line away from it.
    if first_endpoint == 0:
        first_coords.reverse()
    if second_endpoint == 1:
        second_coords.reverse()

    first_join = first_coords[-1]
    second_join = second_coords[0]
    junction = (
        (first_join[0] + second_join[0]) / 2.0,
        (first_join[1] + second_join[1]) / 2.0,
    )

    stitched_coords = first_coords[:-1] + [junction] + second_coords[1:]
    return LineString(stitched_coords)


def minimum_gap_between_parts(parts: list[LineString]) -> float:
    """Return the smallest endpoint-to-endpoint gap between separate parts."""
    gaps = [
        closest_endpoint_pair(parts[i], parts[j])[0]
        for i in range(len(parts))
        for j in range(i + 1, len(parts))
    ]
    return min(gaps) if gaps else 0.0


def merge_centerline(
    streams: gpd.GeoDataFrame,
    snap_tolerance_units: float,
    meters_per_unit: float,
) -> tuple[LineString, list[float]]:
    """Dissolve and snap line parts into one continuous, nonbranching route."""
    valid_geometries = [
        geometry
        for geometry in streams.geometry
        if geometry is not None and not geometry.is_empty
    ]
    if not valid_geometries:
        raise ValueError("The stream dataset contains no nonempty geometries.")

    dissolved = unary_union(valid_geometries)
    merged = dissolved if isinstance(dissolved, LineString) else linemerge(dissolved)
    if isinstance(merged, LineString):
        return merged, []

    if isinstance(merged, MultiLineString):
        parts = line_parts(merged)
        join_gaps: list[float] = []

        # Repeatedly join the closest pair of route endpoints. Restricting joins
        # to endpoints avoids snapping a tributary endpoint to the middle of a
        # mainstem and silently creating a branched route.
        while len(parts) > 1:
            best: tuple[float, int, int, int, int] | None = None
            for first_index in range(len(parts)):
                for second_index in range(first_index + 1, len(parts)):
                    gap, first_endpoint, second_endpoint = closest_endpoint_pair(
                        parts[first_index], parts[second_index]
                    )
                    candidate = (
                        gap,
                        first_index,
                        second_index,
                        first_endpoint,
                        second_endpoint,
                    )
                    if best is None or candidate < best:
                        best = candidate

            if best is None or best[0] > snap_tolerance_units:
                break

            gap, first_index, second_index, first_endpoint, second_endpoint = best
            stitched = stitch_lines(
                parts[first_index],
                parts[second_index],
                first_endpoint,
                second_endpoint,
            )
            join_gaps.append(gap)

            # Remove the higher index first, then replace the lower-index line.
            parts.pop(second_index)
            parts[first_index] = stitched

        if len(parts) == 1:
            return parts[0], join_gaps

        nearest_gap_m = minimum_gap_between_parts(parts) * meters_per_unit
        raise ValueError(
            "The selected stream features could not be converted to one continuous, "
            f"nonbranching route. {len(parts)} line parts remain after snapping "
            f"endpoints within {snap_tolerance_units * meters_per_unit:.2f} m; the "
            f"nearest remaining endpoint gap is {nearest_gap_m:.2f} m. Increase "
            "--snap-tolerance if the parts belong to the same route. If the input "
            "contains branches, select one mainstem with --where."
        )

    raise TypeError(f"Merging produced an unsupported geometry: {merged.geom_type}.")


def orient_from_nearest_endpoint(
    route: LineString, supplied_start: Point
) -> tuple[LineString, Point, float]:
    """Orient a route away from the endpoint closest to the supplied point."""
    first = Point(route.coords[0])
    last = Point(route.coords[-1])

    if supplied_start.distance(first) <= supplied_start.distance(last):
        snapped_start = first
        oriented_route = route
    else:
        snapped_start = last
        oriented_route = LineString(list(route.coords)[::-1])

    return oriented_route, snapped_start, supplied_start.distance(snapped_start)


def distance_units_to_meters(crs: CRS) -> float:
    """Return the number of meters represented by one horizontal CRS unit."""
    if not crs.axis_info:
        raise ValueError("The working CRS does not report linear units.")
    factor = crs.axis_info[0].unit_conversion_factor
    if factor is None or factor <= 0:
        raise ValueError("Could not determine the working CRS linear-unit conversion.")
    return float(factor)


def build_stations(
    route: LineString,
    output_crs: CRS,
    interval_miles: float,
    label_interval_miles: float,
) -> gpd.GeoDataFrame:
    if interval_miles <= 0:
        raise ValueError("--interval must be greater than zero.")
    if label_interval_miles <= 0:
        raise ValueError("--label-interval must be greater than zero.")

    label_every = label_interval_miles / interval_miles
    label_every_int = round(label_every)
    if not math.isclose(label_every, label_every_int, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "--label-interval must be an integer multiple of --interval."
        )

    meters_per_unit = distance_units_to_meters(output_crs)
    route_length_miles = route.length * meters_per_unit / METERS_PER_MILE
    station_count = math.floor((route_length_miles + 1e-10) / interval_miles) + 1

    river_miles: list[float] = []
    labels: list[str | None] = []
    points: list[Point] = []

    for station_index in range(station_count):
        river_mile = station_index * interval_miles
        distance_units = river_mile * METERS_PER_MILE / meters_per_unit
        points.append(route.interpolate(distance_units))
        river_miles.append(round(river_mile, 10))
        labels.append(
            f"RM {river_mile:.1f}"
            if station_index % label_every_int == 0
            else None
        )

    return gpd.GeoDataFrame(
        {
            "station_id": range(station_count),
            "river_mile": river_miles,
            # pandas StringDtype ensures this is written as a text field even
            # when some or all values are null.
            "label": pd.Series(labels, dtype="string"),
        },
        geometry=points,
        crs=output_crs,
    )


def prepare_output(output: Path, args: argparse.Namespace) -> None:
    output = output.resolve()
    protected = {args.streams.resolve(), args.start_point.resolve()}
    if output in protected:
        raise ValueError("The output path cannot be the same as either input path.")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output}\n"
                "Use --overwrite to replace it or provide a different -o path."
            )
        output.unlink()


def main() -> None:
    args = parse_args()
    output = args.output or args.streams.with_name(DEFAULT_OUTPUT.name)

    streams, start = read_inputs(args)
    validate_inputs(streams, start)

    original_crs = CRS.from_user_input(streams.crs)
    measure_crs = working_crs(streams)
    streams_working = streams.to_crs(measure_crs)
    start_working = start.to_crs(measure_crs)

    if args.snap_tolerance < 0:
        raise ValueError("--snap-tolerance must be zero or greater.")

    meters_per_unit = distance_units_to_meters(measure_crs)
    snap_tolerance_units = args.snap_tolerance / meters_per_unit
    route, segment_join_gaps = merge_centerline(
        streams_working,
        snap_tolerance_units=snap_tolerance_units,
        meters_per_unit=meters_per_unit,
    )
    supplied_start = start_working.geometry.iloc[0]
    route, snapped_start, snap_distance_units = orient_from_nearest_endpoint(
        route, supplied_start
    )

    stations = build_stations(
        route=route,
        output_crs=measure_crs,
        interval_miles=args.interval,
        label_interval_miles=args.label_interval,
    )
    stations = stations.to_crs(original_crs)

    prepare_output(output, args)
    stations.to_file(output, layer=OUTPUT_LAYER, driver="GPKG", index=False)

    snap_distance_m = snap_distance_units * meters_per_unit
    route_length_miles = route.length * meters_per_unit / METERS_PER_MILE
    snapped_start_output = gpd.GeoSeries([snapped_start], crs=measure_crs).to_crs(
        original_crs
    ).iloc[0]

    print(f"Created {len(stations):,} stations: {output}")
    print(f"Layer: {OUTPUT_LAYER}")
    print(f"Route length: {route_length_miles:.2f} miles")
    if segment_join_gaps:
        maximum_gap_m = max(segment_join_gaps) * meters_per_unit
        print(
            f"Snapped {len(segment_join_gaps)} segment gap(s); "
            f"maximum original gap: {maximum_gap_m:.2f} m"
        )
    else:
        print("No segment endpoint gaps required snapping")
    print(f"Start point snapped {snap_distance_m:.2f} m to route endpoint")
    print(
        "RM 0 coordinate in output CRS: "
        f"{snapped_start_output.x:.3f}, {snapped_start_output.y:.3f}"
    )


if __name__ == "__main__":
    main()
