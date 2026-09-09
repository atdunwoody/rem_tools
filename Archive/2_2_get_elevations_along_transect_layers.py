"""
Extract minimum-elevation or median-elevation points and endpoints for transect lines.

This version assumes the input transect GeoPackage contains one layer per stream_id,
as written by the previous transect-generation script.

For each transect layer:
  - samples the DEM along each transect at raster-resolution spacing
  - finds the minimum or median elevation point
  - optionally adds start and end points
  - writes the results to a new output layer with the same layer name

Output:
  One GeoPackage layer per input transect layer / stream_id.
"""

import os
import re
from typing import Optional

import fiona
import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point, LineString
from shapely.ops import linemerge


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _as_single_line(geom):
    """
    Return a LineString suitable for .coords and .interpolate().

    - If geom is LineString: return it.
    - If geom is MultiLineString:
        * try linemerge, which may produce a LineString.
        * if still multipart, pick the longest LineString component.
    Returns None if a usable line cannot be produced.
    """
    if geom is None or geom.is_empty:
        return None

    if geom.geom_type == "LineString":
        return geom

    if geom.geom_type == "MultiLineString":
        merged = linemerge(geom)

        if merged.geom_type == "LineString":
            return merged

        if merged.geom_type == "MultiLineString":
            parts = list(merged.geoms)
            if not parts:
                return None
            return max(parts, key=lambda g: g.length)

    return None


def _safe_endpoints(line: LineString):
    """
    Get start and end points from a LineString.
    """
    coords = list(line.coords)

    if len(coords) < 2:
        return None, None

    return Point(coords[0]), Point(coords[-1])


def _sanitize_layer_name(value, prefix: str = "stream") -> str:
    """
    Convert a value to a safe GeoPackage layer name.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        value = "unknown"

    name = str(value).strip()
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = name.strip("_")

    if not name:
        name = "unknown"

    if name[0].isdigit():
        name = f"{prefix}_{name}"

    return name


def _get_input_layers(gpkg_path: str) -> list[str]:
    """
    Return all layer names from a GeoPackage.
    """
    if not os.path.exists(gpkg_path):
        raise FileNotFoundError(f"Input GeoPackage not found: {gpkg_path}")

    layers = list(fiona.listlayers(gpkg_path))

    if not layers:
        raise ValueError(f"No layers found in input GeoPackage: {gpkg_path}")

    return layers


def _base_attrs(row, geometry_field: str = "geometry") -> dict:
    """
    Copy non-geometry attributes from an input row.
    """
    attrs = {}

    for col, value in row.items():
        if col == geometry_field:
            continue

        # Avoid storing unsupported object-like values.
        if isinstance(value, (list, dict, tuple, set)):
            continue

        attrs[col] = value

    return attrs


def _valid_positive_elev(value: float) -> Optional[float]:
    """
    Return elevation if valid and positive, otherwise None.

    This keeps the behavior from the original script, which filtered null
    and non-positive elevations.
    """
    if value is None:
        return None

    if np.isnan(value):
        return None

    if value <= 0:
        return None

    return float(value)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def extract_elevations_along_transect_layers(
    transect_gpkg: str,
    dem_path: str,
    output_gpkg: str,
    method: str = "min",
    flank_points: bool = False,
    overwrite: bool = True,
) -> str:
    """
    Process every layer in a transect GeoPackage and write one output point
    layer per input layer.

    Parameters
    ----------
    transect_gpkg : str
        Input GeoPackage containing transect layers. Each layer should correspond
        to one stream_id.
    dem_path : str
        DEM raster used to sample elevations.
    output_gpkg : str
        Output GeoPackage for elevation points.
    method : str
        "min" for minimum-elevation points, or "median" for median-elevation points.
    flank_points : bool
        If True, also write start and end points for each transect.
    overwrite : bool
        If True, removes the output GeoPackage before writing.

    Returns
    -------
    str
        Output GeoPackage path.
    """
    method = method.lower().strip()

    if method not in {"min", "median"}:
        raise ValueError(f"Unknown method: {method}. Valid options are 'min' or 'median'.")

    output_dir = os.path.dirname(output_gpkg)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if overwrite and os.path.exists(output_gpkg):
        os.remove(output_gpkg)

    layer_names = _get_input_layers(transect_gpkg)

    layers_written = 0
    total_points = 0

    print(f"Found {len(layer_names)} transect layers in: {transect_gpkg}")

    with rasterio.open(dem_path) as src:
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)
        sample_dist = min(res_x, res_y)
        no_data_value = src.nodata

        for layer_name in layer_names:
            print(f"\nProcessing layer: {layer_name}")

            gdf_lines = gpd.read_file(transect_gpkg, layer=layer_name)

            if gdf_lines.empty:
                print(f"Skipping layer '{layer_name}': no features.")
                continue

            crs = gdf_lines.crs
            points = []

            print(f"Creating points for {len(gdf_lines)} transects...")

            for idx, row in gdf_lines.iterrows():
                raw_geom = row.geometry
                line = _as_single_line(raw_geom)

                if line is None or line.is_empty:
                    print(
                        f"Skipping row {idx} in layer '{layer_name}': "
                        f"empty/invalid/unsupported geometry "
                        f"({getattr(raw_geom, 'geom_type', None)})"
                    )
                    continue

                length = line.length

                if length <= 0:
                    print(f"Skipping row {idx} in layer '{layer_name}': zero-length line.")
                    continue

                n_samples = max(int(length / sample_dist) + 1, 2)
                distances = np.linspace(0, length, n_samples)
                sample_pts = [line.interpolate(d) for d in distances]
                coords = [(pt.x, pt.y) for pt in sample_pts]

                values = np.array([val[0] for val in src.sample(coords)], dtype=float)

                if no_data_value is not None and not np.isnan(no_data_value):
                    values = np.ma.masked_equal(values, no_data_value)

                if np.ma.is_masked(values) and values.mask.all():
                    print(f"Skipping row {idx} in layer '{layer_name}': all sampled DEM values are nodata.")
                    continue

                attrs = _base_attrs(row)
                attrs["source_layer"] = layer_name

                # Use stream_id from the transect attributes if present.
                # If absent, infer it from the layer name.
                if "stream_id" not in attrs:
                    attrs["stream_id"] = layer_name

                if method == "min":
                    point_idx = int(values.argmin())
                    point_val = float(values[point_idx])
                    point_type = "min"
                else:
                    point_val = float(np.ma.median(values))
                    filled = np.ma.filled(values, np.nan)
                    point_idx = int(np.nanargmin(np.abs(filled - point_val)))
                    point_type = "median"

                elev = _valid_positive_elev(point_val)

                if elev is None:
                    continue

                main_pt = Point(coords[point_idx])

                main_attrs = attrs.copy()
                main_attrs.update(
                    {
                        "geometry": main_pt,
                        "elevation": elev,
                        "point_type": point_type,
                    }
                )
                points.append(main_attrs)

                if flank_points:
                    start_pt, end_pt = _safe_endpoints(line)

                    if start_pt is None or end_pt is None:
                        print(f"Skipping endpoints for row {idx} in layer '{layer_name}': could not determine endpoints.")
                        continue

                    for flank_type, flank_geom in [("start", start_pt), ("end", end_pt)]:
                        flank_attrs = attrs.copy()
                        flank_attrs.update(
                            {
                                "geometry": flank_geom,
                                "elevation": elev,
                                "point_type": flank_type,
                            }
                        )
                        points.append(flank_attrs)

            if not points:
                print(f"Skipping output for layer '{layer_name}': no valid points created.")
                continue

            gdf_pts = gpd.GeoDataFrame(points, crs=crs)

            if "elevation" in gdf_pts.columns:
                gdf_pts = gdf_pts.dropna(subset=["elevation"])
                gdf_pts = gdf_pts[gdf_pts["elevation"] > 0]

            if gdf_pts.empty:
                print(f"Skipping output for layer '{layer_name}': all points removed by elevation filter.")
                continue

            out_layer = _sanitize_layer_name(layer_name)

            gdf_pts.to_file(
                output_gpkg,
                driver="GPKG",
                layer=out_layer,
            )

            layers_written += 1
            total_points += len(gdf_pts)

            print(
                f"[✔] Wrote {len(gdf_pts)} points to "
                f"'{output_gpkg}' layer='{out_layer}'"
            )

    print(
        f"\n[✔] Finished writing {layers_written} layers and "
        f"{total_points} points to: {output_gpkg}"
    )

    return output_gpkg


# -----------------------------------------------------------------------------
# Backward-compatible wrapper
# -----------------------------------------------------------------------------

def extract_elevations_along_transect(
    transect_gpkg: str,
    dem_path: str,
    output_gpkg: str,
    flank_min_points: bool = False,
    method: str = "min",
):
    """
    Backward-compatible wrapper.

    This now processes all layers in the input transect GeoPackage and writes
    one output layer per input layer.
    """
    return extract_elevations_along_transect_layers(
        transect_gpkg=transect_gpkg,
        dem_path=dem_path,
        output_gpkg=output_gpkg,
        method=method,
        flank_points=flank_min_points,
        overwrite=True,
    )


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    default_transect_gpkg = (
        r"C:\L\Lichen\Lichen - Documents\Projects\20260003_Owens-Snipe Assessment "
        r"(UCSWCD)\07_GIS\1_Analysis\Stream Network Analysis\REM\transects.gpkg"
    )

    default_dem_path = (
        r"C:\L\Lichen\Lichen - Documents\Projects\20260003_Owens-Snipe Assessment "
        r"(UCSWCD)\07_GIS\0_Data_In\Public\LiDAR"
        r"\USGS3ft_proj_2020-2021_merged.tif"
    )

    default_output_gpkg = os.path.join(
        os.path.dirname(default_transect_gpkg),
        "min_elev_points_by_stream_id.gpkg",
    )

    extract_elevations_along_transect(
        transect_gpkg=default_transect_gpkg,
        dem_path=default_dem_path,
        output_gpkg=default_output_gpkg,
        method="min",              # "min" for HAWS or "median" for GGL
        flank_min_points=True,     # include endpoints too
    )