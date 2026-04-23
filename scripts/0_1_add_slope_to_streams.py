from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import geopandas as gpd
import rasterio
from shapely.geometry import LineString, MultiLineString
from shapely.ops import substring
from rasterstats import zonal_stats

try:
    import fiona
except ImportError:
    fiona = None


# -----------------------------------------------------------------------------
# USER INPUTS
# -----------------------------------------------------------------------------
dem = r"C:\L\Lichen\Lichen - Documents\Projects\20240001.4_Tucan 5-15 (CTUIR)\07_GIS\Wenaha\DEMs\USGS10m_EPSG2927.tif"
streams = r"C:\L\Lichen\Lichen - Documents\Projects\20240001.4_Tucan 5-15 (CTUIR)\07_GIS\Wenaha\USDA HydroFlow Metrics\USDA_HydroFlowMet_1997-2006.gpkg"

STREAMS_LAYER: Optional[str] = None  # None -> first layer
SEG_LEN = None  # Set to 10-20x BFW. Use None to leave input lines as-is.
MIN_SEG_LEN = 100  # skip tiny trailing segments shorter than this (same units)
DEM_VERTICAL_UNITS = "meters"  # "feet" or "meters"

OUT_LAYER = "USDA_flowmet_with_slope"  # layer name inside output gpkg


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _list_layers(gpkg_path: str) -> List[str]:
    if fiona is None:
        raise RuntimeError(
            "fiona is required to list layers. Install/import fiona or set STREAMS_LAYER explicitly."
        )
    return list(fiona.listlayers(gpkg_path))


def _to_lines(geom) -> List[LineString]:
    """Explode LineString/MultiLineString to a list of LineStrings."""
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString):
        return [ls for ls in geom.geoms if ls is not None and not ls.is_empty]
    return []


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


def _segmentize_line(
    line: LineString,
    seg_len: Optional[float],
    min_seg_len: float,
) -> List[Dict[str, Any]]:
    """
    Create segments along a line using shapely.ops.substring(start_dist, end_dist).

    If seg_len is None, the input line is returned as a single segment.
    Distances are absolute (same units as CRS).
    """
    L = float(line.length)
    if not np.isfinite(L) or L <= 0:
        return []

    if seg_len is None:
        p_start = line.interpolate(0.0)
        p_end = line.interpolate(L)
        return [
            {
                "seg_index": 0,
                "seg_start": 0.0,
                "seg_end": L,
                "seg_len": L,
                "p_start": p_start,
                "p_end": p_end,
                "geometry": line,
            }
        ]

    segments: List[Dict[str, Any]] = []
    d0 = 0.0
    seg_i = 0

    while d0 < L:
        d1 = min(d0 + seg_len, L)
        if (d1 - d0) < min_seg_len:
            break

        seg_geom = substring(line, d0, d1, normalized=False)
        if not isinstance(seg_geom, LineString) or seg_geom.is_empty or seg_geom.length <= 0:
            d0 = d1
            seg_i += 1
            continue

        p_start = line.interpolate(d0)
        p_end = line.interpolate(d1)

        segments.append(
            {
                "seg_index": seg_i,
                "seg_start": d0,
                "seg_end": d1,
                "seg_len": float(seg_geom.length),
                "p_start": p_start,
                "p_end": p_end,
                "geometry": seg_geom,
            }
        )

        d0 = d1
        seg_i += 1

    return segments


def _get_dem_metadata(dem_path: str) -> Dict[str, Any]:
    with rasterio.open(dem_path) as ds:
        crs = ds.crs
        xres, yres = ds.res
        nodata = ds.nodata

    if crs is None:
        raise ValueError(f"DEM has no CRS: {dem_path}")

    if not crs.is_projected:
        raise ValueError(
            f"DEM CRS is not projected: {crs}. Reproject DEM before computing length-based slope."
        )

    try:
        linear_units = crs.linear_units.lower() if crs.linear_units else None
    except Exception:
        linear_units = None

    return {
        "crs": crs,
        "pixel_width": abs(xres),
        "pixel_height": abs(yres),
        "nodata": nodata,
        "linear_units": linear_units,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    streams_path = Path(streams)
    out_gpkg = streams_path.with_name(streams_path.stem + "_with_slope.gpkg")

    dem_meta = _get_dem_metadata(dem)
    dem_crs = dem_meta["crs"]
    dem_horizontal_units = _normalize_linear_unit(dem_meta["linear_units"])
    dem_vertical_units = _normalize_linear_unit(DEM_VERTICAL_UNITS)

    if dem_horizontal_units not in {"feet", "meters"}:
        raise ValueError(
            f"Unsupported DEM horizontal units: {dem_meta['linear_units']}"
        )
    if dem_vertical_units not in {"feet", "meters"}:
        raise ValueError("DEM_VERTICAL_UNITS must be 'feet' or 'meters'")

    with rasterio.open(dem) as ds:
        # Streams layer selection
        layer = STREAMS_LAYER
        if layer is None:
            layers = _list_layers(streams)
            if not layers:
                raise ValueError(f"No layers found in: {streams}")
            layer = layers[0]
            print(f"[INFO] STREAMS_LAYER not set. Using first layer: {layer}")

        gdf = gpd.read_file(streams, layer=layer)
        if gdf.empty:
            raise ValueError(f"Streams layer is empty: {layer}")

        gdf = gdf[gdf.geometry.notnull()].copy()
        gdf = gdf[~gdf.geometry.is_empty].copy()

        if gdf.crs is None:
            raise ValueError("Streams layer has no CRS. Define it before running.")
        if gdf.crs != dem_crs:
            print(f"[INFO] Reprojecting streams from {gdf.crs} -> {dem_crs}")
            gdf = gdf.to_crs(dem_crs)

        out_rows: List[Dict[str, Any]] = []
        total = len(gdf)

        id_field = None
        for cand in ["id", "ID", "comid", "COMID", "fid", "FID"]:
            if cand in gdf.columns:
                id_field = cand
                break

        buffer_dist = max(dem_meta["pixel_width"], dem_meta["pixel_height"]) * 0.75

        for i, row in gdf.iterrows():
            if (len(out_rows) % 5000) == 0 and len(out_rows) > 0:
                print(f"[PROGRESS] segments created so far: {len(out_rows):,}")

            base_atts = row.drop(labels=["geometry"]).to_dict()
            base_id = row[id_field] if id_field is not None else i

            for part in _to_lines(row.geometry):
                segs = _segmentize_line(part, SEG_LEN, MIN_SEG_LEN)

                if not segs:
                    continue

                sample_geoms = [s["geometry"].buffer(buffer_dist) for s in segs]
                stats = zonal_stats(
                    sample_geoms,
                    dem,
                    stats=["min", "max"],
                    nodata=dem_meta["nodata"],
                    all_touched=True,
                )

                for s, st in zip(segs, stats):
                    elev_min = st.get("min")
                    elev_max = st.get("max")
                    run = float(s["seg_len"])

                    if elev_min is not None and elev_max is not None and np.isfinite(run) and run > 0:
                        dz_native = float(elev_max) - float(elev_min)
                        dz_horiz_units = _convert_length_units(
                            dz_native,
                            dem_vertical_units,
                            dem_horizontal_units,
                        )
                        slope = dz_horiz_units / run
                    else:
                        dz_native = float("nan")
                        dz_horiz_units = float("nan")
                        slope = float("nan")

                    out_rows.append(
                        {
                            **base_atts,
                            "src_id": base_id,
                            "seg_index": s["seg_index"],
                            "seg_start": float(s["seg_start"]),
                            "seg_end": float(s["seg_end"]),
                            "seg_len": run,
                            "elev_min": float(elev_min) if elev_min is not None else float("nan"),
                            "elev_max": float(elev_max) if elev_max is not None else float("nan"),
                            "elev_min_ft": (
                                _convert_length_units(float(elev_min), dem_vertical_units, "feet")
                                if elev_min is not None else float("nan")
                            ),
                            "elev_max_ft": (
                                _convert_length_units(float(elev_max), dem_vertical_units, "feet")
                                if elev_max is not None else float("nan")
                            ),
                            "dz": dz_native,
                            "slope": slope,
                            "slope_pct": slope * 100.0 if np.isfinite(slope) else float("nan"),
                            "slope_deg": math.degrees(math.atan(slope)) if np.isfinite(slope) else float("nan"),
                            "geometry": s["geometry"],
                        }
                    )

        seg_gdf = gpd.GeoDataFrame(out_rows, geometry="geometry", crs=dem_crs)
        print(f"[DONE] Created {len(seg_gdf):,} output features from {total:,} input features.")

        if out_gpkg.exists():
            out_gpkg.unlink()
        seg_gdf.to_file(out_gpkg, layer=OUT_LAYER, driver="GPKG")
        print(f"[WROTE] {out_gpkg} (layer='{OUT_LAYER}')")


if __name__ == "__main__":
    main()