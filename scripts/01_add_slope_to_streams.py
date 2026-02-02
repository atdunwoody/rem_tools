from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import LineString, MultiLineString
from shapely.ops import substring

try:
    import fiona
except ImportError:
    fiona = None


# -----------------------------------------------------------------------------
# USER INPUTS
# -----------------------------------------------------------------------------
dem = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\HAWS Map\Topography\2016_USDA_DEM ndv.tif"
streams = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\HAWS Map\centerline.gpkg"

STREAMS_LAYER: Optional[str] = None  # None -> first layer
SEG_LEN = 15 * 15  # Set to 10-20x BFW
MIN_SEG_LEN = 100  # skip tiny trailing segments shorter than this (same units)

OUT_LAYER = "stream_segments_slope"  # layer name inside output gpkg


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


def _sample_dem(ds: rasterio.io.DatasetReader, x: float, y: float) -> float:
    """
    Fast point sample using raster index (nearest cell).
    Returns np.nan if outside raster or nodata.
    """
    try:
        r, c = rowcol(ds.transform, x, y)
    except Exception:
        return float("nan")

    if r < 0 or c < 0 or r >= ds.height or c >= ds.width:
        return float("nan")

    val = ds.read(1, window=((r, r + 1), (c, c + 1)))[0, 0]
    if ds.nodata is not None and np.isclose(val, ds.nodata, equal_nan=False):
        return float("nan")
    if np.isnan(val):
        return float("nan")
    return float(val)


def _segmentize_line(line: LineString, seg_len: float, min_seg_len: float) -> List[Dict[str, Any]]:
    """
    Create segments along a line using shapely.ops.substring(start_dist, end_dist).
    Distances are absolute (same units as CRS).
    """
    L = float(line.length)
    if not np.isfinite(L) or L <= 0:
        return []

    segments: List[Dict[str, Any]] = []
    d0 = 0.0
    seg_i = 0

    while d0 < L:
        d1 = min(d0 + seg_len, L)
        if (d1 - d0) < min_seg_len:
            break

        seg_geom = substring(line, d0, d1, normalized=False)
        # substring can return Point if the slice collapses; guard:
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    streams_path = Path(streams)
    out_gpkg = streams_path.with_name(streams_path.stem + "_with_slope.gpkg")

    # DEM
    with rasterio.open(dem) as ds:
        dem_crs = ds.crs
        if dem_crs is None:
            raise ValueError("DEM has no CRS. Reproject/define CRS first.")
        if dem_crs.is_geographic:
            raise ValueError(
                f"DEM CRS appears geographic (degrees): {dem_crs}. "
                "Reproject DEM to a projected CRS (meters/feet) before computing segment lengths + slope."
            )

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

        # Ensure valid geometries
        gdf = gdf[gdf.geometry.notnull()].copy()
        gdf = gdf[~gdf.geometry.is_empty].copy()

        # Project streams to DEM CRS for consistent distances + sampling
        if gdf.crs is None:
            raise ValueError("Streams layer has no CRS. Define it before running.")
        if gdf.crs != dem_crs:
            print(f"[INFO] Reprojecting streams from {gdf.crs} -> {dem_crs}")
            gdf = gdf.to_crs(dem_crs)

        # Build segmented output rows
        out_rows: List[Dict[str, Any]] = []
        total = len(gdf)

        # Choose an ID field if present, else use row index
        id_field = None
        for cand in ["id", "ID", "comid", "COMID", "fid", "FID"]:
            if cand in gdf.columns:
                id_field = cand
                break

        for i, row in gdf.iterrows():
            if (len(out_rows) % 5000) == 0 and len(out_rows) > 0:
                print(f"[PROGRESS] segments created so far: {len(out_rows):,}")

            base_atts = row.drop(labels=["geometry"]).to_dict()
            base_id = row[id_field] if id_field is not None else i

            for part in _to_lines(row.geometry):
                segs = _segmentize_line(part, SEG_LEN, MIN_SEG_LEN)
                for s in segs:
                    z0 = _sample_dem(ds, s["p_start"].x, s["p_start"].y)
                    z1 = _sample_dem(ds, s["p_end"].x, s["p_end"].y)

                    run = float(s["seg_len"])

                    # --- FIX: ensure slope is non-negative by defining "start" as higher elevation ---
                    if np.isfinite(z0) and np.isfinite(z1) and run > 0:
                        dz_signed = z1 - z0  # original direction sign (QA)
                        if z1 > z0:
                            z_hi, z_lo = z1, z0
                            swapped = True
                        else:
                            z_hi, z_lo = z0, z1
                            swapped = False

                        dz = z_hi - z_lo          # always >= 0
                        slope = dz / run          # always >= 0
                    else:
                        dz_signed = float("nan")
                        dz = float("nan")
                        slope = float("nan")
                        swapped = False
                    # -------------------------------------------------------------------------------

                    out_rows.append(
                        {
                            **base_atts,
                            "src_id": base_id,
                            "seg_index": s["seg_index"],
                            "seg_start": float(s["seg_start"]),
                            "seg_end": float(s["seg_end"]),
                            "seg_len": float(s["seg_len"]),
                            "z_start": z0,
                            "z_end": z1,
                            "dz_signed": dz_signed,  # may be negative depending on line direction
                            "dz": dz,                # non-negative magnitude
                            "slope_m_m": slope,      # non-negative
                            "slope_pct": slope * 100.0 if np.isfinite(slope) else float("nan"),
                            "slope_deg": math.degrees(math.atan(slope)) if np.isfinite(slope) else float("nan"),
                            "dir_swapped_hi_to_lo": swapped,
                            "geometry": s["geometry"],
                        }
                    )

        seg_gdf = gpd.GeoDataFrame(out_rows, geometry="geometry", crs=dem_crs)
        print(f"[DONE] Created {len(seg_gdf):,} segments from {total:,} input features.")

        # Write output
        if out_gpkg.exists():
            out_gpkg.unlink()  # replace
        seg_gdf.to_file(out_gpkg, layer=OUT_LAYER, driver="GPKG")
        print(f"[WROTE] {out_gpkg} (layer='{OUT_LAYER}')")


if __name__ == "__main__":
    main()
