from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import CRS, Transformer
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import substring


@dataclass
class SegmentRecord:
    parent_fid: int
    part_idx: int
    seg_idx: int
    from_m: float
    to_m: float
    length_m: float
    z_start: Optional[float]
    z_end: Optional[float]
    slope: Optional[float]       # rise/run (m/m), positive = drop downstream
    slope_pct: Optional[float]   # slope * 100
    geometry: LineString


def _validate_paths(gpkg: Path, dem: Path) -> None:
    if not gpkg.exists() or not gpkg.is_file():
        raise FileNotFoundError(f"GeoPackage not found: {gpkg}")
    if not dem.exists() or not dem.is_file():
        raise FileNotFoundError(f"DEM not found: {dem}")


def _ensure_projected_crs(src_crs: CRS, gdf: gpd.GeoDataFrame) -> CRS:
    """
    Return a projected CRS in meters for length/segment ops.
    - If source CRS is already projected in meters, use it.
    - If geographic, estimate a suitable UTM.
    """
    if src_crs is None:
        raise ValueError("Input layer has no CRS. Please define a CRS before running.")
    if src_crs.is_projected and src_crs.axis_info[0].unit_name.lower() in {"metre", "meter", "metres", "meters"}:
        return src_crs
    # Estimate UTM based on geometry centroid
    utm = CRS.estimate_utm_crs(gdf.geometry.unary_union.centroid)
    if utm is None:
        raise ValueError("Failed to estimate a projected CRS for distance operations.")
    return utm


def _iter_lines(geom) -> Iterable[Tuple[int, LineString]]:
    """Yield (part_idx, LineString) for LineString or MultiLineString."""
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield 0, geom
    elif isinstance(geom, MultiLineString):
        for i, part in enumerate(geom.geoms):
            if not part.is_empty:
                yield i, part
    else:
        # GeometryCollections etc.: extract line parts
        for i, part in enumerate(getattr(geom, "geoms", [])):
            if isinstance(part, (LineString, MultiLineString)):
                for j, sub in _iter_lines(part):
                    yield i * 100000 + j, sub  # unique but stable index


def _split_to_interval_parts(ls: LineString, interval_m: float) -> List[Tuple[float, float, LineString]]:
    """
    Split a LineString into consecutive substrings of length 'interval_m',
    returning a list of (from_m, to_m, geometry). The final piece may be shorter.
    """
    L = ls.length
    if L == 0:
        return []
    cuts = np.arange(0.0, L, interval_m, dtype=float)
    if cuts[-1] != 0.0:
        # ensure start at 0
        cuts[0] = 0.0
    # Build ranges [s, e] with last endpoint = L
    ranges = [(float(s), float(min(s + interval_m, L))) for s in cuts]
    # Protect against numerical duplicates
    ranges = [(s, e) for s, e in ranges if e > s]
    parts = []
    for s, e in ranges:
        sub = substring(ls, s, e, normalized=False)
        # substring may return Point for zero-length; filter out
        if isinstance(sub, LineString) and sub.length > 0:
            parts.append((s, e, sub))
    return parts


def _sample_dem_points(
    xs: List[float], ys: List[float], src_crs: CRS, dem_dataset: rasterio.io.DatasetReader
) -> List[Optional[float]]:
    """
    Sample DEM elevations at (x,y) in src_crs. Reprojects to DEM CRS if needed.
    Returns list of z (float or None on nodata/outside).
    """
    dem_crs = CRS(dem_dataset.crs)
    if dem_crs is None:
        raise ValueError("DEM has no CRS.")
    if dem_crs == src_crs:
        coords = list(zip(xs, ys))
    else:
        transformer = Transformer.from_crs(src_crs, dem_crs, always_xy=True)
        X, Y = transformer.transform(xs, ys)
        coords = list(zip(X, Y))

    zs: List[Optional[float]] = []
    for (z,) in dem_dataset.sample(coords):
        if z is None:
            zs.append(None)
        else:
            # Handle nodata
            if dem_dataset.nodata is not None and np.isclose(z, dem_dataset.nodata):
                zs.append(None)
            elif np.isnan(z):
                zs.append(None)
            else:
                zs.append(float(z))
    return zs


def segment_centerline_add_slope(
    input_gpkg: str | Path,
    dem_path: str | Path,
    interval_m: float,
    output_gpkg: str | Path,
    fid_field: str = None,
) -> None:
    """
    Split a stream centerline into fixed-length segments and add longitudinal slope
    computed from DEM elevations at each segment's endpoints.

    Parameters
    ----------
    input_gpkg : str | Path
        Path to input GeoPackage containing a centerline (LineString/MultiLineString).
    input_layer : str
        Layer name to read from the input GeoPackage.
    dem_path : str | Path
        Path to a DEM raster (projected; meters for z are assumed).
    interval_m : float
        Target segment length in meters (last segment per part may be shorter).
    output_gpkg : str | Path
        Path to the output GeoPackage.
    output_layer : str
        Name for the output layer (will be overwritten if it exists).
    fid_field : str, optional
        Name of a field to use as parent feature id (falls back to df.index if None).

    Notes
    -----
    - CRS: The source CRS is preserved in the output. Distance math is performed in a
      projected CRS (meters), reprojecting back for writing.
    - Units: Slope reported as m/m and percent.
    - Slope sign: slope = (z_start - z_end) / length ; positive = drop downstream.
    """
    input_gpkg = Path(str(input_gpkg))
    dem_path = Path(str(dem_path))
    output_gpkg = Path(str(output_gpkg))
    _validate_paths(input_gpkg, dem_path)

    # Read input
    df = gpd.read_file(input_gpkg)
    if df.empty:
        raise ValueError("Input layer has no features.")
    if df.crs is None:
        raise ValueError("Input layer has no CRS. Define one before running.")
    if interval_m <= 0:
        raise ValueError(f"'interval_m' must be > 0, got {interval_m}.")

    # Choose working CRS in meters for segmentation
    work_crs = _ensure_projected_crs(CRS(df.crs), df)
    df_work = df.to_crs(work_crs)

    # Prepare DEM
    dem_ds = rasterio.open(dem_path)

    # Build segment records
    seg_records: List[SegmentRecord] = []
    for row_idx, (idx, row) in enumerate(df_work.iterrows()):
        parent_id = int(row[fid_field]) if fid_field and fid_field in row else int(idx)
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        for part_idx, ls in _iter_lines(geom):
            if ls.length == 0:
                continue
            pieces = _split_to_interval_parts(ls, interval_m=interval_m)
            for seg_idx, (s, e, sub) in enumerate(pieces):
                # Endpoint coords in working CRS (for DEM sampling we'll reproject if needed)
                start: Point = Point(sub.coords[0])
                end: Point = Point(sub.coords[-1])
                xs = [start.x, end.x]
                ys = [start.y, end.y]
                zstart, zend = _sample_dem_points(xs, ys, work_crs, dem_ds)

                L = float(sub.length)
                if L <= 0:
                    slope = None
                    slope_pct = None
                elif zstart is None or zend is None:
                    slope = None
                    slope_pct = None
                else:
                    slope = (zstart - zend) / L
                    slope_pct = slope * 100.0

                seg_records.append(
                    SegmentRecord(
                        parent_fid=parent_id,
                        part_idx=part_idx,
                        seg_idx=seg_idx,
                        from_m=float(s),
                        to_m=float(e),
                        length_m=L,
                        z_start=zstart,
                        z_end=zend,
                        slope=slope,
                        slope_pct=slope_pct,
                        geometry=sub,
                    )
                )

    if not seg_records:
        raise ValueError("No segments were produced. Check input geometry and interval.")

    # Assemble GeoDataFrame in working CRS, then convert back to source CRS
    out_gdf = gpd.GeoDataFrame(
        {
            "parent_id": [r.parent_fid for r in seg_records],
            "part_idx": [r.part_idx for r in seg_records],
            "seg_idx": [r.seg_idx for r in seg_records],
            "from_m": [r.from_m for r in seg_records],
            "to_m": [r.to_m for r in seg_records],
            "length_m": [r.length_m for r in seg_records],
            "z_start": [r.z_start for r in seg_records],
            "z_end": [r.z_end for r in seg_records],
            "slope": [r.slope for r in seg_records],
            "slope_pct": [r.slope_pct for r in seg_records],
        },
        geometry=[r.geometry for r in seg_records],
        crs=work_crs,
    ).to_crs(df.crs)

    #print 5 lowest and highest slopes for sanity check
    sorted_slopes = sorted([r.slope for r in seg_records if r.slope is not None])
    print("5 lowest slopes (m/m):", sorted_slopes[:5])
    print("5 highest slopes (m/m):", sorted_slopes[-5:])
    
    # Write output (overwrite layer if exists)
    # Note: geopandas uses fiona/OGR; GPKG supports layer overwrite via driver options.
    # Easiest approach is to remove existing layer by writing with mode="w" for the first time
    # then "a" for subsequent layers. Here we only write one layer.
    if output_gpkg.exists():
        # remove existing layer if present by writing with driver options
        pass
    out_gdf.to_file(output_gpkg, driver="GPKG")

    dem_ds.close()



default_input = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Chumstick Creek REM\streams\streams_7000k.gpkg"
default_dem = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Chumstick Creek REM\streams\breached_dem.tif"
default_output = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Chumstick Creek REM\streams\streams_7000k_with_slope_50m_breached.gpkg"
default_interval = 50

segment_centerline_add_slope(
    input_gpkg=default_input,
    dem_path=default_dem,
    interval_m=default_interval,
    output_gpkg=default_output,
    fid_field=None,  # or e.g. "FID"
)
