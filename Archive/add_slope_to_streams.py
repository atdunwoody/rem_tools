from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Callable

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.mask
from pyproj import CRS, Transformer
from shapely.geometry import LineString, MultiLineString, Point, mapping, base
from shapely.ops import substring, transform as shp_transform


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
    Pick a projected CRS in meters for distance ops.
    - If already projected in meters, keep it.
    - If geographic, estimate an appropriate UTM from data centroid.
    """
    if src_crs is None:
        raise ValueError("Input layer has no CRS. Please define a CRS before running.")
    if src_crs.is_projected and src_crs.axis_info and src_crs.axis_info[0].unit_name.lower() in {"metre", "meter", "metres", "meters"}:
        return src_crs
    utm = CRS.estimate_utm_crs(gdf.geometry.unary_union.centroid)
    if utm is None:
        raise ValueError("Failed to estimate a projected CRS for distance operations.")
    return utm


def _iter_lines(geom) -> Iterable[Tuple[int, LineString]]:
    """Yield (part_idx, LineString) for LineString or MultiLineString (recursing GeometryCollections)."""
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield 0, geom
    elif isinstance(geom, MultiLineString):
        for i, part in enumerate(geom.geoms):
            if not part.is_empty:
                yield i, part
    else:
        for i, part in enumerate(getattr(geom, "geoms", [])):
            if isinstance(part, (LineString, MultiLineString)):
                for j, sub in _iter_lines(part):
                    yield i * 100000 + j, sub  # stable composite index


def _split_to_interval_parts(ls: LineString, interval_m: float) -> List[Tuple[float, float, LineString]]:
    """
    Split a LineString into consecutive substrings of length 'interval_m',
    returning a list of (from_m, to_m, geometry). The final piece may be shorter.
    """
    L = ls.length
    if L == 0:
        return []
    cuts = np.arange(0.0, L, interval_m, dtype=float)
    if cuts.size == 0 or cuts[0] != 0.0:
        cuts = np.insert(cuts, 0, 0.0)
    ranges = [(float(s), float(min(s + interval_m, L))) for s in cuts]
    ranges = [(s, e) for s, e in ranges if e > s]  # guard against degenerate pieces
    parts: List[Tuple[float, float, LineString]] = []
    for s, e in ranges:
        sub = substring(ls, s, e, normalized=False)
        if isinstance(sub, LineString) and sub.length > 0:
            parts.append((s, e, sub))
    return parts


def _transformer_func(src: CRS, dst: CRS) -> Callable[[float, float, Optional[float]], Tuple[float, float]]:
    """Return a callable (x,y[,z]) -> (X,Y) using pyproj with always_xy=True."""
    t = Transformer.from_crs(src, dst, always_xy=True)
    return lambda x, y, z=None: t.transform(x, y)


def _polygon_to_dem_crs(geom: base.BaseGeometry, src_crs: CRS, dem_crs: CRS) -> base.BaseGeometry:
    """Reproject a shapely geometry from src_crs to dem_crs."""
    if src_crs == dem_crs:
        return geom
    fn = _transformer_func(src_crs, dem_crs)
    return shp_transform(fn, geom)


def _min_dem_in_buffer(
    pt: Point,
    buffer_radius_m: float,
    work_crs: CRS,
    dem_ds: rasterio.io.DatasetReader,
) -> Optional[float]:
    """
    Compute the minimum DEM elevation within a circular buffer of radius 'buffer_radius_m'
    around 'pt' (given in work_crs). Returns None if no valid DEM pixels intersect.
    """
    if buffer_radius_m <= 0:
        raise ValueError(f"'buffer_radius_m' must be > 0, got {buffer_radius_m}.")
    dem_crs = CRS(dem_ds.crs)
    if dem_crs is None:
        raise ValueError("DEM has no CRS.")

    # Buffer in a meters CRS (work_crs), then reproject to DEM CRS for masking.
    buf_work = pt.buffer(buffer_radius_m)  # meters in work_crs
    buf_dem = _polygon_to_dem_crs(buf_work, work_crs, dem_crs)

    try:
        arr, _ = rasterio.mask.mask(dem_ds, [mapping(buf_dem)], crop=True, filled=False)
    except ValueError:
        # No intersection / outside bounds
        return None

    band = arr[0]  # 1-band DEM assumed
    # 'band' is a masked array when filled=False; compress drops masked (nodata) values.
    if np.ma.is_masked(band):
        vals = band.compressed()
    else:
        vals = band[~np.isnan(band)]

    if vals.size == 0:
        return None

    # If the dataset has a nodata that wasn't masked (rare), remove it explicitly.
    nodata = dem_ds.nodata
    if nodata is not None:
        vals = vals[~np.isclose(vals, nodata)]

    if vals.size == 0:
        return None
    return float(vals.min())


def segment_centerline_add_slope(
    input_gpkg: str | Path,
    dem_path: str | Path,
    interval_m: float,
    output_gpkg: str | Path,
    fid_field: str | None = None,
    endpoint_buffer_m: float = 10.0,
) -> None:
    """
    Split a stream centerline into fixed-length segments and add longitudinal slope
    using the minimum DEM elevation within a buffer (default 10 m radius) around each
    segment endpoint.

    CRS assumptions
    ---------------
    - Source linework CRS is preserved in output.
    - Distance operations (segmentation, 10 m buffers) are done in a projected CRS
      with meter units (auto-estimated if needed).
    - DEM values are treated as meters; buffers are reprojected to DEM CRS for masking.

    Parameters
    ----------
    input_gpkg : str | Path
        GeoPackage containing a centerline (LineString/MultiLineString).
    dem_path : str | Path
        DEM raster path.
    interval_m : float
        Target segment length in meters.
    output_gpkg : str | Path
        Output GeoPackage path (single layer).
    fid_field : str, optional
        Field to use as parent feature id. Defaults to the row index if None.
    endpoint_buffer_m : float, optional
        Buffer radius (meters) around each endpoint to search for minimum elevation.
    """
    input_gpkg = Path(str(input_gpkg))
    dem_path = Path(str(dem_path))
    output_gpkg = Path(str(output_gpkg))
    _validate_paths(input_gpkg, dem_path)

    # Read input
    df = gpd.read_file(input_gpkg)
    if df.empty:
        raise ValueError("Input GeoPackage has no features.")
    if df.crs is None:
        raise ValueError("Input layer has no CRS. Define one before running.")
    if interval_m <= 0:
        raise ValueError(f"'interval_m' must be > 0, got {interval_m}.")
    if endpoint_buffer_m <= 0:
        raise ValueError(f"'endpoint_buffer_m' must be > 0, got {endpoint_buffer_m}.")

    # Check if CRS matches
    
    
    
    # Open DEM
    dem_ds = rasterio.open(dem_path)

    seg_records: List[SegmentRecord] = []
    for idx, row in df.iterrows():
        parent_id = int(row[fid_field]) if fid_field and fid_field in row else int(idx)
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        for part_idx, ls in _iter_lines(geom):
            if ls.length == 0:
                continue
            for seg_idx, (s, e, sub) in enumerate(_split_to_interval_parts(ls, interval_m=interval_m)):
                start_pt = Point(sub.coords[0])
                end_pt = Point(sub.coords[-1])

                # Use min elevation within 10 m buffer at each endpoint
                zstart = _min_dem_in_buffer(start_pt, endpoint_buffer_m, df.crs, dem_ds)
                zend = _min_dem_in_buffer(end_pt, endpoint_buffer_m, df.crs, dem_ds)

                L = float(sub.length)
                if L <= 0 or zstart is None or zend is None:
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
        dem_ds.close()
        raise ValueError("No segments were produced. Check geometry and interval.")

    # Assemble and write output (back in source CRS)
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
        crs=df.crs,
    ).to_crs(df.crs)

    out_gdf.to_file(output_gpkg, driver="GPKG")
    dem_ds.close()


# --- Example usage ---
if __name__ == "__main__":
    default_input = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\SFToutle\REM\streams_700k_epsg2927.gpkg"
    default_dem = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\SFToutle\REM\2025_LiDAR_epsg2927.tif"
    default_output = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\SFToutle\REM\streams_700k_with_slope.gpkg"
    default_interval = 100

    segment_centerline_add_slope(
        input_gpkg=default_input,
        dem_path=default_dem,
        interval_m=default_interval,
        output_gpkg=default_output,
        fid_field=None,             # or e.g. "FID"
        endpoint_buffer_m=10.0,     # 10 m buffer around endpoints
    )
