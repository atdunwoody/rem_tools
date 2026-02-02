from whitebox_workflows import WbEnvironment
import whitebox
import os
import geopandas as gpd
import rasterio
from shapely.geometry import LineString, MultiLineString, Point
from typing import Optional, List, Tuple
from rasterstats import zonal_stats
import fiona
import numpy as np


def get_streams(
    dem: str,
    output_dir: str,
    threshold: int = 100000,
    overwrite: bool = False,
    breach_depressions: bool = True,
    thin_n: int = 10,
    create_thinned: bool = True,
    precip_raster: Optional[str] = None,
    # NEW:
    slope_interval: Optional[float] = None,   # e.g., 50 (ft) or 20 (m)
    slope_units: str = "percent",            # "percent" or "ratio"
):
    """
    Processes a DEM to extract streams, save them to a GeoPackage, and—
    optionally—create a thinned centerline layer.

    NEW:
      If slope_interval is provided, sample DEM along each stream at that spacing
      and add slope summary fields to the stream lines + write slope sample points.

    Returns:
      streams_gpkg
    """
    # --- Whitebox setup ------------------------------------------------------
    wbt = whitebox.WhiteboxTools()
    wbe = WbEnvironment()

    # --- Prepare output directory -------------------------------------------
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Intermediate filenames ---------------------------------------------
    filled_dem    = os.path.join(output_dir, "filled_dem.tif")
    d8_pointer    = os.path.join(output_dir, "d8_pointer.tif")
    flow_accum    = os.path.join(output_dir, "flow_accum.tif")
    breached_dem  = os.path.join(output_dir, "breached_dem.tif")
    if overwrite:
        os.remove(filled_dem) if os.path.exists(filled_dem) else None
        os.remove(d8_pointer) if os.path.exists(d8_pointer) else None
        os.remove(flow_accum) if os.path.exists(flow_accum) else None
        os.remove(breached_dem) if os.path.exists(breached_dem) else None

    # --- Breach / fill -------------------------------------------------------
    if breach_depressions and not os.path.exists(breached_dem):
        wbt.breach_depressions_least_cost(dem, breached_dem, 10)
    if not breach_depressions and not os.path.exists(filled_dem):
        wbt.fill_depressions(dem, filled_dem)

    # --- Flow direction & accumulation --------------------------------------
    src_dem = breached_dem if breach_depressions else filled_dem
    if not os.path.exists(d8_pointer) or not os.path.exists(flow_accum):
        wbt.d8_pointer(src_dem, d8_pointer)
        wbt.d8_flow_accumulation(src_dem, flow_accum)

    # --- Extract streams -----------------------------------------------------
    streams_raster = os.path.join(output_dir, f"streams_{int(threshold/1000)}k.tif")
    if not os.path.exists(streams_raster):
        wbt.extract_streams(flow_accum, streams_raster, threshold)

    # --- Raster → Shapefile → GeoPackage ------------------------------------
    streams_shp   = streams_raster.replace(".tif", ".shp")
    streams_gpkg  = streams_raster.replace(".tif", ".gpkg")
    streams_layer = os.path.splitext(os.path.basename(streams_gpkg))[0]

    if not os.path.exists(streams_shp):
        wbt.raster_streams_to_vector(streams_raster, d8_pointer, streams_shp)

    # read vector, ensure proper CRS
    gdf = gpd.read_file(streams_shp)
    with rasterio.open(dem) as src:
        dem_crs = src.crs
    if dem_crs is None:
        raise ValueError("DEM has no CRS.")
    if gdf.crs is None:
        gdf.set_crs(dem_crs, inplace=True)
    else:
        gdf = gdf.to_crs(dem_crs)

    # save to GeoPackage
    gdf.to_file(streams_gpkg, layer=streams_layer, driver="GPKG")

    # --- Thin centerline if requested ---------------------------------------
    thinned_gpkg = None
    if create_thinned:
        print(f"Thinning centerline by keeping every {thin_n}ᵗʰ vertex...")
        thinned_gpkg = streams_gpkg.replace(".gpkg", "_thinned.gpkg")
        thin_centerline(
            input_gpkg=streams_gpkg,
            layer_name=streams_layer,
            output_gpkg=thinned_gpkg,
            output_layer=f"{streams_layer}_thinned",
            n=thin_n
        )

    print("Adding drainage area and bankfull dimensions...")
    add_DA_to_stream(streams_gpkg, flow_accum)
    add_BF_to_streams_Legg(streams_gpkg, precip_raster)
    add_BF_to_streams_Castro(streams_gpkg)
    add_BF_to_streams_Beechie(streams_gpkg)

    # NEW: slope sampling LAST so it doesn't get overwritten by later to_file() calls
    if slope_interval is not None and slope_interval > 0:
        print(f"Adding slope from DEM every {slope_interval} CRS units (units='{slope_units}')...")
        add_stream_slope_from_dem(
            streams_gpkg=streams_gpkg,
            dem_path=dem,
            interval=slope_interval,
            slope_units=slope_units,
            samples_out_gpkg=streams_gpkg.replace(".gpkg", "_slope_samples.gpkg"),
            samples_layer="slope_samples"
        )

    print(f"[✔] Streams extracted to: {streams_gpkg}")
    return streams_gpkg


# --------------------------------------------------------------------------------------
# NEW: slope sampling from DEM at fixed intervals
# --------------------------------------------------------------------------------------
def add_stream_slope_from_dem(
    streams_gpkg: str,
    dem_path: str,
    interval: float,
    *,
    slope_units: str = "percent",               # "percent" or "ratio"
    samples_out_gpkg: Optional[str] = None,     # separate GPKG for sample points (recommended)
    samples_layer: str = "slope_samples",
    line_id_field: str = "stream_id",           # created if missing
) -> None:
    """
    Samples DEM elevations along each stream at a fixed interval (in CRS units),
    computes local slope between samples, and writes:
      - slope summary fields onto the line features
      - an optional point layer of samples (dist, elev, slope)

    Notes:
      - interval is in the stream CRS units (feet or meters depending on CRS).
      - slope is dimensionless; "percent" multiplies by 100.
      - Sample points are written to a separate gpkg by default to avoid overwrites.
    """
    if interval <= 0:
        raise ValueError("interval must be > 0")

    # Load streams and match CRS to DEM
    streams = gpd.read_file(streams_gpkg)
    if streams.empty:
        print("[SLOPE] No stream features found.")
        return

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        dem_nodata = src.nodata

    if dem_crs is None:
        raise ValueError("DEM has no CRS.")

    if streams.crs is None:
        streams = streams.set_crs(dem_crs)
    elif streams.crs != dem_crs:
        streams = streams.to_crs(dem_crs)

    # Ensure a stable per-feature ID
    if line_id_field not in streams.columns:
        streams[line_id_field] = np.arange(len(streams), dtype=int)

    # Prepare outputs
    # (store slope statistics on lines)
    streams["slope_mean"] = np.nan
    streams["slope_median"] = np.nan
    streams["slope_max"] = np.nan

    # Build sample point records
    sample_records = []

    def _iter_line_parts(geom):
        """Yield LineString parts from LineString or MultiLineString; ignore other geometry types."""
        if geom is None or geom.is_empty:
            return
        if isinstance(geom, LineString):
            yield geom
        elif isinstance(geom, MultiLineString):
            for g in geom.geoms:
                if isinstance(g, LineString):
                    yield g

    def _sample_elevations(points: List[Point]) -> List[float]:
        """Sample DEM at point coordinates, returning floats (np.nan if nodata/outside)."""
        if not points:
            return []
        coords = [(p.x, p.y) for p in points]
        vals = []
        with rasterio.open(dem_path) as src:
            for v in src.sample(coords):
                z = float(v[0]) if v is not None and len(v) else np.nan
                if dem_nodata is not None and np.isfinite(z) and z == dem_nodata:
                    z = np.nan
                vals.append(z)
        return vals

    # For each feature, generate points at distances 0, interval, 2*interval, ... end
    for idx, row in streams.iterrows():
        geom = row.geometry
        sid = int(row[line_id_field])

        if geom is None or geom.is_empty:
            continue

        # Concatenate sample points across multipart geometries, keeping continuous distance
        points_all: List[Point] = []
        dists_all: List[float] = []

        dist_offset = 0.0
        for part in _iter_line_parts(geom):
            L = float(part.length)
            if not np.isfinite(L) or L <= 0:
                continue

            # distances along this part
            dists = list(np.arange(0.0, L, interval))
            if not np.isclose(dists[-1] if dists else -1, L):
                dists.append(L)

            pts = [part.interpolate(d) for d in dists]

            # offset distances for multipart
            points_all.extend(pts)
            dists_all.extend([dist_offset + d for d in dists])

            dist_offset += L

        if len(points_all) < 2:
            continue

        z_all = _sample_elevations(points_all)

        # compute segment slopes between successive points
        slopes = []
        for i in range(1, len(points_all)):
            z1, z2 = z_all[i - 1], z_all[i]
            d1, d2 = dists_all[i - 1], dists_all[i]
            dd = d2 - d1

            if not np.isfinite(z1) or not np.isfinite(z2) or not np.isfinite(dd) or dd <= 0:
                s = np.nan
            else:
                s = (z2 - z1) / dd  # rise/run

            slopes.append(s)

            # Save sample record at the *downstream* point (i) with slope for segment (i-1 -> i)
            sample_records.append({
                line_id_field: sid,
                "sample_i": i,
                "dist": float(dists_all[i]),
                "elev": float(z_all[i]) if np.isfinite(z_all[i]) else np.nan,
                "slope_ratio": float(s) if np.isfinite(s) else np.nan,
                "geometry": points_all[i],
            })

        slopes_arr = np.array(slopes, dtype=float)
        slopes_arr = slopes_arr[np.isfinite(slopes_arr)]
        if slopes_arr.size == 0:
            continue

        # Convert units if requested
        if slope_units.lower() == "percent":
            slopes_for_stats = slopes_arr * 100.0
        else:
            slopes_for_stats = slopes_arr

        streams.at[idx, "slope_mean"] = float(np.mean(slopes_for_stats))
        streams.at[idx, "slope_median"] = float(np.median(slopes_for_stats))
        streams.at[idx, "slope_max"] = float(np.max(slopes_for_stats))

    # Write updated streams back (note: follows your existing overwrite pattern)
    streams.to_file(streams_gpkg, driver="GPKG")

    # Write sample points to separate gpkg (recommended)
    if samples_out_gpkg:
        samples_gdf = gpd.GeoDataFrame(sample_records, crs=streams.crs)
        # Add a convenience slope field in requested units
        if slope_units.lower() == "percent":
            samples_gdf["slope"] = samples_gdf["slope_ratio"] * 100.0
            samples_gdf["slope_units"] = "percent"
        else:
            samples_gdf["slope"] = samples_gdf["slope_ratio"]
            samples_gdf["slope_units"] = "ratio"

        samples_gdf.to_file(samples_out_gpkg, layer=samples_layer, driver="GPKG")
        print(f"[SLOPE] Wrote slope sample points to: {samples_out_gpkg} (layer='{samples_layer}')")


def add_DA_to_stream(
    streams_gpkg: str,
    flow_accum_raster: str,
    unit: str = "m") -> None:
    """
    Adds a 'DA_km2' field to the streams layer in `streams_gpkg`,
    by taking the maximum flow-accumulation value within a 1 m buffer
    and converting to km².
    Assumes the flow-accumulation raster has a cell size of 10 m.
    """
    # 1. load
    streams = gpd.read_file(streams_gpkg)

    # 2. buffer
    streams['buffer'] = streams.geometry.buffer(0.1)

    # 3. zonal stats
    with rasterio.open(flow_accum_raster) as src:
        nodata = src.nodata
        cell_size = src.res[0]  # assuming square cells

    stats = zonal_stats(
        streams['buffer'],
        flow_accum_raster,
        # pick the 99th percentile to avoid outliers
        stats=['median'],
        nodata=nodata
    )

    # 4. compute conversion factor: km²-per-cell-unit
    if unit.lower() == "ft":
        # 10 meter cell size, so multiply by 100 to get m²
        # 1 ft² → 0.092903 m², then → km² by ×1e-6
        conv = 0.092903 * 1e-6 * cell_size**2
    else:
        # assume metres: 1 m² → 1e-6 km²
        conv = 1e-6 * cell_size**2

    # 5. element‑wise multiply
    streams['DA_km2'] = [
        (s['median'] or 0) * conv
        for s in stats
    ]

    # 6. cleanup & save
    streams = streams.drop(columns='buffer')
    streams.to_file(streams_gpkg, driver="GPKG")

def add_BF_to_streams_Legg(streams_gpkg_path: str, precip_raster_path: str = None) -> None:
    """
    Reads stream features from a GeoPackage, computes mean annual precipitation (cm)
    for each feature based on intersecting PRISM raster data, adds a new precip field,
    then computes bankfull width (m) and depth (m) using Legg & Olson 2015.

    Parameters
    ----------
    streams_gpkg_path : str
        Path to the input streams GeoPackage.
    precip_raster_path : str
        Path to the input annual precipitation raster (PRISM .tif in mm).
    output_gpkg_path : str
        Path where the output GeoPackage (with new fields) will be written.
    layer : str, optional
        Name of the layer to read from the input GeoPackage. If None, the default
        (first) layer will be used.
    """
    # constants
    KM2_TO_MI2 = 0.386102          # km² → mi²
    CM_TO_IN = 1.0 / 2.54          # cm → in
    FT_TO_M = 0.3048               # ft → m

    # 1. Read streams layer
    streams = gpd.read_file(streams_gpkg_path)

    # after reading:
    streams = streams[streams.geometry.notnull()]
    streams = streams[streams.is_valid]

    # 4. Compute drainage area in mi² and precip in inches
    streams['DA_mi2'] = streams['DA_km2'] * KM2_TO_MI2
    streams['ann_precip_in'] = 72.17 * CM_TO_IN # 72.17 is the average annual precipitation in cm for the GRMW basin
    streams['ann_precip_cm'] = 72.17 # 72.17 is the average annual precipitation in cm for the GRMW basin

    # 5. Bankfull width (m) based on Legg & Olson 2015:
    #    width_ft = 1.16 * 0.91 * (DA_mi2^0.381) * (precip_in^0.634)
    #    convert ft → m
    streams['BF_width_Legg_m'] = (
        FT_TO_M *
        1.16 * 0.91 *
        (streams['DA_mi2'] ** 0.381) *
        (streams['ann_precip_in'] ** 0.634)
    )

    # 6. Bankfull depth (m) based on Legg & Olson 2015:
    #    depth = 0.0939 * (DA_km2^0.233) * (precip_cm^0.264)
    streams['BF_depth_Legg_m'] = (
        0.0939 *
        (streams['DA_km2'] ** 0.233) *
        (streams['ann_precip_cm'] ** 0.264)
    )

    # 7. Write to new GeoPackage (overwrites if exists)
    streams.to_file(streams_gpkg_path, driver='GPKG')
    return streams_gpkg_path

def add_BF_to_streams_Castro(streams_gpkg_path: str) -> None:
    """
    Reads stream features from a GeoPackage, computes bankfull width (m) and depth (m)
    using Castro & Jackson 2001.

    Parameters
    ----------
    streams_gpkg_path : str
        Path to the input streams GeoPackage.
    """
    # 1. Read streams layer
    streams = gpd.read_file(streams_gpkg_path)
    km2_to_mi2 = 0.386102  # km² to mi²
    ft_to_m = 0.3048  # ft to m
    # 2. Compute bankfull width (m) based on Castro & Jackson 2001:
    #    width_m = ft_to_m * 9.40 * (DA * m2_to_mi2) ** 0.42
    streams['BF_width_Castro_m'] = ft_to_m * 9.40 * ((streams['DA_km2'] * km2_to_mi2) ** 0.42)

    # 3. Compute bankfull depth (m) based on Castro & Jackson 2001:
    #    depth_ft = 0.61 * DA_mi^2 **0.33
    streams['BF_depth_Castro_m'] = ft_to_m * 0.61 * ((streams['DA_km2'] * km2_to_mi2) ** 0.33)

    # 4. Write to new GeoPackage (overwrites if exists)
    streams.to_file(streams_gpkg_path, driver='GPKG')
    return streams_gpkg_path

def add_BF_to_streams_Beechie(streams_gpkg_path: str) -> None:
    """
    Reads stream features from a GeoPackage, computes bankfull width (m) and depth (m)
    using Beechie and IMAKI 2013.

    Parameters
    ----------
    streams_gpkg_path : str
        Path to the input streams GeoPackage.
    """
    # 1. Read streams layer
    streams = gpd.read_file(streams_gpkg_path)
    # 2. Compute bankfull width (m) based on Beechie and IMAKI 2013:
    P_cm_yr = 72.17  # average annual precipitation in cm for the GRMW basin based on PRISM
    streams['BF_width_Beechie_m'] = 0.177 * ((streams['DA_km2']) ** 0.397) * P_cm_yr ** 0.453
    streams['BF_depth_Beechie_m'] = streams['BF_width_Beechie_m'] * streams['BF_depth_Castro_m'] / streams['BF_width_Castro_m']

    # 4. Write to new GeoPackage (overwrites if exists)
    streams.to_file(streams_gpkg_path, driver='GPKG')
    return streams_gpkg_path

def thin_centerline( input_gpkg: str, layer_name: str, output_gpkg: str,
    output_layer: Optional[str] = None, n: int = 10) -> None:
    """
    Keeps every nᵗʰ vertex in each LineString (or part of a MultiLineString)
    from the specified layer in input_gpkg, writing to output_gpkg.
    """
    gdf = gpd.read_file(input_gpkg, layer=layer_name)

    def _thin(geom):
        if geom is None or geom.is_empty:
            return geom
        if isinstance(geom, LineString):
            coords = list(geom.coords)
            pts = coords[::n]
            if coords[-1] not in pts:
                pts.append(coords[-1])
            return LineString(pts)
        if isinstance(geom, MultiLineString):
            parts = []
            for part in geom.geoms:
                coords = list(part.coords)
                pts = coords[::n]
                if coords[-1] not in pts:
                    pts.append(coords[-1])
                parts.append(LineString(pts))
            return MultiLineString(parts)
        return geom

    gdf["geometry"] = gdf.geometry.apply(_thin)
    out_layer = output_layer or layer_name
    gdf.to_file(output_gpkg, layer=out_layer, driver="GPKG")

def threshold_lines_by_length(
    input_gpkg: str,
    output_gpkg: str,
    threshold: float = 1200.0
) -> None:
    """
    Read the first layer of `input_gpkg`, keep only LineString/MultiLineString
    features longer than `threshold` (in CRS units, typically metres), and
    write them to `output_gpkg` in the same layer name.
    """
    # get first layer name
    layers = fiona.listlayers(input_gpkg)
    if not layers:
        raise ValueError(f"No layers found in {input_gpkg!r}")
    layer = layers[0]

    # load features
    gdf = gpd.read_file(input_gpkg, layer=layer)

    # select only line geometries
    is_line = gdf.geometry.type.isin(["LineString", "MultiLineString"])
    lines = gdf[is_line].copy()

    # compute length (in CRS units)
    lines["__length"] = lines.geometry.length

    # filter by threshold
    filtered = lines[lines["__length"] > threshold].drop(columns="__length")

    # write back, preserving layer name
    filtered.to_file(output_gpkg, layer=layer, driver="GPKG")


if __name__ == "__main__":
    dem = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\HAWS Map\Topography\2016_USDA_DEM ndv.tif"
    output_dir = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\HAWS Map\Streams"
    threshold = 5000000   # CRS units x cell size (e.g., 5 million for 10 m cells = 50 km²)

    get_streams(
        dem=dem,
        output_dir=output_dir,
        threshold=threshold,
        overwrite=False,
        breach_depressions=True,
        create_thinned=False,
        precip_raster=None,
        slope_interval=15*15,      # set to 10-20x BFW
        slope_units="percent",    # "percent" or "ratio"
    )
