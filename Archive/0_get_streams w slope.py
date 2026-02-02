# get_streams.py

from whitebox_workflows import WbEnvironment
import whitebox
import os
import geopandas as gpd
import rasterio
from shapely.geometry import LineString, MultiLineString, MultiPoint
from shapely.ops import split
from typing import Optional
from rasterstats import zonal_stats
import fiona
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling


def reproject_raster(src_path: str, dst_path: str, dst_crs) -> None:
    """
    Reproject a raster to a new CRS using rasterio.warp.
    """
    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError(f"Source raster {src_path!r} has no CRS; cannot reproject.")

        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )

        kwargs = src.meta.copy()
        kwargs.update(
            {
                "crs": dst_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )


def get_streams(
    dem: str,
    output_dir: str,
    threshold: int = 100000,
    overwrite: bool = False,
    breach_depressions: bool = True,
    thin_n: int = 10,
    create_thinned: bool = True,
    precip_raster: Optional[str] = None,
    segment_length: Optional[float] = None,
    calc_BF_dims: Optional[bool] = False,
):
    """
    Processes a DEM to extract streams, save them to a GeoPackage, and—optionally—
    create a thinned centerline layer.

    If `segment_length` is provided, it is interpreted as a *target segment length
    in feet*. The script will:

      1. Ensure the DEM and derived streams are in a projected CRS.
         - If the DEM is not projected, it is reprojected to Albers (EPSG:5070, metres).
      2. Convert `segment_length` from feet into the CRS linear units (typically metres
         in EPSG:5070, or feet if the CRS is in feet).
      3. Split each line into approximately fixed-length segments in CRS units.
      4. Compute slope for each segment as:

            slope = abs(z_end - z_start) / segment_length_actual

         where `segment_length_actual` is the geometric length of each segment in
         CRS units (so the slope is unitless, e.g., m/m or ft/ft).

    The slope is stored in a field named `slope`.

    Returns:
        Path to final streams GeoPackage (segmented or not, depending on options).
    """
    # --- Whitebox setup ------------------------------------------------------
    wbt = whitebox.WhiteboxTools()
    wbe = WbEnvironment()

    # --- Prepare output directory -------------------------------------------
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- Ensure DEM is in a projected CRS (reproject to Albers if needed) ---
    with rasterio.open(dem) as src:
        dem_crs = src.crs

    if dem_crs is None:
        raise ValueError("DEM has no CRS; cannot proceed.")

    if not dem_crs.is_projected:
        albers_epsg = "EPSG:5070"
        print(
            f"Input DEM CRS {dem_crs} is not projected. "
            f"Reprojecting to Albers ({albers_epsg}) for processing..."
        )
        dem_albers = os.path.join(output_dir, "dem_albers.tif")
        if overwrite or not os.path.exists(dem_albers):
            reproject_raster(dem, dem_albers, albers_epsg)
        dem = dem_albers
        with rasterio.open(dem) as src2:
            dem_crs = src2.crs
        print(f"DEM reprojected. Using {dem} with CRS {dem_crs}.")

    # --- Intermediate filenames ---------------------------------------------
    filled_dem = os.path.join(output_dir, "filled_dem.tif")
    d8_pointer = os.path.join(output_dir, "d8_pointer.tif")
    flow_accum = os.path.join(output_dir, "flow_accum.tif")
    breached_dem = os.path.join(output_dir, "breached_dem.tif")

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
    streams_shp = streams_raster.replace(".tif", ".shp")
    streams_gpkg = streams_raster.replace(".tif", ".gpkg")
    streams_layer = os.path.splitext(os.path.basename(streams_gpkg))[0]

    if not os.path.exists(streams_shp):
        wbt.raster_streams_to_vector(streams_raster, d8_pointer, streams_shp)

    # read vector, ensure proper CRS (should match projected DEM)
    gdf = gpd.read_file(streams_shp)
    with rasterio.open(dem) as src:
        dem_crs = src.crs

    if dem_crs is None:
        raise ValueError("Processing DEM has no CRS after reprojection step; aborting.")

    if gdf.crs is None:
        gdf.set_crs(dem_crs, inplace=True)
    else:
        gdf = gdf.to_crs(dem_crs)

    if not gdf.crs.is_projected:
        # This *should* not happen, but guard just in case.
        raise ValueError(
            f"Streams layer CRS {gdf.crs} is not projected even after DEM "
            "reprojection. Cannot safely segment by length."
        )

    # save to GeoPackage (unsegmented)
    gdf.to_file(streams_gpkg, layer=streams_layer, driver="GPKG")

    # --- Prepare segment length in CRS units (segment_length given in feet) --
    target_gpkg = streams_gpkg
    if segment_length is not None:
        # interpret input segment_length as feet, convert to CRS units
        crs = gdf.crs
        unit_name = None
        try:
            unit_name = crs.axis_info[0].unit_name.lower()
        except Exception:
            unit_name = None

        # US survey foot (more precise) to metres
        FT_TO_M = 0.304800609601

        if unit_name in ("metre", "meter", "metres", "meters"):
            seg_len_crs = segment_length * FT_TO_M
            print(
                f"Segment length requested: {segment_length} ft "
                f"→ {seg_len_crs:.3f} m in CRS {crs.to_string()}"
            )
        elif unit_name in ("foot", "feet", "us_survey_foot", "foot_us", "foot_survey_us"):
            seg_len_crs = segment_length  # CRS already in feet
            print(
                f"Segment length requested: {segment_length} ft "
                f"(CRS units are feet in {crs.to_string()})"
            )
        else:
            # Fallback: treat segment_length as already in CRS units
            seg_len_crs = segment_length
            print(
                f"Segment length requested: {segment_length} (assumed CRS units) "
                f"in {crs.to_string()} (unknown linear unit)."
            )

        print(
            f"Splitting streams into ~{segment_length} ft segments "
            f"(~{seg_len_crs:.3f} {unit_name or 'CRS units'}) and adding slope..."
        )

        segmented_gpkg = streams_gpkg.replace(".gpkg", "_with_slope.gpkg")
        split_streams_by_length_and_add_slope(
            input_gpkg=streams_gpkg,
            dem=dem,
            segment_length_crs=seg_len_crs,
            output_gpkg=segmented_gpkg,
            layer_name=streams_layer,
        )
        target_gpkg = segmented_gpkg

    print("Adding drainage area to streams...")
    add_DA_to_stream(target_gpkg, flow_accum)

    # --- Thin centerline if requested ---------------------------------------
    thinned_gpkg = None
    if create_thinned:
        print(f"Thinning centerline by keeping every {thin_n}ᵗʰ vertex...")
        thinned_gpkg = target_gpkg.replace(".gpkg", "_thinned.gpkg")
        thin_centerline(
            input_gpkg=target_gpkg,
            layer_name=streams_layer,
            output_gpkg=thinned_gpkg,
            output_layer=f"{streams_layer}_thinned",
            n=thin_n,
        )

    if calc_BF_dims:
        print("Adding bankfull regression estimates to streams...")
        add_BF_to_streams_Legg(target_gpkg, precip_raster)
        add_BF_to_streams_Castro(target_gpkg)
        add_BF_to_streams_Beechie(target_gpkg)

    print(f"[✔] Streams extracted to: {target_gpkg}")
    return target_gpkg


def add_DA_to_stream(
    streams_gpkg: str,
    flow_accum_raster: str,
    unit: str = "m",
) -> None:
    """
    Adds a 'DA_km2' field to the streams layer in `streams_gpkg`,
    by taking the median flow-accumulation value within a small buffer
    and converting to km².

    Assumes the flow-accumulation raster values represent *number of cells*
    upstream. Cell area is computed from raster resolution.

    Parameters
    ----------
    streams_gpkg : str
        Path to streams GeoPackage.
    flow_accum_raster : str
        Path to flow accumulation raster (cell counts).
    unit : {'m', 'ft'}
        Linear units of the DEM/flow accumulation raster.
    """
    # 1. load
    streams = gpd.read_file(streams_gpkg)

    # 2. buffer
    streams["buffer"] = streams.geometry.buffer(0.1)

    # 3. zonal stats
    with rasterio.open(flow_accum_raster) as src:
        nodata = src.nodata
        cell_size = src.res[0]  # assuming square cells

    stats = zonal_stats(
        streams["buffer"],
        flow_accum_raster,
        stats=["median"],
        nodata=nodata,
    )

    # 4. compute conversion factor: km²-per-cell
    if unit.lower() == "ft":
        # cell_size in feet → cell area in ft² → m² → km²
        conv = 0.092903 * 1e-6 * cell_size**2
    else:
        # assume metres: cell_size in m → area in m² → km²
        conv = 1e-6 * cell_size**2

    # 5. element-wise multiply
    streams["DA_km2"] = [(s["median"] or 0) * conv for s in stats]
    streams["DA_mi2"] = streams["DA_km2"] * 0.386102

    # 6. cleanup & save
    streams = streams.drop(columns="buffer")
    streams.to_file(streams_gpkg, driver="GPKG")


def thin_centerline(
    input_gpkg: str,
    layer_name: str,
    output_gpkg: str,
    output_layer: Optional[str] = None,
    n: int = 10,
) -> None:
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


def split_streams_by_length_and_add_slope(
    input_gpkg: str,
    dem: str,
    segment_length_crs: float,
    output_gpkg: str,
    layer_name: Optional[str] = None,
) -> None:
    """
    Split each LineString/MultiLineString in `input_gpkg` into segments of approx.
    `segment_length_crs` (in CRS units, typically metres), compute slope for each segment
    as abs(z_end - z_start) / L_segment using the DEM, and write to `output_gpkg`.

    The output preserves all original attributes and adds:
        - slope : float  (unitless gradient in DEM units per CRS distance units)
    """

    if segment_length_crs <= 0:
        raise ValueError("segment_length_crs must be > 0")

    # Determine layer
    if layer_name is None:
        layers = fiona.listlayers(input_gpkg)
        if not layers:
            raise ValueError(f"No layers found in {input_gpkg!r}")
        layer_name = layers[0]

    gdf = gpd.read_file(input_gpkg)

    if gdf.crs is None or not gdf.crs.is_projected:
        raise ValueError(
            f"Input streams layer CRS {gdf.crs} is not projected. "
            "Streams should already have been reprojected to a projected CRS "
            "before segmentation."
        )

    def _split_line(geom: LineString, seg_len: float):
        total_len = geom.length
        if total_len <= seg_len:
            return [geom]

        # Distances along the line at which to split
        distances = np.arange(seg_len, total_len, seg_len)
        if len(distances) == 0:
            return [geom]

        points = [geom.interpolate(d) for d in distances]
        result = split(geom, MultiPoint(points))
        return list(result.geoms)

    def _segmentize_geometry(geom, seg_len: float):
        if geom is None or geom.is_empty:
            return []
        if isinstance(geom, LineString):
            return _split_line(geom, seg_len)
        if isinstance(geom, MultiLineString):
            segs = []
            for part in geom.geoms:
                segs.extend(_split_line(part, seg_len))
            return segs
        # Non-line geometries are ignored
        return []

    new_rows = []
    skipped_no_samples = 0
    skipped_zero_len = 0

    with rasterio.open(dem) as src:
        for _, row in gdf.iterrows():
            geom = row.geometry
            segments = _segmentize_geometry(geom, segment_length_crs)

            for seg in segments:
                coords = list(seg.coords)
                if len(coords) < 2:
                    skipped_zero_len += 1
                    continue

                (x1, y1) = coords[0][:2]
                (x2, y2) = coords[-1][:2]

                # Sample DEM at segment endpoints
                samples = list(src.sample([(x1, y1), (x2, y2)]))
                if len(samples) != 2:
                    skipped_no_samples += 1
                    continue

                z1 = float(samples[0][0])
                z2 = float(samples[1][0])

                seg_len = seg.length
                if seg_len == 0:
                    slope = None
                    skipped_zero_len += 1
                else:
                    slope = abs(z2 - z1) / seg_len

                attrs = row.to_dict()
                attrs["geometry"] = seg
                attrs["slope"] = slope
                new_rows.append(attrs)

    if not new_rows:
        raise ValueError(
            "No segments created; check input data and segment_length."
        )

    if skipped_no_samples > 0 or skipped_zero_len > 0:
        print(
            f"Segmentation warnings: {skipped_no_samples} segments skipped due to "
            f"no DEM samples; {skipped_zero_len} segments skipped due to zero length."
        )

    out_gdf = gpd.GeoDataFrame(new_rows, crs=gdf.crs)
    out_gdf.to_file(output_gpkg, driver="GPKG")


def threshold_lines_by_length(
    input_gpkg: str,
    output_gpkg: str,
    threshold: float = 1200.0,
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
    dem = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Minam RIver\output_USGS1m.tif"
    output_dir = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Minam RIver\Streams_1m\with slope"
    threshold = 5700000  # contributing area threshold (cell count * cell_area²)
    get_streams(
        dem=dem,
        output_dir=output_dir,
        threshold=threshold,
        overwrite=False,
        breach_depressions=True,
        create_thinned=False,
        precip_raster=None,   # Optional, can be set to a PRISM raster path
        segment_length=15*30*3.28, # in feet, will be converted to CRS units
    )
