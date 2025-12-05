from __future__ import annotations
from pathlib import Path
import re
from typing import Optional
import geopandas as gpd
from shapely.geometry.base import BaseGeometry

def _safe_name(val) -> str:
    """Turn any attribute value into a safe filename stem."""
    if val is None or (isinstance(val, float) and str(val) == "nan"):
        val = "unnamed"
    name = str(val).strip()
    # keep letters, numbers, dash, underscore; collapse whitespace to underscores
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_\-]+", "", name)
    # avoid empty
    return name or "unnamed"

def _fix_invalid_polys(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Light validity fix for polygons (no-op for lines/points)
    if gdf.geom_type.isin(["Polygon", "MultiPolygon"]).any():
        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf

def clip_and_save_each(
    clipping_vector: str | Path,
    vector_to_clip: str | Path,
    out_dir: str | Path,
    name_field: str = "descriptio",
    clip_layer: Optional[str] = None,
    vector_layer: Optional[str] = None,
    driver: Optional[str] = None,  # None => pick from extension, else e.g. "GPKG" or "ESRI Shapefile"
    out_extension: Optional[str] = ".gpkg",  # used if driver is None and out paths are directories
    predicate: str = "intersects",  # "intersects" is robust; change to "within" if desired
) -> None:
    """
    For each polygon feature in `clipping_vector`, clip `vector_to_clip` and save to a separate file.

    Parameters
    ----------
    clipping_vector : path to polygon dataset (any OGR-supported format)
    vector_to_clip : path to dataset to be clipped (lines/points/polys)
    out_dir : directory to write outputs
    name_field : attribute on clipping polygons used to name outputs (default 'descriptio')
    clip_layer, vector_layer : layer names for multi-layer containers (e.g., GPKG). Use None for single-layer files.
    driver : explicit output driver (e.g., "GPKG", "ESRI Shapefile"). If None, inferred from extension.
    out_extension : default extension (e.g., ".gpkg") when writing into a directory without per-file driver specified
    predicate : spatial predicate used to preselect candidates ("intersects", "contains", etc.)
    """
    clipping_vector = Path(clipping_vector)
    vector_to_clip = Path(vector_to_clip)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read data
    clips = gpd.read_file(clipping_vector, layer=clip_layer)
    data = gpd.read_file(vector_to_clip, layer=vector_layer)

    if clips.empty:
        print("[!] Clipping vector has no features. Exiting.")
        return
    if data.empty:
        print("[!] Vector to clip has no features. Exiting.")
        return

    # Basic geometry fixes for polygons
    clips = _fix_invalid_polys(clips)
    data = _fix_invalid_polys(data)

    # Handle CRS
    if clips.crs is None:
        raise ValueError("Clipping vector has no CRS defined.")
    if data.crs is None:
        raise ValueError("Vector to clip has no CRS defined.")

    if clips.crs != data.crs:
        print(f"[!] CRS mismatch: clips={clips.crs.to_string()} vs data={data.crs.to_string()}. Reprojecting data to match clips.")
        data = data.to_crs(clips.crs)

    # Spatial index for fast candidate selection
    sindex = data.sindex

    total = len(clips)
    written = 0
    skipped_empty = 0

    # Keep track of used names to avoid accidental overwrites
    taken_names = set()

    for idx, row in clips.iterrows():
        geom: BaseGeometry = row.geometry
        if geom is None or geom.is_empty:
            print(f"[{idx+1}/{total}] Skipping empty geometry.")
            continue

        # Name handling
        raw_name = row.get(name_field, None)
        base_name = _safe_name(raw_name)
        # Ensure unique filename
        out_name = base_name
        counter = 2
        while out_name.lower() in taken_names:
            out_name = f"{base_name}_{counter}"
            counter += 1
        taken_names.add(out_name.lower())

        # Candidate filter via spatial index
        bbox = geom.bounds
        candidate_idx = list(sindex.intersection(bbox))
        if not candidate_idx:
            print(f"[{idx+1}/{total}] '{out_name}' -> no candidates (bbox).")
            skipped_empty += 1
            continue

        candidates = data.iloc[candidate_idx]
        # precise predicate filter (pre-clip)
        if predicate != "intersects":
            mask = candidates.intersects(geom) if predicate == "intersects" else getattr(candidates, predicate)(geom)
        else:
            mask = candidates.intersects(geom)
        candidates = candidates.loc[mask]

        if candidates.empty:
            print(f"[{idx+1}/{total}] '{out_name}' -> no intersecting features.")
            skipped_empty += 1
            continue

        # Clip
        clipped = gpd.clip(candidates, geom)

        # Drop empties created by clipping
        clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notna()]
        if clipped.empty:
            print(f"[{idx+1}/{total}] '{out_name}' -> empty after clip.")
            skipped_empty += 1
            continue

        # Decide output path & driver
        # If out_dir is a directory, write one file per polygon
        if driver is None:
            # infer from a desired extension
            ext = out_extension if out_extension is not None else ".gpkg"
            path = out_dir / f"{out_name}{ext}"
            drv = None  # let fiona infer from extension
            layer_name = "clipped"
        else:
            # driver specified; pick extension accordingly if not given
            ext = {
                "GPKG": ".gpkg",
                "ESRI Shapefile": ".shp",
                "GEOJSON": ".geojson",
            }.get(driver.upper() if driver else "", ".gpkg")
            path = out_dir / f"{out_name}{ext}"
            drv = driver
            layer_name = "clipped" if drv.upper() == "GPKG" else None

        # Save
        save_kwargs = {}
        if drv is not None:
            save_kwargs["driver"] = drv
        if layer_name and (save_kwargs.get("driver", "") == "" or save_kwargs.get("driver", "").upper() == "GPKG" or path.suffix.lower() == ".gpkg"):
            clipped.to_file(path, layer=layer_name, **save_kwargs)
        else:
            clipped.to_file(path, **save_kwargs)

        written += 1
        print(f"[{idx+1}/{total}] Wrote {len(clipped)} feature(s) -> {path}")

    print(f"\nDone. Wrote {written} file(s). Skipped (empty/no-intersect): {skipped_empty}. Out dir: {out_dir}")


if __name__ == "__main__":
    clipping_vector = r"C:\L\OneDrive - Lichen\Documents\Projects\Salmon Cr\HAWS\rem_areas.gpkg"
    vector_to_clip = r"C:\L\OneDrive - Lichen\Documents\Projects\Salmon Cr\HAWS\transects_bendy_smooth_100.0m_100.0m.gpkg"

    out_dir = r"C:\L\OneDrive - Lichen\Documents\Projects\Salmon Cr\HAWS\transects_by_site"

    clip_and_save_each(
        clipping_vector=clipping_vector,
        vector_to_clip=vector_to_clip,
        out_dir=out_dir,
        name_field="region",   # default; change if your field differs
        clip_layer=None,           # set if your files have multiple layers
        vector_layer=None,
        driver=None,               # or "GPKG", "ESRI Shapefile", etc.
        out_extension=".gpkg",     # outputs one .gpkg per polygon
        predicate="intersects",
    )
