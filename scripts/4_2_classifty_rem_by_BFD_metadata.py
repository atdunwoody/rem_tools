from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence
from xml.sax.saxutils import escape as xml_escape

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from rasterio.warp import Resampling, reproject
from shapely.geometry import shape


# RGBA class colors used for both the raster color table and polygon QGIS style.
# Edit these if you want a different default ramp.
DEFAULT_CLASS_COLORS: tuple[tuple[int, int, int, int], ...] = (
    (8, 48, 107, 210),     # deep blue
    # (49, 130, 189, 210),   # blue
    (107, 174, 214, 210),  # light blue
    (49, 163, 84, 210),    # green
    (254, 178, 76, 210),   # yellow/orange
    (240, 59, 32, 210),    # red-orange, extra
    (189, 0, 38, 210),     # deep red, extra
)


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy scalar values."""

    def default(self, obj):  # noqa: D401
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _choose_int_dtype(max_class: int):
    """Pick a compact integer dtype for class raster."""
    if max_class <= 255:
        return rasterio.uint8
    if max_class <= 65535:
        return rasterio.uint16
    return rasterio.uint32


def _format_bin_label(low: Optional[float], high: Optional[float]) -> str:
    """
    Human-readable label for the proportion bin.
      low=None means (-inf, high]
      high=None means (low, +inf)
    """
    if low is None and high is not None:
        return f"<= {high:g}x"
    if low is not None and high is not None:
        return f"> {low:g}x to <= {high:g}x"
    if low is not None and high is None:
        return f"> {low:g}x"
    return "unclassified"


def _rgba_to_hex(rgba: Sequence[int]) -> str:
    """Convert an RGBA tuple to an RGB hex string for vector attributes."""
    r, g, b, *_ = [int(v) for v in rgba]
    return f"#{r:02X}{g:02X}{b:02X}"


def _rgba_to_qgis(rgba: Sequence[int]) -> str:
    """Convert RGBA tuple to the comma-delimited format used in QGIS QML."""
    r, g, b, a = [int(v) for v in rgba]
    return f"{r},{g},{b},{a}"


def _normalize_colors(
    n_classes: int,
    class_colors: Optional[Sequence[Sequence[int]]] = None,
) -> list[tuple[int, int, int, int]]:
    """Return one RGBA color per class."""
    source = list(class_colors) if class_colors is not None else list(DEFAULT_CLASS_COLORS)
    if len(source) < n_classes:
        raise ValueError(
            f"Need at least {n_classes} colors, but only {len(source)} were provided. "
            "Add colors to class_colors or DEFAULT_CLASS_COLORS."
        )

    colors: list[tuple[int, int, int, int]] = []
    for color in source[:n_classes]:
        if len(color) == 3:
            r, g, b = color
            a = 210
        elif len(color) == 4:
            r, g, b, a = color
        else:
            raise ValueError("Each color must be RGB or RGBA.")

        rgba = tuple(int(v) for v in (r, g, b, a))
        if any(v < 0 or v > 255 for v in rgba):
            raise ValueError("Color values must be 0 to 255.")
        colors.append(rgba)

    return colors


def _build_class_records(
    thresholds: Sequence[float],
    colors: Sequence[Sequence[int]],
) -> list[dict]:
    """Build one metadata record per class."""
    records: list[dict] = []
    for class_id, high in enumerate(thresholds, start=1):
        low = None if class_id == 1 else float(thresholds[class_id - 2])
        label = _format_bin_label(low, float(high))
        rgba = tuple(int(v) for v in colors[class_id - 1])

        records.append(
            {
                "class_id": class_id,
                "threshold_low_exclusive": low,
                "threshold_high_inclusive": float(high),
                "threshold_units": "multiples_of_bankfull_stage",
                "threshold_label": label,
                "color_rgba": list(rgba),
                "color_hex": _rgba_to_hex(rgba),
            }
        )

    return records


def _build_classification_metadata(
    *,
    rem_raster_path: str,
    bf_raster_path: Optional[str],
    bf_static_value: Optional[float],
    thresholds: Sequence[float],
    class_records: Sequence[dict],
    out_nodata: int,
    rem_units: str,
    bf_raster_units: str,
    classification_units: str,
    bf_unit_conversion_factor: float,
) -> dict:
    """Build metadata that is written to both raster and polygon outputs."""
    bf_source = (
        {"type": "raster", "path": bf_raster_path, "input_units": bf_raster_units}
        if bf_raster_path is not None
        else {"type": "static", "value": bf_static_value, "input_units": rem_units}
    )

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classification_name": "REM bankfull-stage multiples",
        "classification_expression": "ratio = REM / BF_stage",
        "rem_raster_path": rem_raster_path,
        "rem_units": rem_units,
        "bf_source": bf_source,
        "bf_unit_conversion_factor": bf_unit_conversion_factor if bf_raster_path is not None else 1.0,
        "classification_units": classification_units,
        "thresholds": [float(t) for t in thresholds],
        "threshold_units": "multiples_of_bankfull_stage",
        "class_records": list(class_records),
        "unclassified_rule": {
            "condition": f"ratio > {float(thresholds[-1]):g}",
            "output_value": out_nodata,
            "label": "unclassified / nodata",
        },
    }


def _write_raster_metadata_and_colormap(
    dst: rasterio.io.DatasetWriter,
    *,
    metadata: dict,
    class_records: Sequence[dict],
    out_nodata: int,
    write_colormap: bool = True,
) -> None:
    """Write classification metadata and a raster color table to the class GeoTIFF."""
    class_label_lookup = {
        int(r["class_id"]): str(r["threshold_label"])
        for r in class_records
    }
    class_color_lookup = {
        int(r["class_id"]): r["color_rgba"]
        for r in class_records
    }

    # Dataset-level tags. These are visible in GDAL, rasterio, and many GIS metadata panels.
    dst.update_tags(
        rem_bf_classification="REM / BF_stage",
        rem_bf_threshold_units="multiples_of_bankfull_stage",
        rem_bf_thresholds=json.dumps(metadata["thresholds"], cls=NumpyJSONEncoder),
        rem_bf_classes=json.dumps(class_label_lookup, cls=NumpyJSONEncoder),
        rem_bf_unclassified_rule=json.dumps(metadata["unclassified_rule"], cls=NumpyJSONEncoder),
        rem_bf_metadata_json=json.dumps(metadata, cls=NumpyJSONEncoder),
    )

    # Band-level tags. QGIS and GDAL can expose these under band metadata.
    dst.update_tags(
        1,
        class_names=json.dumps(class_label_lookup, cls=NumpyJSONEncoder),
        class_colors_rgba=json.dumps(class_color_lookup, cls=NumpyJSONEncoder),
        threshold_units="multiples_of_bankfull_stage",
    )

    if not write_colormap:
        return

    # Paletted GeoTIFF symbology. Most GIS software will read this directly for integer rasters.
    colormap = {
        out_nodata: (0, 0, 0, 0),
        **{
            int(r["class_id"]): tuple(int(v) for v in r["color_rgba"])
            for r in class_records
        },
    }

    try:
        dst.write_colormap(1, colormap)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] Could not write raster color table: {exc}")


def _build_qgis_polygon_qml(
    *,
    layer_name: str,
    class_records: Sequence[dict],
    category_field: str = "class_id",
) -> str:
    """Build a compact QGIS categorized polygon style."""
    categories = []
    symbols = []
    for idx, rec in enumerate(class_records):
        class_id = int(rec["class_id"])
        label = xml_escape(str(rec["threshold_label"]))
        color = _rgba_to_qgis(rec["color_rgba"])
        categories.append(
            f'<category render="true" symbol="{idx}" value="{class_id}" label="{label}"/>'
        )
        symbols.append(
            f'''
      <symbol alpha="1" clip_to_extent="1" force_rhr="0" name="{idx}" type="fill">
        <layer class="SimpleFill" enabled="1" locked="0" pass="0">
          <Option type="Map">
            <Option name="color" type="QString" value="{color}"/>
            <Option name="joinstyle" type="QString" value="bevel"/>
            <Option name="offset" type="QString" value="0,0"/>
            <Option name="offset_unit" type="QString" value="MM"/>
            <Option name="outline_color" type="QString" value="35,35,35,255"/>
            <Option name="outline_style" type="QString" value="solid"/>
            <Option name="outline_width" type="QString" value="0.15"/>
            <Option name="outline_width_unit" type="QString" value="MM"/>
            <Option name="style" type="QString" value="solid"/>
          </Option>
        </layer>
      </symbol>'''
        )

    return f'''<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34" styleCategories="AllStyleCategories">
  <renderer-v2 attr="{xml_escape(category_field)}" enableorderby="0" forceraster="0" symbollevels="0" type="categorizedSymbol">
    <categories>
      {''.join(categories)}
    </categories>
    <symbols>
      {''.join(symbols)}
    </symbols>
  </renderer-v2>
  <layername>{xml_escape(layer_name)}</layername>
  <layerGeometryType>2</layerGeometryType>
</qgis>
'''


def _write_qgis_style_sidecar(
    *,
    output_polygons_path: str,
    polygon_layer: str,
    qml: str,
) -> str:
    """Write a sidecar QML style as a fallback for QGIS."""
    gpkg = Path(output_polygons_path)
    qml_path = gpkg.with_name(f"{gpkg.stem}_{polygon_layer}.qml")
    qml_path.write_text(qml, encoding="utf-8")
    return str(qml_path)


def _write_qgis_style_to_gpkg(
    *,
    output_polygons_path: str,
    polygon_layer: str,
    qml: str,
) -> None:
    """
    Write the categorized polygon style into the GeoPackage layer_styles table.

    This is QGIS-specific, but it allows the style to travel with the GeoPackage.
    """
    if Path(output_polygons_path).suffix.lower() != ".gpkg":
        return

    with sqlite3.connect(output_polygons_path) as con:
        row = con.execute(
            "SELECT column_name FROM gpkg_geometry_columns WHERE table_name = ?",
            (polygon_layer,),
        ).fetchone()
        geometry_column = row[0] if row else "geom"

        con.execute(
            """
            CREATE TABLE IF NOT EXISTS layer_styles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                f_table_catalog TEXT,
                f_table_schema TEXT,
                f_table_name TEXT,
                f_geometry_column TEXT,
                styleName TEXT,
                styleQML TEXT,
                styleSLD TEXT,
                useAsDefault INTEGER,
                description TEXT,
                owner TEXT,
                ui TEXT,
                update_time DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        con.execute(
            "DELETE FROM layer_styles WHERE f_table_name = ? AND styleName = ?",
            (polygon_layer, "REM BF classes"),
        )
        con.execute(
            """
            INSERT INTO layer_styles (
                f_table_catalog,
                f_table_schema,
                f_table_name,
                f_geometry_column,
                styleName,
                styleQML,
                styleSLD,
                useAsDefault,
                description,
                owner,
                ui
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "",
                "",
                polygon_layer,
                geometry_column,
                "REM BF classes",
                qml,
                "",
                1,
                "Categorized REM/BF-stage threshold polygons",
                "",
                "",
            ),
        )
        con.commit()


def _write_gpkg_metadata(
    *,
    output_polygons_path: str,
    polygon_layer: str,
    metadata: dict,
) -> None:
    """Write classification metadata into OGC GeoPackage metadata tables."""
    if Path(output_polygons_path).suffix.lower() != ".gpkg":
        return

    metadata_json = json.dumps(metadata, cls=NumpyJSONEncoder, indent=2)

    with sqlite3.connect(output_polygons_path) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS gpkg_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                md_scope TEXT NOT NULL DEFAULT 'dataset',
                md_standard_uri TEXT NOT NULL,
                mime_type TEXT NOT NULL DEFAULT 'text/plain',
                metadata TEXT NOT NULL DEFAULT ''
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS gpkg_metadata_reference (
                reference_scope TEXT NOT NULL,
                table_name TEXT,
                column_name TEXT,
                row_id_value INTEGER,
                timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                md_file_id INTEGER NOT NULL,
                md_parent_id INTEGER,
                FOREIGN KEY (md_file_id) REFERENCES gpkg_metadata(id),
                FOREIGN KEY (md_parent_id) REFERENCES gpkg_metadata(id)
            )
            """
        )

        # Avoid accumulating duplicate metadata records if this is rerun against an existing file.
        existing_ids = [
            row[0]
            for row in con.execute(
                """
                SELECT md_file_id
                FROM gpkg_metadata_reference
                WHERE table_name = ?
                  AND reference_scope = 'table'
                """,
                (polygon_layer,),
            ).fetchall()
        ]
        if existing_ids:
            placeholders = ",".join("?" for _ in existing_ids)
            con.execute(
                f"DELETE FROM gpkg_metadata_reference WHERE md_file_id IN ({placeholders})",
                existing_ids,
            )
            con.execute(
                f"DELETE FROM gpkg_metadata WHERE id IN ({placeholders})",
                existing_ids,
            )

        cur = con.execute(
            """
            INSERT INTO gpkg_metadata (md_scope, md_standard_uri, mime_type, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (
                "dataset",
                "https://atdunwoody.local/rem-bankfull-stage-thresholds",
                "application/json",
                metadata_json,
            ),
        )
        md_id = int(cur.lastrowid)
        con.execute(
            """
            INSERT INTO gpkg_metadata_reference (
                reference_scope,
                table_name,
                column_name,
                row_id_value,
                md_file_id,
                md_parent_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("table", polygon_layer, None, None, md_id, None),
        )
        con.commit()


def classify_rem_by_bankfull(
    rem_raster_path: str,
    output_class_raster_path: str,
    output_polygons_path: str,
    *,
    # Choose ONE of these:
    bf_raster_path: Optional[str] = None,
    bf_static_value: Optional[float] = None,
    # Classification thresholds (proportions of BF)
    thresholds: Sequence[float] = (0.5, 1.0, 2.0),
    # Units / conversion settings
    rem_units: str = "ft",
    bf_raster_units: str = "m",
    classification_units: str = "ft",
    bf_unit_conversion_factor: float = 3.28084,
    # Symbology / metadata settings
    class_colors: Optional[Sequence[Sequence[int]]] = None,
    write_raster_colormap: bool = True,
    write_polygon_qgis_style: bool = True,
    write_polygon_qgis_sidecar: bool = True,
    # Raster / polygon output settings
    out_nodata: int = 0,
    polygon_driver: str = "GPKG",
    polygon_layer: str = "rem_bf_classes",
    dissolve_polygons: bool = True,
) -> None:
    """
    Classify a REM raster by how many multiples of bankfull stage the REM is above.

    Ratio = REM / BF

    thresholds=[t1, t2, ... tN] produces N classes (NO overflow class):
      class 1: ratio <= t1
      class 2: t1 < ratio <= t2
      ...
      class N: t{N-1} < ratio <= tN

    IMPORTANT: ratio > tN becomes nodata/unclassified (out_nodata).

    Nodata rules:
      - REM nodata -> output nodata
      - BF nodata -> output nodata (if using bf_raster_path)
      - BF <= 0 -> output nodata

    Outputs:
      1) Classified raster with threshold metadata tags and a color table.
      2) Polygon file with per-class threshold fields and, for GeoPackage outputs,
         embedded GeoPackage metadata and an embedded QGIS categorized style.
    """
    thresholds = [float(t) for t in thresholds]
    if len(thresholds) == 0:
        raise ValueError("thresholds must contain at least one value.")
    if sorted(thresholds) != list(thresholds):
        raise ValueError("thresholds must be sorted ascending (e.g., [0.5, 1, 2]).")
    if any(t < 0 for t in thresholds):
        raise ValueError("thresholds should be non-negative proportions.")

    if (bf_raster_path is None) == (bf_static_value is None):
        raise ValueError("Provide exactly one of bf_raster_path OR bf_static_value (not both).")
    if bf_static_value is not None and bf_static_value <= 0:
        raise ValueError("bf_static_value must be > 0.")
    if bf_raster_path is not None and bf_unit_conversion_factor <= 0:
        raise ValueError("bf_unit_conversion_factor must be > 0.")

    colors = _normalize_colors(len(thresholds), class_colors)
    class_records = _build_class_records(thresholds, colors)
    class_lookup = pd.DataFrame(class_records)

    classification_metadata = _build_classification_metadata(
        rem_raster_path=rem_raster_path,
        bf_raster_path=bf_raster_path,
        bf_static_value=bf_static_value,
        thresholds=thresholds,
        class_records=class_records,
        out_nodata=out_nodata,
        rem_units=rem_units,
        bf_raster_units=bf_raster_units,
        classification_units=classification_units,
        bf_unit_conversion_factor=bf_unit_conversion_factor,
    )

    print(f"[CLASSIFY] REM: {rem_raster_path}")
    if bf_raster_path:
        print(f"[CLASSIFY] BF raster: {bf_raster_path}")
        print(f"[CLASSIFY] BF conversion: {bf_raster_units} * {bf_unit_conversion_factor:g} -> {classification_units}")
    else:
        print(f"[CLASSIFY] BF static: {bf_static_value:g} {classification_units}")
    print(f"[CLASSIFY] Thresholds: {thresholds}")
    print(f"[OUT] Class raster: {output_class_raster_path}")
    print(f"[OUT] Polygons: {output_polygons_path} (layer={polygon_layer})")

    # -------------------------
    # Read REM
    # -------------------------
    with rasterio.open(rem_raster_path) as src_rem:
        rem = src_rem.read(1, masked=True)  # masked array
        rem_meta = src_rem.meta.copy()
        rem_crs = src_rem.crs
        rem_transform = src_rem.transform
        rem_shape = src_rem.shape

        # -------------------------
        # Get BF array on REM grid
        # -------------------------
        if bf_static_value is not None:
            bf = np.ma.masked_array(
                np.full(rem_shape, float(bf_static_value), dtype="float32"),
                mask=np.zeros(rem_shape, dtype=bool),
            )
        else:
            with rasterio.open(bf_raster_path) as src_bf:
                bf_nodata = src_bf.nodata
                bf_resampled = np.empty(rem_shape, dtype="float32")

                reproject(
                    source=rasterio.band(src_bf, 1),
                    destination=bf_resampled,
                    src_transform=src_bf.transform,
                    src_crs=src_bf.crs,
                    dst_transform=rem_transform,
                    dst_crs=rem_crs,
                    dst_resolution=src_rem.res,
                    resampling=Resampling.bilinear,
                    dst_nodata=bf_nodata,
                )

                if bf_nodata is None:
                    # If BF raster has no explicit nodata, treat NaNs as nodata.
                    bf = np.ma.masked_invalid(bf_resampled)
                else:
                    bf = np.ma.masked_equal(bf_resampled, bf_nodata)

                # Convert BF raster to the same units as the REM before classifying.
                bf = bf * bf_unit_conversion_factor

    # -------------------------
    # Build valid mask
    # -------------------------
    rem_data = rem.data.astype("float32", copy=False)
    bf_data = bf.data.astype("float32", copy=False)

    valid = (~rem.mask) & (~bf.mask) & np.isfinite(rem_data) & np.isfinite(bf_data) & (bf_data > 0)

    # Ratio = REM / BF
    ratio = np.full(rem_shape, np.nan, dtype="float32")
    ratio[valid] = rem_data[valid] / bf_data[valid]

    # -------------------------
    # Classify (values > last threshold become NODATA)
    # -------------------------
    n_classes = len(thresholds)
    dtype = _choose_int_dtype(n_classes)

    cls = np.full(rem_shape, out_nodata, dtype=np.dtype(dtype))

    # Class 1: ratio <= t1
    t0 = thresholds[0]
    cls[(ratio <= t0) & valid] = 1

    # Middle (and last) classes: (t{i-1}, t{i}]
    for i in range(1, len(thresholds)):
        lo = thresholds[i - 1]
        hi = thresholds[i]
        cls[(ratio > lo) & (ratio <= hi) & valid] = i + 1

    # IMPORTANT: do not classify ratio > thresholds[-1].
    # Those remain out_nodata (unclassified).

    # -------------------------
    # Write class raster
    # -------------------------
    out_meta = rem_meta.copy()
    out_meta.update(
        dtype=dtype,
        count=1,
        nodata=out_nodata,
        compress="lzw",
    )

    out_dir = os.path.dirname(output_class_raster_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with rasterio.open(output_class_raster_path, "w", **out_meta) as dst:
        dst.write(cls, 1)
        _write_raster_metadata_and_colormap(
            dst,
            metadata=classification_metadata,
            class_records=class_records,
            out_nodata=out_nodata,
            write_colormap=write_raster_colormap,
        )

    print(f"[✔] Wrote class raster ({n_classes} classes). Unclassified: ratio > {thresholds[-1]:g}x -> nodata")
    if write_raster_colormap:
        print("[✔] Added raster metadata tags and a raster color table")

    # -------------------------
    # Polygonize
    # -------------------------
    # Mask nodata and also ignore background 0.
    mask = cls != out_nodata

    geoms = []
    vals = []
    for geom, val in shapes(cls, mask=mask, transform=rem_transform):
        v = int(val)
        if v == out_nodata:
            continue
        geoms.append(shape(geom))
        vals.append(v)

    if len(geoms) == 0:
        raise RuntimeError("No polygons were created (all nodata or empty mask).")

    gdf = gpd.GeoDataFrame({"class_id": vals}, geometry=geoms, crs=rem_crs)

    # Add per-feature threshold metadata. These fields are useful even outside QGIS.
    gdf = gdf.merge(class_lookup, on="class_id", how="left")
    gdf["Proportion of BF stage"] = gdf["threshold_label"]
    gdf["all_thresholds"] = json.dumps(thresholds, cls=NumpyJSONEncoder)
    gdf["threshold_scheme"] = (
        "REM / BF_stage; thresholds="
        + json.dumps(thresholds, cls=NumpyJSONEncoder)
        + f"; ratio > {thresholds[-1]:g} is nodata"
    )

    # GeoPackage handles lists poorly, so store color_rgba as text for the vector table.
    gdf["color_rgba"] = gdf["color_rgba"].apply(lambda x: json.dumps(x, cls=NumpyJSONEncoder))

    if dissolve_polygons:
        # Merge polygons by class_id and preserve the class metadata fields.
        keep_cols = [
            "class_id",
            "threshold_low_exclusive",
            "threshold_high_inclusive",
            "threshold_units",
            "threshold_label",
            "color_rgba",
            "color_hex",
            "Proportion of BF stage",
            "all_thresholds",
            "threshold_scheme",
        ]
        attrs = gdf[keep_cols].drop_duplicates("class_id")
        gdf = gdf.dissolve(by="class_id", as_index=False)[["class_id", "geometry"]].merge(
            attrs,
            on="class_id",
            how="left",
        )

    # Write polygon output.
    out_poly_dir = os.path.dirname(output_polygons_path)
    if out_poly_dir:
        os.makedirs(out_poly_dir, exist_ok=True)

    gdf.to_file(output_polygons_path, layer=polygon_layer, driver=polygon_driver)

    if polygon_driver.upper() == "GPKG":
        _write_gpkg_metadata(
            output_polygons_path=output_polygons_path,
            polygon_layer=polygon_layer,
            metadata=classification_metadata,
        )

    if write_polygon_qgis_style:
        qml = _build_qgis_polygon_qml(
            layer_name=polygon_layer,
            class_records=class_records,
            category_field="class_id",
        )
        if polygon_driver.upper() == "GPKG":
            _write_qgis_style_to_gpkg(
                output_polygons_path=output_polygons_path,
                polygon_layer=polygon_layer,
                qml=qml,
            )
            print("[✔] Added embedded QGIS polygon style to GeoPackage")
        if write_polygon_qgis_sidecar:
            qml_path = _write_qgis_style_sidecar(
                output_polygons_path=output_polygons_path,
                polygon_layer=polygon_layer,
                qml=qml,
            )
            print(f"[✔] Wrote QGIS sidecar style: {qml_path}")

    print(f"[✔] Wrote polygons: {output_polygons_path} (features={len(gdf)})")


if __name__ == "__main__":
    rem_raster = r"C:\L\Lichen\Lichen - Documents\Projects\20260003_Owens-Snipe Assessment (UCSWCD)\07_GIS\1_Analysis\Stream Network Analysis\REM\Final REM\HAWS_REM_3ft_600idw_varonoi.tif"

    # OPTION A: BF raster (reprojected to REM).
    # Default assumption: BF raster is in meters and REM is in feet.
    bf_raster = r"C:\L\Lichen\Lichen - Documents\Projects\20260003_Owens-Snipe Assessment (UCSWCD)\07_GIS\1_Analysis\Stream Network Analysis\REM\Expanded network\Voronoi Test v2\BF_depth_Legg_m.tif"

    # OPTION B: Static BF stage, already in the same units as REM. Uncomment to use.
    # bf_static = 2.22  # ft from StreamStats average BF Depth

    out_class_raster = r"C:\L\Lichen\Lichen - Documents\Projects\20260003_Owens-Snipe Assessment (UCSWCD)\07_GIS\1_Analysis\Stream Network Analysis\REM\Final REM\HAWS_REM_3ft_600idw_varonoi_classified_v2.tif"
    out_polygons = r"C:\L\Lichen\Lichen - Documents\Projects\20260003_Owens-Snipe Assessment (UCSWCD)\07_GIS\1_Analysis\Stream Network Analysis\REM\Final REM\HAWS_REM_3ft_600idw_varonoi_polygons_v2.gpkg"

    thresholds = [0.5, 1, 2, 3, 4, 6]  # Anything > 3x becomes nodata/unclassified.

    classify_rem_by_bankfull(
        rem_raster_path=rem_raster,
        output_class_raster_path=out_class_raster,
        output_polygons_path=out_polygons,
        bf_raster_path=bf_raster,     # Use BF raster.
        # bf_static_value=bf_static,  # Or use static BF.
        thresholds=thresholds,
        polygon_layer="floodplain",
        dissolve_polygons=False,
    )
