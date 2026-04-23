from pathlib import Path

import geopandas as gpd
import simplekml
from shapely.geometry import (
    Polygon,
    MultiPolygon,
    GeometryCollection,
)
from shapely.ops import orient

try:
    from shapely import make_valid
except ImportError:
    make_valid = None


# ---------------------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------------------
input_vector = r"C:\L\Lichen\Lichen - Documents\Projects\20240010_RoaringCr Des (CCD)\07_GIS\Analysis\Entiat REM\Entiat_HAWS_REM_3ft_17_classes.gpkg"
layer_name = None
class_field = "ClassRange"
output_kml = r"C:\L\Lichen\Lichen - Documents\Projects\20240010_RoaringCr Des (CCD)\07_GIS\Analysis\Entiat REM\Exports\Entiat_HAWS_REM_more_breaks_v1.kml"
name_field = "ClassRange"


# ---------------------------------------------------------------------
# CLASS COLORS - 12 class
# ---------------------------------------------------------------------
# CLASS_HEX = {
#     "-10-0": "#8f9bcc",
#     "0-1": "#b8c2ee",
#     "1-1.5": "#dee1ee",
#     "1.5-2": "#ecf1ec",
#     "2-3": "#cddacc",
#     "3-4": "#afc4af",
#     "4-5": "#9bb59a",
#     "5-6": "#fffad9",
#     "6-7": "#ffebcf",
#     "7-10": "#ffceb8",
#     "10-15": "#d48d7a",
# }

# CLASS_DRAW_ORDER = [
#     "10-15",
#     "7-10",
#     "6-7",
#     "5-6",
#     "4-5",
#     "3-4",
#     "2-3",
#     "1.5-2",
#     "1-1.5",
#     "0-1",
#     "-10-0",
# ]

# ---------------------------------------------------------------------
# CLASS COLORS - 17 class
# ---------------------------------------------------------------------
CLASS_HEX = {
    "-10-0":  "#9aa5d9",
    "0-1":    "#c7cef5",
    "1-1.5":  "#f2f3f8",
    "1.5-2":  "#eef3ee",
    "2-3":    "#d7e0d6",
    "3-4":    "#becdbd",
    "4-5":    "#afc3ae",
    "5-6":    "#f7f3d8",
    "6-7":    "#f6e7cf",
    "7-8":    "#f6dec7",
    "8-9":    "#f3cfbe",
    "9-10":   "#f1c0b5",
    "10-11":  "#edb0a9",
    "11-12":  "#e59f98",
    "12-13":  "#dc8f89",
    "13-14":  "#d27e79",
    "14-15":  "#c96f6f",
}

CLASS_DRAW_ORDER = [
    "-10-0",
    "0-1",
    "1-1.5",
    "1.5-2",
    "2-3",
    "3-4",
    "4-5",
    "5-6",
    "6-7",
    "7-8",
    "8-9",
    "9-10",
    "10-11",
    "11-12",
    "12-13",
    "13-14",
    "14-15",
]
# first written = bottom, last written = top

DRAW_PRIORITY = {
    class_name: i
    for i, class_name in enumerate(CLASS_DRAW_ORDER)
}

DEFAULT_HEX = "#8f9bcc"
DEFAULT_CLASS = "Other"
DEFAULT_DRAW_PRIORITY = -1


def hex_to_kml_color(hex_color: str, alpha: int = 255) -> str:
    """
    Convert #RRGGBB to KML color string AABBGGRR.
    """
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        raise ValueError(f"Expected 6-digit hex color, got: {hex_color}")
    rr = hex_color[0:2]
    gg = hex_color[2:4]
    bb = hex_color[4:6]
    aa = f"{alpha:02x}"
    return f"{aa}{bb}{gg}{rr}"


print("Class write order and KML colors (first=bottom, last=top):")
for class_name in CLASS_DRAW_ORDER:
    hex_color = CLASS_HEX.get(class_name, DEFAULT_HEX)
    kml_color = hex_to_kml_color(hex_color, alpha=255)
    print(
        f"  {class_name}: hex={hex_color}, kml={kml_color}, "
        f"write_order={DRAW_PRIORITY[class_name]}"
    )


def normalize_class_value(value) -> str:
    """
    Normalize class text to improve matching robustness.
    """
    if value is None:
        return ""

    v = str(value).strip()
    v = v.replace("–", "-").replace("—", "-").replace("−", "-")
    v = v.replace(" ", "")
    return v


def canonicalize_class_value(value) -> str:
    """
    Return the canonical class label used by CLASS_HEX/styles.
    """
    v = normalize_class_value(value)
    if v in CLASS_HEX:
        return v
    return DEFAULT_CLASS


def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    KML expects lon/lat WGS84.
    """
    if gdf.crs is None:
        raise ValueError("Input layer has no CRS. Define it before exporting to KML.")
    return gdf.to_crs(4326)


def coords_2d(seq):
    """
    Drop Z if present.
    """
    return [(x, y) for x, y, *rest in seq]


def repair_geometry(geom):
    """
    Repair invalid geometry where possible.
    """
    if geom is None or geom.is_empty:
        return geom

    if make_valid is not None:
        try:
            return make_valid(geom)
        except Exception:
            pass

    try:
        return geom.buffer(0)
    except Exception:
        return geom


def orient_polygonal_geometry(geom):
    """
    Orient polygon rings consistently for cleaner KML output.
    """
    if geom is None or geom.is_empty:
        return geom

    if isinstance(geom, Polygon):
        return orient(geom, sign=1.0)

    if isinstance(geom, MultiPolygon):
        return MultiPolygon([orient(part, sign=1.0) for part in geom.geoms])

    if isinstance(geom, GeometryCollection):
        parts = []
        for part in geom.geoms:
            oriented = orient_polygonal_geometry(part)
            if oriented is not None and not oriented.is_empty:
                parts.append(oriented)
        return GeometryCollection(parts)

    return geom


def extract_polygonal(geom):
    """
    Keep only polygonal parts after make_valid, since repair can return collections.
    """
    if geom is None or geom.is_empty:
        return None

    if isinstance(geom, Polygon):
        return geom

    if isinstance(geom, MultiPolygon):
        return geom

    if isinstance(geom, GeometryCollection):
        polys = []
        for part in geom.geoms:
            if isinstance(part, Polygon):
                polys.append(part)
            elif isinstance(part, MultiPolygon):
                polys.extend(list(part.geoms))

        if len(polys) == 0:
            return None
        if len(polys) == 1:
            return polys[0]
        return MultiPolygon(polys)

    return None


def build_styles() -> dict[str, simplekml.Style]:
    """
    Build one reusable KML style per class.
    Polygon-only styling.
    """
    styles = {}

    all_classes = CLASS_DRAW_ORDER + [DEFAULT_CLASS]
    for class_name in all_classes:
        hex_color = CLASS_HEX.get(class_name, DEFAULT_HEX)

        style = simplekml.Style()
        style.polystyle.color = hex_to_kml_color(hex_color, alpha=255)
        style.polystyle.fill = 1
        style.polystyle.outline = 0

        styles[class_name] = style

    return styles


def add_polygon(container, geom, feature_name: str, style: simplekml.Style):
    """
    Add polygon or multipolygon geometry to the KML document.
    """
    if geom is None or geom.is_empty:
        return

    if isinstance(geom, Polygon):
        polygon_kwargs = {
            "name": feature_name,
            "outerboundaryis": coords_2d(geom.exterior.coords),
        }

        if len(geom.interiors) > 0:
            polygon_kwargs["innerboundaryis"] = [
                coords_2d(ring.coords) for ring in geom.interiors
            ]

        f = container.newpolygon(**polygon_kwargs)
        f.style = style
        f.tessellate = 1

    elif isinstance(geom, MultiPolygon):
        for i, part in enumerate(geom.geoms, start=1):
            polygon_kwargs = {
                "name": f"{feature_name}_{i}",
                "outerboundaryis": coords_2d(part.exterior.coords),
            }

            if len(part.interiors) > 0:
                polygon_kwargs["innerboundaryis"] = [
                    coords_2d(ring.coords) for ring in part.interiors
                ]

            f = container.newpolygon(**polygon_kwargs)
            f.style = style
            f.tessellate = 1


def prepare_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Clean and standardize geometries for stable polygon KML export.
    """
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    gdf = ensure_wgs84(gdf)

    gdf["geometry"] = gdf.geometry.apply(repair_geometry)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    gdf["geometry"] = gdf.geometry.apply(extract_polygonal)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    gdf["geometry"] = gdf.geometry.apply(orient_polygonal_geometry)

    gdf = gdf.explode(index_parts=True, ignore_index=True)

    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    return gdf


def export_styled_kml(
    input_vector: str,
    output_kml: str,
    class_field: str = "ClassRange",
    layer_name: str | None = None,
    name_field: str | None = None,
):
    read_kwargs = {}
    if layer_name is not None:
        read_kwargs["layer"] = layer_name

    gdf = gpd.read_file(input_vector, **read_kwargs)

    if class_field not in gdf.columns:
        raise ValueError(
            f"Field '{class_field}' not found. Available fields: {list(gdf.columns)}"
        )

    gdf = prepare_geometries(gdf)

    gdf["_raw_class"] = gdf[class_field]
    gdf["_normalized_class"] = gdf[class_field].apply(normalize_class_value)
    gdf["_class"] = gdf[class_field].apply(canonicalize_class_value)
    gdf["_draw_priority"] = gdf["_class"].map(DRAW_PRIORITY).fillna(DEFAULT_DRAW_PRIORITY)
    gdf["_feature_order"] = range(len(gdf))

    mismatch_mask = gdf["_class"] == DEFAULT_CLASS
    if mismatch_mask.any():
        mismatch_df = (
            gdf.loc[mismatch_mask, ["_raw_class", "_normalized_class"]]
            .fillna("<NULL>")
            .astype(str)
            .value_counts()
            .reset_index(name="count")
        )

        print("\nUnmatched class values mapped to 'Other':")
        for _, rec in mismatch_df.iterrows():
            print(
                f"  raw={rec['_raw_class']!r}, "
                f"normalized={rec['_normalized_class']!r}, "
                f"count={rec['count']}"
            )

    gdf = gdf.sort_values(
        by=["_draw_priority", "_feature_order"],
        ascending=[True, True],
        kind="stable",
    ).copy()

    kml = simplekml.Kml()
    kml.document.name = Path(input_vector).stem

    styles = build_styles()
    root_name = Path(input_vector).stem

    for idx, row in gdf.iterrows():
        class_value = row["_class"]
        style = styles.get(class_value, styles[DEFAULT_CLASS])

        if name_field and name_field in row and row[name_field] is not None:
            feature_name = str(row[name_field])
        else:
            feature_name = f"{root_name}_{idx}"

        add_polygon(kml, row.geometry, feature_name, style)

    output_lower = output_kml.lower()
    if output_lower.endswith(".kmz"):
        kml.savekmz(output_kml)
    else:
        kml.save(output_kml)

    print(f"\nSaved: {output_kml}")


if __name__ == "__main__":
    export_styled_kml(
        input_vector=input_vector,
        output_kml=output_kml,
        class_field=class_field,
        layer_name=layer_name,
        name_field=name_field,
    )