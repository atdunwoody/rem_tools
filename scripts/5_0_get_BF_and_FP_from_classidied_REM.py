from __future__ import annotations

from pathlib import Path
import warnings

import geopandas as gpd
import pandas as pd
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union


# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------
INPUT_GPKG = Path(
    r"C:\L\Lichen\Lichen - Documents\Projects\20260005_CTCR_OmakCreek\07_GIS\1_Analysis\Initial Geomorph\Streams\burn DEM\HAWS_REM_3ft_600idw_polygons_6xbfd_beechie.gpkg"
)

# Set to None to read the first layer in the GeoPackage.
INPUT_LAYER: str | None = None
CLASS_FIELD = "class_id"

# Include source features whose class_id is less than or equal to these values.
BANKFULL_MAX_CLASS_ID = 2
FLOODPLAIN_MAX_CLASS_ID = 3
VALLEY_MAX_CLASS_ID = 5

# Distance used to shrink and then expand each dissolved polygon. This value is
# in the input CRS units. For example, use 3.0 for 3 feet in a feet-based CRS.
CLEAN_DISTANCE = 5.0

OUTPUT_FOLDER = r"C:\L\Lichen\Lichen - Documents\Projects\20260005_CTCR_OmakCreek\07_GIS\1_Analysis\Initial Geomorph\Streams\burn DEM\Domains"

BANKFULL_OUTPUT = Path(OUTPUT_FOLDER, "omak_creek_bankfull_polygon.gpkg")
FLOODPLAIN_OUTPUT = Path(OUTPUT_FOLDER, "omak_creek_floodplain_polygon.gpkg")
VALLEY_OUTPUT = Path(OUTPUT_FOLDER, "omak_creek_valley_polygon.gpkg")


def polygonal_parts(geometry):
    """Return only Polygon and MultiPolygon parts from a geometry."""
    if geometry is None or geometry.is_empty:
        return None
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        parts = [
            part
            for part in geometry.geoms
            if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty
        ]
        return unary_union(parts) if parts else None
    return None


def dissolve_geometries(geometries):
    """Dissolve geometries, supporting both current and older GeoPandas."""
    try:
        return geometries.union_all()
    except AttributeError:
        return geometries.unary_union


def build_polygon(
    source: gpd.GeoDataFrame,
    max_class_id: float,
    polygon_type: str,
    clean_distance: float,
) -> gpd.GeoDataFrame:
    """Select, dissolve, and clean one threshold-based polygon."""
    selected = source.loc[source[CLASS_FIELD] <= max_class_id].copy()
    if selected.empty:
        raise ValueError(
            f"No features have {CLASS_FIELD} <= {max_class_id} "
            f"for the {polygon_type} polygon."
        )

    geometry = polygonal_parts(dissolve_geometries(selected.geometry))
    if geometry is None or geometry.is_empty:
        raise ValueError(f"The dissolved {polygon_type} geometry is empty.")

    if clean_distance > 0:
        # Morphological opening removes isolated and narrow polygon artifacts.
        geometry = geometry.buffer(-clean_distance)
        if geometry.is_empty:
            raise ValueError(
                f"Shrinking the {polygon_type} polygon by {clean_distance} "
                "removed the entire geometry. Use a smaller CLEAN_DISTANCE."
            )
        geometry = polygonal_parts(geometry.buffer(clean_distance))

    if geometry is None or geometry.is_empty:
        raise ValueError(f"The cleaned {polygon_type} geometry is empty.")

    return gpd.GeoDataFrame(
        {
            "polygon_type": [polygon_type],
            "max_class_id": [max_class_id],
            "clean_distance": [clean_distance],
        },
        geometry=[geometry],
        crs=source.crs,
    )


def write_polygon(
    polygon: gpd.GeoDataFrame,
    output_path: Path,
    layer_name: str,
) -> None:
    """Write one polygon layer to a GeoPackage."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    polygon.to_file(output_path, layer=layer_name, driver="GPKG", mode="w")
    print(f"Saved {layer_name}: {output_path}")


def main() -> None:
    if CLEAN_DISTANCE < 0:
        raise ValueError("CLEAN_DISTANCE must be zero or greater.")

    source = gpd.read_file(INPUT_GPKG, layer=INPUT_LAYER)
    if source.empty:
        raise ValueError(f"The input layer contains no features: {INPUT_GPKG}")
    if source.crs is None:
        raise ValueError("The input layer has no CRS.")
    if not source.crs.is_projected:
        raise ValueError(
            "The input layer must use a projected CRS before distance-based "
            "buffering. Reproject it and rerun the script."
        )
    if CLASS_FIELD not in source.columns:
        raise KeyError(
            f"Field {CLASS_FIELD!r} was not found. Available fields: "
            f"{list(source.columns)}"
        )
    
    class_values = pd.to_numeric(source[CLASS_FIELD], errors="coerce")
    invalid_count = int(class_values.isna().sum() - source[CLASS_FIELD].isna().sum())
    if invalid_count:
        warnings.warn(
            f"Ignoring {invalid_count} feature(s) with nonnumeric {CLASS_FIELD} values.",
            stacklevel=2,
        )
    source[CLASS_FIELD] = class_values
    source = source.loc[source.geometry.notna() & ~source.geometry.is_empty].copy()

    bankfull = build_polygon(
        source=source,
        max_class_id=BANKFULL_MAX_CLASS_ID,
        polygon_type="bankfull",
        clean_distance=CLEAN_DISTANCE,
    )
    floodplain = build_polygon(
        source=source,
        max_class_id=FLOODPLAIN_MAX_CLASS_ID,
        polygon_type="floodplain",
        clean_distance=CLEAN_DISTANCE,
    )
    valley = build_polygon(
        source=source,
        max_class_id=VALLEY_MAX_CLASS_ID,
        polygon_type="valley",
        clean_distance=CLEAN_DISTANCE,
    )

    write_polygon(bankfull, BANKFULL_OUTPUT, "bankfull_polygon")
    write_polygon(floodplain, FLOODPLAIN_OUTPUT, "floodplain_polygon")
    write_polygon(valley, VALLEY_OUTPUT, "valley_polygon")


if __name__ == "__main__":
    main()
