from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.errors import WindowError
from rasterio.features import geometry_mask, geometry_window
from shapely.geometry import mapping


# -----------------------------------------------------------------------------
# User inputs
# -----------------------------------------------------------------------------
LINE_GPKG = Path(
    r"C:\L\Lichen\Lichen - Documents\Projects\20240006.1_Willamette Cove (MFA)"
    r"\04_Analysis\BSTEM\Transect Elevations\priority transects.gpkg"
)

RASTER = Path(
    r"C:\L\Lichen\Lichen - Documents\Projects\20240006.1_Willamette Cove (MFA)"
    r"\04_Analysis\BSTEM\Received Data\Modeling Results\Spring Flow"
    r"\EC_SpringFlow_Shear Stress (11SEP2025 17 00 00).tif"
)

TRANSECT_FIELD = "transect"
RASTER_BAND = 1

OUTPUT_CSV = LINE_GPKG.with_name("priority_transects_raster_min_max.csv")
OUTPUT_GPKG = LINE_GPKG.with_name("priority_transects_raster_min_max.gpkg")
OUTPUT_LAYER = "transect_raster_stats"


def get_line_cell_values(
    src: rasterio.io.DatasetReader,
    geometry,
    band: int = 1,
) -> np.ndarray:
    """Return valid values from all raster cells touched by a line."""
    if geometry is None or geometry.is_empty:
        return np.array([], dtype=float)

    try:
        # Padding ensures that perfectly horizontal or vertical lines located
        # on a cell boundary still produce a nonzero read window.
        window = geometry_window(
            src,
            [mapping(geometry)],
            pad_x=0.5,
            pad_y=0.5,
            north_up=True,
        )
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
    except WindowError:
        return np.array([], dtype=float)

    data = src.read(band, window=window, masked=True)
    if data.size == 0:
        return np.array([], dtype=float)

    touched = geometry_mask(
        [mapping(geometry)],
        out_shape=data.shape,
        transform=src.window_transform(window),
        invert=True,
        all_touched=True,
    )

    valid = touched & ~np.ma.getmaskarray(data)
    values = np.asarray(data.data[valid], dtype=float)
    return values[np.isfinite(values)]


def summarize_raster_along_lines(
    line_gpkg: Path,
    raster_path: Path,
    transect_field: str,
    output_csv: Path,
    output_gpkg: Path,
    output_layer: str,
    raster_band: int = 1,
) -> pd.DataFrame:
    """Calculate, report, and save raster minimum and maximum for each line."""
    lines = gpd.read_file(line_gpkg)

    if transect_field not in lines.columns:
        raise KeyError(
            f"Field {transect_field!r} was not found. Available fields: "
            f"{', '.join(str(column) for column in lines.columns)}"
        )
    if lines.crs is None:
        raise ValueError("The line layer has no defined coordinate reference system.")

    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError("The raster has no defined coordinate reference system.")
        if not 1 <= raster_band <= src.count:
            raise ValueError(
                f"Raster band {raster_band} is invalid; raster has {src.count} band(s)."
            )

        sample_lines = lines.to_crs(src.crs)
        records: list[dict[str, object]] = []

        for transect, geometry in zip(
            sample_lines[transect_field], sample_lines.geometry, strict=True
        ):
            values = get_line_cell_values(src, geometry, band=raster_band)
            records.append(
                {
                    transect_field: transect,
                    "raster_min": float(np.min(values)) if values.size else np.nan,
                    "raster_max": float(np.max(values)) if values.size else np.nan,
                    "raster_mean": float(np.mean(values)) if values.size else np.nan,
                    "cell_count": int(values.size),
                }
            )

    results = pd.DataFrame.from_records(records)

    # Assign by row order so duplicate transect labels remain separate features.
    output_lines = lines.copy()
    output_lines["raster_min"] = results["raster_min"].to_numpy()
    output_lines["raster_max"] = results["raster_max"].to_numpy()
    output_lines["raster_mean"] = results["raster_mean"].to_numpy()
    output_lines["cell_count"] = results["cell_count"].to_numpy()

    results.to_csv(output_csv, index=False)
    output_lines.to_file(output_gpkg, layer=output_layer, driver="GPKG")

    print("\nRaster values along each transect")
    print(results.to_string(index=False, na_rep="No valid cells"))
    print(f"\nCSV:  {output_csv}")
    print(f"GPKG: {output_gpkg}")

    empty_count = int((results["cell_count"] == 0).sum())
    if empty_count:
        print(
            f"Warning: {empty_count} line(s) did not intersect any valid raster cells."
        )

    return results


if __name__ == "__main__":
    summarize_raster_along_lines(
        line_gpkg=LINE_GPKG,
        raster_path=RASTER,
        transect_field=TRANSECT_FIELD,
        output_csv=OUTPUT_CSV,
        output_gpkg=OUTPUT_GPKG,
        output_layer=OUTPUT_LAYER,
        raster_band=RASTER_BAND,
    )
