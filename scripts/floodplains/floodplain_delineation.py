from pathlib import Path
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union
import geopandas as gpd


def raster_lt_threshold_to_polygon(
    raster_path,
    threshold=6.5,
    out_path=None,
    dissolve=True,
):
    raster_path = Path(raster_path)

    if out_path is None:
        out_path = raster_path.with_suffix("").as_posix() + f"_lt{str(threshold).replace('.', 'p')}.gpkg"
    out_path = Path(out_path)

    with rasterio.open(raster_path) as src:
        band = src.read(1, masked=True)  # masked array using nodata
        transform = src.transform
        crs = src.crs

        # True where raster < threshold and not nodata
        mask = (band < threshold) & (~band.mask)

        if not mask.any():
            raise ValueError("No pixels found below the threshold; nothing to polygonize.")

        # Polygonize: convert mask to polygons; value 1 = below threshold
        mask_uint8 = mask.astype("uint8")
        polygons = []
        for geom, val in shapes(mask_uint8, mask=mask, transform=transform):
            if val == 1:
                polygons.append(shape(geom))

    if not polygons:
        raise ValueError("No polygons were created from the mask.")

    # Optionally dissolve to a single (multi)polygon covering the entire area
    if dissolve:
        unioned = unary_union(polygons)
        geoms = [unioned]
    else:
        geoms = polygons

    gdf = gpd.GeoDataFrame({"id": range(len(geoms))}, geometry=geoms, crs=crs)

    # Layer name from output filename
    layer_name = out_path.stem

    gdf.to_file(out_path, layer=layer_name, driver="GPKG")
    print(f"Saved polygon(s) to: {out_path} (layer='{layer_name}')")


if __name__ == "__main__":
    raster = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\REMs\REM\HAWS_REM_1m.tif"
    output = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\REMs\jc_floodplain_10m.gpkg"
    print(f"Deineating floodplains in {raster}...")
    raster_lt_threshold_to_polygon(raster, threshold=10, out_path=output)