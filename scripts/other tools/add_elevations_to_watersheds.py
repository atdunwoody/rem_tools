import geopandas as gpd
import os

def assign_min_elev_to_watersheds(
    min_points_gpkg: str,
    watersheds_gpkg: str,
    output_gpkg: str,
    points_layer: str = None,
    watersheds_layer: str = None,
    output_layer: str = "watersheds_with_elev"
):
    # 1. read
    pts = gpd.read_file(min_points_gpkg, layer=points_layer)
    ws  = gpd.read_file(watersheds_gpkg, layer=watersheds_layer)

    # 2. turn the DataFrame index (which corresponds to the GPKG FID) into a column named 'fid'
    pts = pts.reset_index().rename(columns={'index': 'fid'})

    # 3. merge the elevation into ws
    ws = ws.merge(
        pts[['fid', 'elevation']],
        left_on='WS_ID',
        right_on='fid',
        how='left'
    ).drop(columns=['fid'])

    # 4. write out
    ws.to_file(output_gpkg, layer=output_layer, driver="GPKG")


watersheds_file = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\Streams Clipped to LiDAR\10 m spacing\GRMW_watersheds_dissolved.gpkg"
min_points_file = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Documents\Projects\Atlas\Streams Clipped to LiDAR\10 m spacing\min_elev_points_100m.gpkg"
output_file     = os.path.join(
    os.path.dirname(min_points_file),
    "GRMW_watersheds_dissolved_with_min_elev.gpkg"
)

assign_min_elev_to_watersheds(
    min_points_gpkg=min_points_file,
    watersheds_gpkg=watersheds_file,
    output_gpkg=output_file
)
