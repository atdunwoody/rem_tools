import numpy as np
import rasterio
from rasterio.transform import xy
from scipy.ndimage import uniform_filter, maximum_filter
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path


def smooth_3x3_mean(arr, nodata=None):
    """
    3x3 mean filter that respects nodata.
    """
    from scipy.ndimage import convolve

    kernel = np.ones((3, 3), dtype="float32")

    if nodata is None:
        # Simple uniform filter
        return uniform_filter(arr, size=3, mode="nearest")

    # Mask valid data
    valid = arr != nodata
    arr_filled = np.where(valid, arr, 0.0)

    # Sum and count in the neighborhood
    sum_vals = convolve(arr_filled, kernel, mode="nearest")
    count_vals = convolve(valid.astype("float32"), kernel, mode="nearest")

    with np.errstate(divide="ignore", invalid="ignore"):
        smoothed = sum_vals / count_vals

    # Where no valid neighbors, set to nodata
    smoothed[~np.isfinite(smoothed)] = nodata
    smoothed[~valid] = nodata

    return smoothed


def detect_focalflow_like_peaks(
    arr,
    nodata=None,
    min_height_diff=0.3,
    min_outward_dirs=6,
):
    """
    FocalFlow-like peak detector on a smoothed DEM.

    Parameters
    ----------
    arr : 2D np.ndarray
        Smoothed DSM.
    nodata : scalar or None
        Nodata value. If None, NaNs are used.
    min_height_diff : float
        Minimum elevation difference (center - neighbor) required for at least
        one neighbor, to avoid picking flat plateaus.
    min_outward_dirs : int
        Minimum number of neighbors that must be lower-or-equal than the center
        to be considered a 'divergent' peak. Range: 0–8.

    Returns
    -------
    mask : 2D boolean np.ndarray
        True where cell is a FocalFlow-like peak.
    """
    arr = arr.astype("float32")

    if nodata is None:
        valid = np.isfinite(arr)
    else:
        valid = arr != nodata

    # Replace nodata with -inf so they never get picked
    arr_valid = np.where(valid, arr, -np.inf)

    # 8 neighbor shifts: (dy, dx)
    shifts = [
        (-1,  0),  # N
        (-1,  1),  # NE
        ( 0,  1),  # E
        ( 1,  1),  # SE
        ( 1,  0),  # S
        ( 1, -1),  # SW
        ( 0, -1),  # W
        (-1, -1),  # NW
    ]

    center = arr_valid
    diffs = []

    for dy, dx in shifts:
        neigh = np.roll(np.roll(arr_valid, dy, axis=0), dx, axis=1)
        diffs.append(center - neigh)

    # Shape: (8, rows, cols)
    diffs = np.stack(diffs, axis=0)

    # Count neighbors that are lower-or-equal
    outward_dirs = (diffs >= 0).sum(axis=0)

    # Max height difference to any neighbor
    max_diff = diffs.max(axis=0)

    # Conditions:
    # 1) valid center cell
    # 2) enough outward directions
    # 3) at least one neighbor is lower by min_height_diff
    peak_mask = (
        valid
        & (outward_dirs >= min_outward_dirs)
        & (max_diff >= min_height_diff)
    )

    return peak_mask


def thin_peaks_by_local_max(arr, peak_mask, nodata=None):
    """
    Thin candidate peaks so that only the highest cell
    in any 3x3 neighborhood is kept.
    """
    if nodata is None:
        valid_vals = np.where(np.isfinite(arr), arr, -np.inf)
    else:
        valid_vals = np.where(arr != nodata, arr, -np.inf)

    # 3x3 maximum of the center values
    local_max = maximum_filter(valid_vals, size=3, mode="nearest")

    # Keep only those peaks that are equal to the local maximum
    thinned = peak_mask & (valid_vals == local_max)
    return thinned


def process_tree_heights(
    dsm_path: str,
    be_path: str,
    out_dir: str,
    dsm_invert_name: str = "dsm_invert.tif",
    dsm_filter_name: str = "dsm_filter.tif",
    tree_heights_name: str = "tree_heights.tif",
    points_name: str = "tree height points.gpkg",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dsm_invert_path = out_dir / dsm_invert_name
    dsm_filter_path = out_dir / dsm_filter_name
    tree_heights_path = out_dir / tree_heights_name
    points_path = out_dir / points_name

    # ------------------------------------------------------------------
    # 1. Read DSM
    # ------------------------------------------------------------------
    with rasterio.open(dsm_path) as src_dsm:
        dsm = src_dsm.read(1).astype("float32")
        profile = src_dsm.profile
        transform = src_dsm.transform
        dsm_nodata = src_dsm.nodata
        dsm_crs = src_dsm.crs

    # ------------------------------------------------------------------
    # 2. Invert DSM and save
    # ------------------------------------------------------------------
    dsm_invert = -dsm
    inv_profile = profile.copy()
    inv_profile.update(dtype="float32", nodata=dsm_nodata)
    with rasterio.open(dsm_invert_path, "w", **inv_profile) as dst:
        dst.write(dsm_invert, 1)

    # ------------------------------------------------------------------
    # 3. Low-pass filter (3x3 mean) on inverted DSM and save
    # ------------------------------------------------------------------
    dsm_filter = smooth_3x3_mean(dsm_invert, nodata=dsm_nodata)
    filt_profile = profile.copy()
    filt_profile.update(dtype="float32", nodata=dsm_nodata)
    with rasterio.open(dsm_filter_path, "w", **filt_profile) as dst:
        dst.write(dsm_filter, 1)

    # ------------------------------------------------------------------
    # 4. Build smoothed DSM from filtered inverted DSM
    #    (equivalent to smoothing the DSM itself)
    # ------------------------------------------------------------------
    if dsm_nodata is not None:
        smoothed_dsm = np.where(dsm_filter == dsm_nodata, dsm_nodata, -dsm_filter)
    else:
        smoothed_dsm = -dsm_filter

    # ------------------------------------------------------------------
    # 5. Detect "FocalFlow-like" peaks on the smoothed DSM
    # ------------------------------------------------------------------
    raw_peak_mask = detect_focalflow_like_peaks(
        smoothed_dsm,
        nodata=dsm_nodata,
        min_height_diff=0.3,   # tweak as needed
        min_outward_dirs=6,    # tweak as needed (6–8 is reasonable)
    )

    # Optional: thin peaks so you only keep the highest in each 3x3
    local_max_mask = thin_peaks_by_local_max(
        smoothed_dsm,
        raw_peak_mask,
        nodata=dsm_nodata,
    )

    # ------------------------------------------------------------------
    # 6. Tree heights raster: DSM masked by local maxima
    # ------------------------------------------------------------------
    if dsm_nodata is None:
        out_nodata = -9999.0
    else:
        out_nodata = dsm_nodata

    tree_heights = np.where(local_max_mask, dsm, out_nodata).astype("float32")

    th_profile = profile.copy()
    th_profile.update(dtype="float32", nodata=out_nodata)
    with rasterio.open(tree_heights_path, "w", **th_profile) as dst:
        dst.write(tree_heights, 1)

    # ------------------------------------------------------------------
    # 7. Convert treetop cells to points
    # ------------------------------------------------------------------
    rows, cols = np.where(local_max_mask)
    if len(rows) == 0:
        print("No local maxima detected; no points created.")
        return

    xs, ys = xy(transform, rows, cols)
    points = [Point(x, y) for x, y in zip(xs, ys)]
    dsm_vals = dsm[rows, cols]

    # ------------------------------------------------------------------
    # 8. Extract bare-earth (BE) values at those points
    # ------------------------------------------------------------------
    with rasterio.open(be_path) as src_be:
        be = src_be.read(1).astype("float32")
        be_nodata = src_be.nodata
        be_crs = src_be.crs

        xs_be, ys_be = xs, ys

        # Reproject coordinates if DSM and BE CRSs differ
        if be_crs is not None and dsm_crs is not None and be_crs != dsm_crs:
            from pyproj import Transformer

            transformer = Transformer.from_crs(dsm_crs, be_crs, always_xy=True)
            xs_be, ys_be = transformer.transform(xs, ys)

        coords = list(zip(xs_be, ys_be))
        samples = list(src_be.sample(coords))
        be_vals = np.array([s[0] for s in samples], dtype="float32")

    # Treat BE nodata as NaN in the attributes
    if be_nodata is not None:
        be_vals_clean = be_vals.copy()
        be_vals_clean[be_vals_clean == be_nodata] = np.nan
    else:
        be_vals_clean = be_vals

    # ------------------------------------------------------------------
    # 9. Build GeoDataFrame and save as GeoPackage
    # ------------------------------------------------------------------
    gdf = gpd.GeoDataFrame(
        {
            "dsm_val": dsm_vals,
            "be_val": be_vals_clean,
            "tree_height_raw": dsm_vals,
            "height_above_ground": dsm_vals - be_vals_clean,
        },
        geometry=points,
        crs=dsm_crs,
    )

    gdf.to_file(points_path, driver="GPKG")
    print(f"Saved tree height raster to: {tree_heights_path}")
    print(f"Saved tree-height points to: {points_path}")


if __name__ == "__main__":

    # # ----------------- Leidl -----------------
    dsm_path = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\HAWS Map\Topography\2016_USDA_DSM.tif"
    be_path = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\HAWS Map\Topography\2016_USDA_DEM ndv.tif"
    out_dir = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\HAWS Map\Trees"    

    process_tree_heights(dsm_path, be_path, out_dir)




