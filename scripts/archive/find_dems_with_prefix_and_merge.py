from __future__ import annotations

import os
from pathlib import Path
from typing import List

import rasterio
from rasterio.merge import merge


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------
SEARCH_DIRS = [
    r"C:\Users\AlexThornton-Dunwood\Downloads\LDQ-42122H5",
    r"C:\Users\AlexThornton-Dunwood\Downloads\LDQ-42122H6",
    r"C:\Users\AlexThornton-Dunwood\Downloads\LDQ-43122A6",
]

# CSV you just created earlier (single column "filename", values like UMPLWJ_388500_994500_20160927)
NAME_CSV = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\HAWS Map\Topography\umplwj_filenames.csv"  

OUT_TIF = r"C:\L\Lichen\Lichen - Documents\Marketing\Proposals\Luck Creek\HAWS Map\Topography\2016_USDA_DSM.tif"

# The files we want to find look like: DSM_<name>.img   (name is from csv)
PREFIX = "DSM_"
EXT = ".img"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_targets(csv_path: str) -> List[str]:
    """Return list of target basenames (no DSM_ prefix, no extension)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "filename" in df.columns:
        names = df["filename"].dropna().astype(str).tolist()
    else:
        # fallback: first column
        names = df.iloc[:, 0].dropna().astype(str).tolist()

    # normalize just in case
    names = [n.strip() for n in names if n.strip()]
    return names


def find_matching_imgs(search_dirs: List[str], targets: List[str]) -> List[Path]:
    """
    Search folders recursively for rasters named DSM_<target>.img (case-insensitive).
    Returns list of full paths found.
    """
    wanted = {f"{PREFIX}{t}{EXT}".lower() for t in targets}

    found: List[Path] = []
    for d in search_dirs:
        root = Path(d)
        if not root.exists():
            print(f"[WARN] Missing directory: {root}")
            continue

        for p in root.rglob(f"*{EXT}"):
            if p.name.lower() in wanted:
                found.append(p)

    # de-dup (in case files appear in multiple places)
    found = sorted(set(found))
    return found


def mosaic_to_geotiff(img_paths: List[Path], out_tif: str) -> None:
    if not img_paths:
        raise RuntimeError("No matching .img rasters found; nothing to merge.")

    out_tif = str(Path(out_tif))
    Path(out_tif).parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Merging {len(img_paths)} rasters...")
    srcs = [rasterio.open(str(p)) for p in img_paths]

    try:
        mosaic, out_transform = merge(srcs)  # mosaic: (bands, rows, cols)

        # Assume all sources share CRS/dtype/nodata/band count
        ref = srcs[0]
        out_meta = ref.meta.copy()
        out_meta.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=out_transform,
            compress="DEFLATE",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        with rasterio.open(out_tif, "w", **out_meta) as dst:
            dst.write(mosaic)

        print(f"[DONE] Wrote: {out_tif}")

    finally:
        for s in srcs:
            s.close()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    targets = load_targets(NAME_CSV)
    print(f"[INFO] Loaded {len(targets)} target names from CSV.")

    img_paths = find_matching_imgs(SEARCH_DIRS, targets)
    print(f"[INFO] Found {len(img_paths)} matching DSM_*.img files.")

    # helpful reporting: what's missing?
    found_names = {p.stem.lower() for p in img_paths}  # e.g., dsm_umplwj_...
    wanted_names = {f"{PREFIX}{t}".lower() for t in targets}
    missing = sorted(wanted_names - found_names)
    if missing:
        print(f"[WARN] Missing {len(missing)} expected rasters. Example(s):")
        for m in missing[:10]:
            print("   ", m + EXT)

    # Merge
    mosaic_to_geotiff(img_paths, OUT_TIF)


if __name__ == "__main__":
    main()
