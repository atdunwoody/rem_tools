import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

def classify_raster(input_raster_path: str,
                    reference_raster_path: str,
                    output_raster_path: str) -> None:
    """
    Classify an input raster against a reference raster into four classes:
      1 if input < 0.5 * reference
      2 if 0.5 * reference < input < reference
      3 if reference < input < 2 * reference
      4 if input > 2 * reference

    Pixels where either input or reference is nodata are left as nodata.
    """
    print(f"Classifying raster: {input_raster_path} \nusing {reference_raster_path} \n→ {output_raster_path}")
    # Open input as a masked array
    with rasterio.open(input_raster_path) as src_in:
        in_band = src_in.read(1, masked=True)              # masked.where(src_in.nodata)
        meta = src_in.meta.copy()
        in_nodata = src_in.nodata
        # Ensure we have an output nodata value (use 0 if input has none)
        out_nodata = 0
        meta.update(dtype=rasterio.uint8,
                    count=1,
                    nodata=out_nodata)

        # Prepare array to receive the resampled reference
        ref_data = np.empty(src_in.shape, dtype=np.float32)

        with rasterio.open(reference_raster_path) as src_ref:
            ref_nodata = src_ref.nodata
            # Reproject reference → input grid
            reproject(
                source=rasterio.band(src_ref, 1),
                destination=ref_data,
                src_transform=src_ref.transform,
                src_crs=src_ref.crs,
                dst_transform=src_in.transform,
                dst_crs=src_in.crs,
                dst_resolution=src_in.res,
                resampling=Resampling.bilinear,
                dst_nodata=ref_nodata
            )
            # Mask out reference nodata
            ref_band = np.ma.masked_equal(ref_data, ref_nodata)

    # Start output filled with nodata
    cls = np.full(src_in.shape, out_nodata, dtype=np.uint8)

    # Build a “valid data” mask
    valid = (~in_band.mask) & (~ref_band.mask)

    # Apply rules only where valid
    inp = in_band.data
    ref = ref_band.data

    cls[(inp < 0.5 * ref) & valid] = 1
    cls[(inp > 0.5 * ref) & (inp < ref) & valid] = 2
    cls[(inp > ref)       & (inp < 2 * ref) & valid] = 3
    cls[(inp > 2 * ref)  & (inp < 4 * ref) & valid] = 4
    cls[(inp > 4 * ref) & valid] = 5

    # Write out
    with rasterio.open(output_raster_path, 'w', **meta) as dst:
        dst.write(cls, 1)

    print(f"[✔] Classification complete. Output saved to: {output_raster_path}")

if __name__ == "__main__":


    input_raster = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Marketing\Proposals\Ukiah Meadows UCSWCD\REM\REM_HAWS_clipped.tif"
    
    reference_raster = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Marketing\Proposals\Ukiah Meadows UCSWCD\REM\BF_depth_Legg_m.tif"
    output_raster = r"C:\Users\AlexThornton-Dunwood\OneDrive - Lichen Land & Water\Lichen Drive\Marketing\Proposals\Ukiah Meadows UCSWCD\REM\REM_HAWS_cat_Legg.tif"

    classify_raster(input_raster, reference_raster, output_raster)
    print(f"Classification completed. Output saved to {output_raster}")
    