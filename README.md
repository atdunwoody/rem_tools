# remtools

# Basin‑Wide REM Protocol

This repository implements a basin‑wide River Elevation Model (REM) protocol for the GRMW Atlas metric update project. The workflow generates an REM over a full basin using three DEM inputs:

1. **USGS 10 m DEM**  
2. **Water Surface Elevation (WSE) DEM**  
3. **Bathymetric DEM**  

All processing is performed with Python scripts that read from and write to GeoPackage files.

---

## Workflow Overview

1. **Stream Network Delineation**  
   - Script: `scripts/0_get_streams.py`  
   - Uses WhiteboxTools to hydrologically condition the USGS 10 m LiDAR DEM.  
   - Thresholds flow‑accumulation to create a vectorized stream network.  

2. **Transect Generation**  
   - Script: `scripts/1_get_transects.py`  
   - Generates perpendicular lines (500 m length) at a user‑defined spacing (default: 10 m) along the stream network.  

3. **Elevation Extraction Along Transects**  
   - Script: `scripts/2_get_elevations_along_transect.py`  
   - For each transect, extracts the minimum elevation from the WSE DEM.  
   - Creates three points per transect:  
     - Point at minimum elevation  
     - Two endpoints of the transect  

4. **Water Surface Interpolation**  
   - Script: `scripts/3_interpolate_water_surface.py`  
   - Applies Inverse Distance Weighting (IDW) to the WSE points.  
   - Uses a maximum interpolation radius equal to the valley’s maximum width.  
   - Outputs an interpolated WSE raster.  

5. **REM Calculation**  
   - REM = **Bathymetric DEM** − **Interpolated WSE raster**  

---





