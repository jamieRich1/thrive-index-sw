import geopandas as gpd
from pathlib import Path

print("Starting greenspace geometry processing...")

#Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "greenspace" / "opgrsp_gb.gpkg"
SW_BOUNDARY_PATH = DATA_DIR / "processed" / "lad_sw_outline.geojson"
OUTPUT_FILE = DATA_DIR / "processed" / "sw_greenspace_geometries.geoparquet"

#Data Load
print("Loading source files...")

try:
    #Load the raw OS greenspace data
    greenspace_gdf = gpd.read_file(RAW_DATA_PATH, layer='greenspace_site')
    print(f"Loaded {len(greenspace_gdf)} total UK greenspace polygons.")
    #Load the boundary for the South West
    sw_boundary = gpd.read_file(SW_BOUNDARY_PATH)
    print(f"Loaded South West boundary.")
except Exception as e:
    print(f"ERROR: Could not load source files. Check paths and file integrity. Details: {e}")
    exit()

print("Aligning Coordinate Reference Systems (CRS) to EPSG:27700...")
greenspace_gdf = greenspace_gdf.to_crs(epsg=27700)
sw_boundary = sw_boundary.to_crs(epsg=27700)
print(f"Greenspace CRS: {greenspace_gdf.crs}")
print(f"SW Boundary CRS: {sw_boundary.crs}")

#Clip Greenspace
print("Clipping greenspace polygons to the South West boundary...")
sw_greenspace_gdf = gpd.clip(greenspace_gdf, sw_boundary)
num_polygons = len(sw_greenspace_gdf)
print(f"Number of polygons after clipping: {num_polygons}")

if num_polygons == 0:
    print("WARNING: The clip operation resulted in 0 polygons. The output file will be empty.")
    print("This can happen if the source files don't geographically overlap or have CRS issues.")
else:
    #Save
    print(f"Saving {num_polygons} processed geometries to {OUTPUT_FILE}...")
    sw_greenspace_gdf[['geometry']].to_parquet(OUTPUT_FILE)
    print("Success! File saved.")
print("Script finished.")