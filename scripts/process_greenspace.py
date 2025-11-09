import geopandas as gpd
import pandas as pd
from pathlib import Path

#Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "greenspace" / "opgrsp_gb.gpkg"
PROCESSED_DATA_PATH = DATA_DIR / "processed"
LSOA_BOUNDARIES_PATH = PROCESSED_DATA_PATH / "boundaries_lsoa.geoparquet"
OUTPUT_FILE = PROCESSED_DATA_PATH / "lsoa_greenspace.parquet"

#Load Data
print("Loading data...")
greenspace_gdf = gpd.read_file(RAW_DATA_PATH)
lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_PATH)

#Prepare Geo Dataframes
print("Converting CRS to EPSG:27700 for accurate area calculations...")
lsoa_gdf = lsoa_gdf.to_crs(epsg=27700)
greenspace_gdf = greenspace_gdf.to_crs(epsg=27700)

#Spatial Join
print("Performing spatial intersection... (this may take a few minutes)")
intersected_gdf = gpd.overlay(lsoa_gdf[["area_code", "geometry"]], greenspace_gdf, how="intersection")
intersected_gdf['greenspace_area_m2'] = intersected_gdf.geometry.area

#Aggregate and quantify greenspace
print("Aggregating greenspace area per LSOA...")
greenspace_by_lsoa = intersected_gdf.groupby('area_code')['greenspace_area_m2'].sum().reset_index()
lsoa_gdf['lsoa_area_m2'] = lsoa_gdf.geometry.area

#Final DF
print("Calculating final greenspace percentage metric...")
final_gdf = lsoa_gdf.merge(greenspace_by_lsoa, on='area_code', how='left')
final_gdf['greenspace_area_m2'] = final_gdf['greenspace_area_m2'].fillna(0)
final_gdf['greenspace_percentage'] = (final_gdf['greenspace_area_m2'] / final_gdf['lsoa_area_m2']) * 100
output_df = final_gdf[['area_code', 'greenspace_percentage']]
print(f"Saving processed data to {OUTPUT_FILE}...")
output_df.to_parquet(OUTPUT_FILE)
print("Done!")