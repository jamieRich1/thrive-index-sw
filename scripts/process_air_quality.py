import pandas as pd
import geopandas as gpd
from pathlib import Path
import os
import pyproj
import numpy as np
import sys
import re
from functools import reduce

#Paths and Constants
os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "air_quality"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
OUTPUT_FILE = PROCESSED_DATA_DIR / "lsoa_annual_air_quality.parquet"


#Scoring based on WHO standards
def calculate_who_based_score(pollutant, concentration):
    """Calculates a 0-100 score based on WHO annual guidelines using linear interpolation."""
    if pollutant == 'no2':
        #Concentration points (ug/m3): Lower is better
        x_points = [0, 10, 20, 40]
        #Corresponding Score points: Higher is better
        y_points = [100, 90, 50, 10]
    elif pollutant == 'pm25':
        #Concentration points (ug/m3)
        x_points = [0, 5, 10, 25]
        #Corresponding Score points
        y_points = [100, 90, 50, 10]
    else:
        return 0
    return np.interp(concentration, x_points, y_points)


#Data Processing
def process_pollutant(lsoa_gdf, config, pollutant_name):
    print(f"--- Processing {pollutant_name.upper()} ---")

    if not config['filepath'].exists():
        print(f"Warning: File not found {config['filepath'].name}. Skipping.")
        return None

    try:
        df = pd.read_csv(config['filepath'])
        if not all(col in df.columns for col in [config['value_col'], config['x_col'], config['y_col']]):
            print(f"ERROR: File {config['filepath'].name} is missing one of the required columns: ")
            print(f"  Expected: {config['value_col']}, {config['x_col']}, {config['y_col']}")
            print(f"  Found: {df.columns.tolist()}")
            return None

        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[config['x_col']], df[config['y_col']]),
                               crs="EPSG:27700")
    except Exception as e:
        print(f"ERROR loading {pollutant_name} file: {e}")
        return None

    print("Step 1: Finding points within LSOAs...")
    joined_gdf = gpd.sjoin(lsoa_gdf[['area_code', 'geometry']], gdf, how="inner", predicate="contains")
    lsoa_with_data = joined_gdf.groupby('area_code')[config['value_col']].mean().reset_index()
    lsoas_with_data_codes = set(lsoa_with_data['area_code'])
    lsoas_without_data_gdf = lsoa_gdf[~lsoa_gdf['area_code'].isin(lsoas_with_data_codes)]

    if not lsoas_without_data_gdf.empty:
        print(f"Step 2: Found {len(lsoas_without_data_gdf)} LSOAs without internal points. Finding nearest...")
        # Note: This is SPATIAL imputation (filling missing LSOAs), not temporal imputation (filling missing years).
        # This is a valid part of the primary processing.
        nearest_join = gpd.sjoin_nearest(lsoas_without_data_gdf[['area_code', 'geometry']], gdf, how="left")
        lsoa_with_nearest_data = nearest_join.groupby('area_code')[config['value_col']].mean().reset_index()
        combined_df = pd.concat([lsoa_with_data, lsoa_with_nearest_data], ignore_index=True)
    else:
        combined_df = lsoa_with_data

    print(f"Aggregated {pollutant_name.upper()} data for {len(combined_df)} LSOAs.")
    return combined_df.rename(columns={config['value_col']: f'{pollutant_name}_mean_concentration'})


#Main Process for Time Series
if __name__ == "__main__":
    print("Starting air quality processing (RAW EXTRACTION - NO IMPUTATION)...")
    try:
        lsoa_boundaries = gpd.read_parquet(LSOA_BOUNDARIES_FILE).to_crs("EPSG:27700")
    except Exception as e:
        print(f"ERROR: Could not load LSOA boundaries file: {LSOA_BOUNDARIES_FILE}")
        print(e)
        sys.exit(1)

    # Note: This range now reflects the data you have (2018-2024).
    # 2025 will be handled by the master build script.
    YEARS_TO_PROCESS = list(range(2018, 2025))
    all_annual_data = []

    print(f"Found {len(YEARS_TO_PROCESS)} years to process: {YEARS_TO_PROCESS}")

    for year in YEARS_TO_PROCESS:
        print(f"\n--- Processing data for year: {year} ---")
        pollutant_config_for_year = {
            'no2': {
                'filepath': RAW_DATA_DIR / f"no2_{year}.csv",
                'value_col': f'no2{year}_Band_1', 'x_col': 'X', 'y_col': 'Y'
            },
            'pm25': {
                'filepath': RAW_DATA_DIR / f"pm25_{year}.csv",
                'value_col': f'pm25{year}_Band_1', 'x_col': 'X', 'y_col': 'Y'
            }
        }

        processed_dataframes = []
        for name, config in pollutant_config_for_year.items():
            result_df = process_pollutant(lsoa_boundaries, config, name)
            if result_df is not None:
                processed_dataframes.append(result_df)

        if not processed_dataframes:
            print(f"Warning: No pollutant data processed for {year}. Skipping this year.")
            continue

        year_df = reduce(lambda left, right: pd.merge(left, right, on='area_code', how='outer'), processed_dataframes)
        year_df['no2_score'] = year_df[f'no2_mean_concentration'].apply(lambda x: calculate_who_based_score('no2', x))
        year_df['pm25_score'] = year_df[f'pm25_mean_concentration'].apply(
            lambda x: calculate_who_based_score('pm25', x))
        year_df['air_quality_score'] = year_df[['no2_score', 'pm25_score']].mean(axis=1)
        year_df['year'] = year
        all_annual_data.append(year_df)

    if not all_annual_data:
        print("No data processed for any year. Exiting.")
        sys.exit(1)

    #Combine to 1 DF
    print("\nCombining all processed years...")
    final_df = pd.concat(all_annual_data, ignore_index=True)

    #Final Columns
    output_cols = [
        'area_code', 'year', 'no2_mean_concentration', 'pm25_mean_concentration', 'air_quality_score'
    ]
    final_output_cols = [col for col in output_cols if col in final_df.columns]
    final_df[final_output_cols].to_parquet(OUTPUT_FILE, index=False)

    print(f"Success! Combined raw annual air quality data saved to {OUTPUT_FILE}")
    print("Script finished.")