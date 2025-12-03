import pandas as pd
import geopandas as gpd
from pathlib import Path
import os
import pyproj
import numpy as np
import sys
from functools import reduce

# Paths and Constants
os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "air_quality"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
OUTPUT_FILE = PROCESSED_DATA_DIR / "lsoa_annual_air_quality.parquet"


# Scoring based on WHO standards
def calculate_who_based_score(pollutant, concentration):
    if pd.isna(concentration) or concentration < 0: return np.nan
    if pollutant == 'no2':
        # Concentration points (ug/m3): Lower is better
        x_points = [0, 10, 20, 40]
        y_points = [100, 90, 50, 10]
    elif pollutant == 'pm25':
        x_points = [0, 5, 10, 25]
        y_points = [100, 90, 50, 10]
    else:
        return 0
    return np.interp(concentration, x_points, y_points)


def process_pollutant(lsoa_gdf, config, pollutant_name):
    print(f"--- Processing {pollutant_name.upper()} ---")
    if not config['filepath'].exists():
        print(f"Warning: File not found {config['filepath'].name}. Skipping.")
        return None

    try:
        df = pd.read_csv(config['filepath'])
        # Filter out negative values (Sentinel values like -999)
        val_col = config['value_col']
        initial_count = len(df)
        df = df[df[val_col] >= 0].copy()
        dropped = initial_count - len(df)
        if dropped > 0:
            print(f"   -> Dropped {dropped} rows with negative values.")

        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[config['x_col']], df[config['y_col']]),
                               crs="EPSG:27700")
    except Exception as e:
        print(f"ERROR loading {pollutant_name} file: {e}")
        return None

    print("Step 1: Finding points within LSOAs...")
    joined_gdf = gpd.sjoin(lsoa_gdf[['area_code', 'geometry']], gdf, how="inner", predicate="contains")
    lsoa_with_data = joined_gdf.groupby('area_code')[val_col].mean().reset_index()

    lsoas_with_data_codes = set(lsoa_with_data['area_code'])
    lsoas_without_data_gdf = lsoa_gdf[~lsoa_gdf['area_code'].isin(lsoas_with_data_codes)]

    if not lsoas_without_data_gdf.empty:
        print(f"Step 2: Found {len(lsoas_without_data_gdf)} LSOAs without internal points. Finding nearest...")
        nearest_join = gpd.sjoin_nearest(lsoas_without_data_gdf[['area_code', 'geometry']], gdf, how="left")
        lsoa_with_nearest_data = nearest_join.groupby('area_code')[val_col].mean().reset_index()
        combined_df = pd.concat([lsoa_with_data, lsoa_with_nearest_data], ignore_index=True)
    else:
        combined_df = lsoa_with_data

    print(f"Aggregated {pollutant_name.upper()} data for {len(combined_df)} LSOAs.")
    return combined_df.rename(columns={val_col: f'{pollutant_name}_mean_concentration'})


if __name__ == "__main__":
    print("Starting air quality processing (Cleaned)...")
    try:
        lsoa_boundaries = gpd.read_parquet(LSOA_BOUNDARIES_FILE).to_crs("EPSG:27700")
    except Exception as e:
        sys.exit(f"ERROR: Could not load LSOA boundaries: {e}")

    # Process 2018-2024 (2025 will be imputed by Master Builder)
    YEARS_TO_PROCESS = list(range(2018, 2025))
    all_annual_data = []

    for year in YEARS_TO_PROCESS:
        print(f"\n--- Processing data for year: {year} ---")
        pollutant_config = {
            'no2': {'filepath': RAW_DATA_DIR / f"no2_{year}.csv", 'value_col': f'no2{year}_Band_1', 'x_col': 'X',
                    'y_col': 'Y'},
            'pm25': {'filepath': RAW_DATA_DIR / f"pm25_{year}.csv", 'value_col': f'pm25{year}_Band_1', 'x_col': 'X',
                     'y_col': 'Y'}
        }

        dfs = []
        for name, cfg in pollutant_config.items():
            res = process_pollutant(lsoa_boundaries, cfg, name)
            if res is not None: dfs.append(res)

        if not dfs: continue

        year_df = reduce(lambda left, right: pd.merge(left, right, on='area_code', how='outer'), dfs)

        # Calculate score
        year_df['no2_score'] = year_df['no2_mean_concentration'].apply(lambda x: calculate_who_based_score('no2', x))
        year_df['pm25_score'] = year_df['pm25_mean_concentration'].apply(lambda x: calculate_who_based_score('pm25', x))
        year_df['air_quality_score'] = year_df[['no2_score', 'pm25_score']].mean(axis=1)
        year_df['year'] = year
        all_annual_data.append(year_df)

    if not all_annual_data: sys.exit("No data processed.")

    final_df = pd.concat(all_annual_data, ignore_index=True)
    cols = ['area_code', 'year', 'no2_mean_concentration', 'pm25_mean_concentration', 'air_quality_score']
    final_df[[c for c in cols if c in final_df.columns]].to_parquet(OUTPUT_FILE, index=False)
    print(f"Success! Saved to {OUTPUT_FILE}")