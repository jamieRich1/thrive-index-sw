import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree
import numpy as np
import sys
import re

print("Starting healthcare data processing")

#Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "healthcare"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
GP_LOCATIONS_FILE = RAW_DATA_DIR / "epraccur.csv"
SURVEY_FILE_PATTERN = "GPPS_*.csv"
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
POSTCODE_FILE = PROCESSED_DATA_DIR / "sw_postcodes.parquet"
ANNUAL_SCORES_OUTPUT_LSOA = PROCESSED_DATA_DIR / "lsoa_annual_healthcare_scores.parquet"
STATIC_DETAILS_OUTPUT_LSOA = PROCESSED_DATA_DIR / "lsoa_healthcare_details.parquet"
HISTORICAL_GP_SCORES_OUTPUT = PROCESSED_DATA_DIR / "gp_historical_satisfaction.parquet"
NEAREST_N_GPS = 3
YEARS_TO_PROCESS = list(range(2018, 2026))

def extract_year_from_filename(filename):
    match = re.search(r'_(\d{4})_', filename)
    if match: return int(match.group(1))
    match_2025 = re.search(r'GPPS_(\d{4})_Practice_data', filename)
    if match_2025: return int(match_2025.group(1))
    return None

def load_all_satisfaction_data():
    print("  -> Loading historical satisfaction files...")
    survey_files = list(RAW_DATA_DIR.glob(SURVEY_FILE_PATTERN))
    all_data = []
    for f in survey_files:
        year = extract_year_from_filename(f.name)
        if not year: continue
        try:
            df_check = pd.read_csv(f, nrows=1, encoding='utf-8-sig')
            cols = df_check.columns.tolist()
            sat_col, org_col = None, None
            for c in ['overallexp.pcteval', 'Q28_12pct', 'q28_12pct']:
                if c in cols: sat_col = c; break
            for c in ['ad_practicecode', 'practice_code', 'Practice_Code']:
                if c in cols: org_col = c; break
            if not sat_col or not org_col: continue
            df = pd.read_csv(f, usecols=[org_col, sat_col], encoding='utf-8-sig', low_memory=False)
            df.columns = ['org_code', 'sat_val']
            df['org_code'] = df['org_code'].str.strip()
            df['satisfaction_pct'] = pd.to_numeric(df['sat_val'], errors='coerce') * 100
            df = df[(df['satisfaction_pct'] >= 0) & (df['satisfaction_pct'] <= 100)]
            df['year'] = year
            all_data.append(df[['org_code', 'year', 'satisfaction_pct']])
        except Exception as e:
            print(f"Error {f.name}: {e}")
    return pd.concat(all_data, ignore_index=True) if all_data else None

def load_practice_locations():
    if not GP_LOCATIONS_FILE.exists(): return None
    df = pd.read_csv(GP_LOCATIONS_FILE, header=None, usecols=[0, 1, 9, 25],
                     names=['org_code', 'name', 'postcode', 'setting_code'], encoding='latin-1', dtype=str)
    df = df[df['setting_code'] == '4'].copy()
    df['postcode'] = df['postcode'].str.replace(' ', '').str.upper()
    return df[['org_code', 'name', 'postcode']]

def geocode_practices(gp_df):
    postcodes = pd.read_parquet(POSTCODE_FILE)
    postcodes['Postcode'] = postcodes['Postcode'].str.replace(' ', '').str.upper()
    gp_gdf = gp_df.merge(postcodes, left_on='postcode', right_on='Postcode', how='inner')
    return gpd.GeoDataFrame(gp_gdf, geometry=gpd.points_from_xy(gp_gdf.longitude, gp_gdf.latitude),
                            crs="EPSG:4326").to_crs("EPSG:27700")

def process_nearest_gps(lsoa_gdf, gp_gdf, perf_df):
    print("  -> Calculating nearest GPs...")
    gp_coords = np.array(list(zip(gp_gdf.geometry.x, gp_gdf.geometry.y)))
    lsoa_coords = np.array(list(zip(lsoa_gdf.geometry.centroid.x, lsoa_gdf.geometry.centroid.y)))
    tree = cKDTree(gp_coords)
    dists, indices = tree.query(lsoa_coords, k=NEAREST_N_GPS)
    map_rows = []
    for i, lsoa_code in enumerate(lsoa_gdf['area_code']):
        for rank in range(NEAREST_N_GPS):
            idx = indices[i][rank]
            gp_row = gp_gdf.iloc[idx]
            map_rows.append({
                'area_code': lsoa_code,
                'rank': rank + 1,
                'org_code': gp_row['org_code'],
                'gp_name': gp_row['name'],
                'distance': dists[i][rank] / 1000.0
            })
    map_df = pd.DataFrame(map_rows)

    # Crosss Join Years
    years_df = pd.DataFrame({'year': YEARS_TO_PROCESS})
    map_expanded = map_df.merge(years_df, how='cross')
    merged_df = map_expanded.merge(perf_df, on=['org_code', 'year'], how='left')
    pivot_df = merged_df.pivot_table(
        index=['area_code', 'year'],
        columns='rank',
        values=['satisfaction_pct', 'distance', 'gp_name', 'org_code'],
        aggfunc='first'
    )
    pivot_df.columns = [f"gp_{col[1]}_{col[0]}" for col in pivot_df.columns]
    pivot_df.columns = pivot_df.columns.str.replace('satisfaction_pct', 'satisfaction')
    return pivot_df.reset_index()

# Main
if __name__ == "__main__":
    perf_df = load_all_satisfaction_data()
    gp_locs = load_practice_locations()
    if perf_df is None or gp_locs is None: sys.exit("Error loading data.")
    latest = perf_df['year'].max()
    active_gps = perf_df[perf_df['year'] == latest]['org_code'].unique()
    gp_locs = gp_locs[gp_locs['org_code'].isin(active_gps)]
    gp_gdf = geocode_practices(gp_locs)
    perf_df.to_parquet(HISTORICAL_GP_SCORES_OUTPUT, index=False)
    lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE).to_crs("EPSG:27700")
    final_df = process_nearest_gps(lsoa_gdf, gp_gdf, perf_df)
    final_df.to_parquet(ANNUAL_SCORES_OUTPUT_LSOA, index=False)
    print(f"Saved annual scores to {ANNUAL_SCORES_OUTPUT_LSOA}")
    static_df = final_df[final_df['year'] == latest].copy()
    renames = {}
    for i in range(1, NEAREST_N_GPS + 1):
        renames[f'gp_{i}_gp_name'] = f'gp_{i}_name'
        renames[f'gp_{i}_distance'] = f'gp_{i}_distance_km'
    static_df.rename(columns=renames, inplace=True)
    static_df.to_parquet(STATIC_DETAILS_OUTPUT_LSOA, index=False)
    print("Done.")