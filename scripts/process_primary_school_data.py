import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree
import numpy as np
import sys
import re

print("Starting WEIGHTED Primary School (KS2) Processing...")

#Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "schools"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
POSTCODE_FILE = PROCESSED_DATA_DIR / "sw_postcodes.parquet"
OUTPUT_FILE = PROCESSED_DATA_DIR / "lsoa_annual_primary_weighted.parquet"
HISTORICAL_OUTPUT = PROCESSED_DATA_DIR / "primary_school_historical_data.parquet"
NEAREST_N = 3
YEARS_TO_PROCESS = list(range(2018, 2026))
LOC_COLS_TARGET = {'urn': 'urn', 'schname': 'name', 'pcode': 'postcode'}
PERF_COLS_TARGET = {
    'urn': 'urn',
    'ptrwm_exp': 'pass_rate',
    'read_average': 'read_score',
    'mat_average': 'math_score'
}
SENTINEL_VALUES = ['SUPP', 'NE', 'LOWCOV', 'NA', '#N/A', 'NEW', 'NP', '', 'nan']


#Helpers
def extract_year_from_filename(filename):
    match = re.search(r'(\d{4})-(\d{4})', filename)
    return int(match.group(2)) if match else None

def robust_load_csv(filepath, target_map, encoding='latin-1'):
    year = extract_year_from_filename(filepath.name)
    if not year: return None, None
    try:
        headers_actual = pd.read_csv(filepath, nrows=0, encoding=encoding).columns
        headers_map = {h.lower(): h for h in headers_actual}
        use_cols, rename_map = [], {}
        found_targets = set()
        for target_low, target_clean in target_map.items():
            if target_clean in found_targets: continue
            if target_low in headers_map:
                actual = headers_map[target_low]
                use_cols.append(actual)
                rename_map[actual] = target_clean
                found_targets.add(target_clean)
        if 'urn' not in rename_map.values(): return None, year
        df = pd.read_csv(filepath, usecols=use_cols, encoding=encoding, dtype=str, low_memory=False)
        df.rename(columns=rename_map, inplace=True)
        return df, year
    except:
        return None, year

def load_school_locations():
    print("  -> Loading school locations...")
    files = [f for f in RAW_DATA_DIR.glob("*ks2*.csv") if 'provisional' not in f.name.lower()]
    all_locs = []
    for f in files:
        df, year = robust_load_csv(f, LOC_COLS_TARGET)
        if df is not None and not df.empty:
            df['year'] = year
            df.dropna(subset=['urn'], inplace=True)
            all_locs.append(df)
    if not all_locs: return None
    master = pd.concat(all_locs).sort_values('year').drop_duplicates('urn', keep='last')
    if POSTCODE_FILE.exists():
        postcodes = pd.read_parquet(POSTCODE_FILE)
        merged = master.merge(postcodes, left_on='postcode', right_on='Postcode', how='inner')
        return gpd.GeoDataFrame(merged, geometry=gpd.points_from_xy(merged.longitude, merged.latitude),
                                crs="EPSG:4326").to_crs("EPSG:27700")
    else:
        print("Warning: Postcode file not found, cannot geocode.")
        return None

def load_school_performance():
    print("  -> Loading performance data...")
    files = [f for f in RAW_DATA_DIR.glob("*ks2*.csv") if 'provisional' not in f.name.lower()]
    all_perf = []
    for f in files:
        df, year = robust_load_csv(f, PERF_COLS_TARGET)
        if df is not None and not df.empty:
            df.dropna(subset=['urn'], inplace=True)
            df['urn'] = df['urn'].astype(str)
            df = df[df['urn'] != 'nan']
            for c in ['pass_rate', 'read_score', 'math_score']:
                if c in df.columns:
                    df[c] = df[c].astype(str).str.replace('%', '', regex=False)
                    df[c] = pd.to_numeric(df[c].replace(SENTINEL_VALUES, np.nan), errors='coerce')
                else:
                    df[c] = np.nan
            df['year'] = year
            all_perf.append(df)
    return pd.concat(all_perf, ignore_index=True) if all_perf else None

def process_weighted_primary(lsoa_gdf, school_gdf, perf_df):
    print("  -> Calculating weighted indicators (Distance Weighted Average)...")

    #Spatial Indexing/Centriods
    lsoa_centroids = lsoa_gdf.geometry.centroid
    lsoa_coords = np.array(list(zip(lsoa_centroids.x, lsoa_centroids.y)))
    school_coords = np.array(list(zip(school_gdf.geometry.x, school_gdf.geometry.y)))
    tree = cKDTree(school_coords)
    dists, idxs = tree.query(lsoa_coords, k=NEAREST_N)

    #Long Format
    lsoa_indices = np.repeat(np.arange(len(lsoa_gdf)), NEAREST_N)
    flat_idxs = idxs.flatten()
    flat_dists = dists.flatten()
    long_df = pd.DataFrame({
        'area_code': lsoa_gdf.iloc[lsoa_indices]['area_code'].values,
        'urn': school_gdf.iloc[flat_idxs]['urn'].values,
        'distance_km': flat_dists / 1000.0
    })

    #Joins
    years_df = pd.DataFrame({'year': YEARS_TO_PROCESS})
    long_df = long_df.merge(years_df, how='cross')
    long_df['urn'] = long_df['urn'].astype(str)
    perf_df['urn'] = perf_df['urn'].astype(str)
    merged_df = long_df.merge(perf_df, on=['urn', 'year'], how='left')

    #Wighted Aggregation Loop
    metrics = ['pass_rate', 'read_score', 'math_score']
    final_dfs = []
    for col in metrics:
        if col not in merged_df.columns: continue
        print(f"     Processing {col}...")
        valid = merged_df.dropna(subset=[col]).copy()
        if valid.empty:
            print(f"     Warning: No valid data for {col}")
            continue
        valid['weight'] = 1 / (valid['distance_km'] + 0.1)
        valid['weighted_val'] = valid[col] * valid['weight']
        grouped = valid.groupby(['area_code', 'year'])[['weighted_val', 'weight']].sum()
        grouped[f'primary_{col}_weighted'] = grouped['weighted_val'] / grouped['weight']
        final_dfs.append(grouped[[f'primary_{col}_weighted']])
    if not final_dfs: return pd.DataFrame()
    result = pd.concat(final_dfs, axis=1).reset_index()
    return result

#Main
if __name__ == "__main__":
    if not LSOA_BOUNDARIES_FILE.exists():
        sys.exit(f"Error: LSOA boundaries not found at {LSOA_BOUNDARIES_FILE}")
    schools_gdf = load_school_locations()
    perf_df = load_school_performance()
    if schools_gdf is None or perf_df is None:
        sys.exit("Error loading data.")
    hist_df = perf_df.copy().rename(columns={'urn': 'URN', 'pass_rate': 'avg_ks2_pass_rate'})
    hist_df['URN'] = hist_df['URN'].astype(str)
    hist_df.to_parquet(HISTORICAL_OUTPUT, index=False)
    lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE).to_crs("EPSG:27700")
    final_df = process_weighted_primary(lsoa_gdf, schools_gdf, perf_df)
    lsoa_codes = lsoa_gdf['area_code'].unique()
    master_index = pd.MultiIndex.from_product([lsoa_codes, YEARS_TO_PROCESS], names=['area_code', 'year'])
    master_df = pd.DataFrame(index=master_index).reset_index()
    final_df = master_df.merge(final_df, on=['area_code', 'year'], how='left')
    print(f"Saving to {OUTPUT_FILE}...")
    final_df.to_parquet(OUTPUT_FILE, index=False)
    print("Done.")