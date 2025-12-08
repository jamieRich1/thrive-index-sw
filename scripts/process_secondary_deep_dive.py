import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree
import numpy as np
import sys
import re

print("Starting SECONDARY School Deep Dive Data Processing...")

# Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "schools"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
POSTCODE_FILE = PROCESSED_DATA_DIR / "sw_postcodes.parquet"
OUTPUT_FILE = PROCESSED_DATA_DIR / "lsoa_annual_secondary_deepdive.parquet"
NEAREST_N = 3
YEARS_TO_PROCESS = list(range(2018, 2026))
# Column Mappings
LOC_COLS_TARGET = {'urn': 'urn', 'schname': 'name', 'pcode': 'postcode', 'nftype': 'nftype'}
PERF_COLS_TARGET = {
    'urn': 'urn',
    'p8mea': 'progress_8',
    'att8scr': 'attainment_8'
}
SENTINEL_VALUES = ['SUPP', 'NE', 'LOWCOV', 'NA', '#N/A', 'NEW', 'NP', '', 'nan']

# Helpers
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
    files = [f for f in RAW_DATA_DIR.glob("*ks4*.csv") if 'provisional' not in f.name.lower()]
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
        master['postcode_clean'] = master['postcode'].str.replace(' ', '').str.upper()
        postcodes['Postcode'] = postcodes['Postcode'].str.replace(' ', '').str.upper()
        merged = master.merge(postcodes, left_on='postcode_clean', right_on='Postcode', how='inner')
        return gpd.GeoDataFrame(merged, geometry=gpd.points_from_xy(merged.longitude, merged.latitude),
                                crs="EPSG:4326").to_crs("EPSG:27700")
    else:
        print("Warning: Postcode file not found, cannot geocode.")
        return None

def load_school_performance():
    print("  -> Loading performance data...")
    files = [f for f in RAW_DATA_DIR.glob("*ks4*.csv") if 'provisional' not in f.name.lower()]
    all_perf = []
    for f in files:
        df, year = robust_load_csv(f, PERF_COLS_TARGET)
        if df is not None and not df.empty:
            df.dropna(subset=['urn'], inplace=True)
            df['urn'] = df['urn'].astype(str)
            df = df[df['urn'] != 'nan']
            for c in ['progress_8', 'attainment_8']:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c].replace(SENTINEL_VALUES, np.nan), errors='coerce')
                else:
                    df[c] = np.nan
            df['year'] = year
            all_perf.append(df)
    return pd.concat(all_perf, ignore_index=True) if all_perf else None

def process_nearest_schools_deepdive(lsoa_gdf, school_gdf, perf_df):
    print("  -> Calculating nearest schools and mapping history...")
    # 1. Spatial Indexing to find Nearest 3 (Using latest school locations)
    lsoa_coords = np.array(list(zip(lsoa_gdf.geometry.centroid.x, lsoa_gdf.geometry.centroid.y)))
    school_coords = np.array(list(zip(school_gdf.geometry.x, school_gdf.geometry.y)))
    tree = cKDTree(school_coords)
    dists, indices = tree.query(lsoa_coords, k=NEAREST_N)
    # 2. Build the mappings
    lsoa_codes = lsoa_gdf['area_code'].values
    years = np.array(YEARS_TO_PROCESS)
    records = []
    # Create lookup dictionaries for performance data for speed
    perf_df['urn'] = perf_df['urn'].astype(str)
    perf_df = perf_df.drop_duplicates(subset=['urn', 'year'], keep='last')
    perf_lookup = perf_df.set_index(['urn', 'year']).to_dict('index')
    # Create lookup for static school info
    school_info = school_gdf.set_index('urn')[['name', 'nftype']].to_dict('index')
    print("  -> Constructing time-series rows...")
    for i, lsoa_code in enumerate(lsoa_codes):
        # Get the 3 nearest schools for this LSOA
        school_indices = indices[i]
        schools_for_lsoa = []
        for rank, idx in enumerate(school_indices):
            urn = school_gdf.iloc[idx]['urn']
            info = school_info.get(urn, {})
            schools_for_lsoa.append({
                'rank': rank + 1,
                'urn': urn,
                'name': info.get('name', 'Unknown'),
                'nftype': info.get('nftype', 'NA')
            })
        # For every year, create a row for this LSOA
        for year in years:
            row = {'area_code': lsoa_code, 'year': year}
            for school in schools_for_lsoa:
                r = school['rank']
                urn = school['urn']
                # Base Info
                row[f'school_{r}_name'] = school['name']
                row[f'school_{r}_urn'] = urn
                row[f'school_{r}_nftype'] = school['nftype']
                # Performance Data for this specific year
                perf_data = perf_lookup.get((urn, year), {})
                row[f'school_{r}_progress_8'] = perf_data.get('progress_8', np.nan)
                row[f'school_{r}_attainment_8'] = perf_data.get('attainment_8', np.nan)
                row[f'school_{r}_data_year'] = year if perf_data else np.nan
            records.append(row)
    return pd.DataFrame(records)

# Main
if __name__ == "__main__":
    if not LSOA_BOUNDARIES_FILE.exists():
        sys.exit(f"Error: LSOA boundaries not found at {LSOA_BOUNDARIES_FILE}")
    schools_gdf = load_school_locations()
    perf_df = load_school_performance()
    if schools_gdf is None or perf_df is None:
        sys.exit("Error loading data.")
    lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE).to_crs("EPSG:27700")
    final_df = process_nearest_schools_deepdive(lsoa_gdf, schools_gdf, perf_df)
    print(f"Saving {len(final_df)} rows to {OUTPUT_FILE}...")
    final_df.to_parquet(OUTPUT_FILE, index=False)
    print("Done.")