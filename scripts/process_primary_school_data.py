import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree
import numpy as np
import sys
import re

print("Starting PRIMARY school (KS2) data processing (FINAL FIX)...")

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "schools"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
POSTCODE_FILE = PROCESSED_DATA_DIR / "sw_postcodes.parquet"

OUTPUT_FILE_LSOA = PROCESSED_DATA_DIR / "lsoa_annual_primary_scores.parquet"
STATIC_DETAILS_OUTPUT_LSOA = PROCESSED_DATA_DIR / "lsoa_primary_education_details.parquet"
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
    postcodes = pd.read_parquet(POSTCODE_FILE)
    merged = master.merge(postcodes, left_on='postcode', right_on='Postcode', how='inner')
    return gpd.GeoDataFrame(merged, geometry=gpd.points_from_xy(merged.longitude, merged.latitude),
                            crs="EPSG:4326").to_crs("EPSG:27700")


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


def process_nearest_schools(lsoa_gdf, schools_gdf, perf_df):
    print("  -> Calculating nearest neighbours...")
    school_coords = np.array(list(zip(schools_gdf.geometry.x, schools_gdf.geometry.y)))
    lsoa_coords = np.array(list(zip(lsoa_gdf.geometry.centroid.x, lsoa_gdf.geometry.centroid.y)))
    tree = cKDTree(school_coords)
    dists, indices = tree.query(lsoa_coords, k=NEAREST_N)
    map_rows = []
    for i, lsoa_code in enumerate(lsoa_gdf['area_code']):
        for rank in range(NEAREST_N):
            idx = indices[i][rank]
            map_rows.append({
                'area_code': lsoa_code, 'rank': rank + 1,
                'urn': str(schools_gdf.iloc[idx]['urn']),
                'name': schools_gdf.iloc[idx].get('name', 'Unknown'),
                'distance': dists[i][rank]
            })
    map_df = pd.DataFrame(map_rows)
    years_df = pd.DataFrame({'year': YEARS_TO_PROCESS})
    map_expanded = map_df.merge(years_df, how='cross')
    merged_df = map_expanded.merge(perf_df, on=['urn', 'year'], how='left')
    pivot_df = merged_df.pivot_table(index=['area_code', 'year'], columns='rank',
                                     values=['pass_rate', 'read_score', 'math_score', 'distance', 'name', 'urn'],
                                     aggfunc='first')
    pivot_df.columns = [f"primary_school_{col[1]}_{col[0]}" for col in pivot_df.columns]
    return pivot_df.reset_index()


if __name__ == "__main__":
    schools_gdf = load_school_locations()
    perf_df = load_school_performance()
    if schools_gdf is None or perf_df is None: sys.exit("Error loading data.")

    hist_df = perf_df.copy().rename(columns={'urn': 'URN', 'pass_rate': 'avg_ks2_pass_rate'})
    hist_df['URN'] = hist_df['URN'].astype(str)
    if 'read_score' in hist_df: hist_df['avg_primary_scaled_score'] = hist_df[['read_score', 'math_score']].mean(axis=1)
    hist_df.to_parquet(HISTORICAL_OUTPUT, index=False)

    lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE).to_crs("EPSG:27700")
    final_df = process_nearest_schools(lsoa_gdf, schools_gdf, perf_df)
    final_df.to_parquet(OUTPUT_FILE_LSOA, index=False)
    print(f"Saved annual scores to {OUTPUT_FILE_LSOA}")

    latest_year = final_df['year'].max()
    static_df = final_df[final_df['year'] == latest_year].copy()
    for i in range(1, NEAREST_N + 1):
        if f'primary_school_{i}_read_score' in static_df:
            static_df[f'primary_school_{i}_avg_scaled_score'] = static_df[
                [f'primary_school_{i}_read_score', f'primary_school_{i}_math_score']].mean(axis=1)

    # TYPO FIXED HERE: static -> static_df
    static_df.to_parquet(STATIC_DETAILS_OUTPUT_LSOA, index=False)
    print("Done.")