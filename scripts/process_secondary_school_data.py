import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree
import numpy as np
import sys
import re

print("Starting SECONDARY school (KS4) data processing (FINAL FIX)...")

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "schools"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
POSTCODE_FILE = PROCESSED_DATA_DIR / "sw_postcodes.parquet"
OUTPUT_FILE_LSOA = PROCESSED_DATA_DIR / "lsoa_annual_secondary_scores.parquet"
STATIC_DETAILS_OUTPUT_LSOA = PROCESSED_DATA_DIR / "lsoa_secondary_education_details.parquet"
HISTORICAL_OUTPUT = PROCESSED_DATA_DIR / "secondary_school_historical_data.parquet"

NEAREST_N = 3
YEARS_TO_PROCESS = list(range(2018, 2026))

# Columns verified by your audit
LOC_COLS = ['URN', 'SCHNAME', 'PCODE', 'NFTYPE']
PERF_COLS = ['URN', 'P8MEA', 'ATT8SCR']
SENTINEL_VALUES = ['SUPP', 'NE', 'LOWCOV', 'NA', '#N/A', 'NEW', 'NP', '', 'nan']


def extract_year_from_filename(filename):
    match = re.search(r'(\d{4})-(\d{4})', filename)
    return int(match.group(2)) if match else None


def load_school_locations():
    print("  -> Loading school locations...")
    files = [f for f in RAW_DATA_DIR.glob("*ks4*.csv") if 'provisional' not in f.name.lower()]
    all_locs = []
    for f in files:
        year = extract_year_from_filename(f.name)
        if not year: continue
        try:
            df = pd.read_csv(f, usecols=LOC_COLS, encoding='latin-1', dtype=str, low_memory=False)
            df.rename(columns={'URN': 'urn', 'SCHNAME': 'name', 'PCODE': 'postcode', 'NFTYPE': 'nftype'}, inplace=True)

            # Force URN to string and remove NaNs
            df['urn'] = df['urn'].astype(str)
            df = df[df['urn'] != 'nan']

            df['year'] = year
            all_locs.append(df)
        except:
            pass
    if not all_locs: return None
    master = pd.concat(all_locs).sort_values('year').drop_duplicates('urn', keep='last')

    postcodes = pd.read_parquet(POSTCODE_FILE)
    merged = master.merge(postcodes, left_on='postcode', right_on='Postcode', how='inner')
    return gpd.GeoDataFrame(merged, geometry=gpd.points_from_xy(merged.longitude, merged.latitude),
                            crs="EPSG:4326").to_crs("EPSG:27700")


def load_school_performance():
    print("  -> Loading performance data...")
    files = [f for f in RAW_DATA_DIR.glob("*ks4*.csv") if 'provisional' not in f.name.lower()]
    all_perf = []
    for f in files:
        year = extract_year_from_filename(f.name)
        if not year: continue
        try:
            df = pd.read_csv(f, usecols=PERF_COLS, encoding='latin-1', dtype=str, low_memory=False)
            df.rename(columns={'URN': 'urn', 'P8MEA': 'progress_8', 'ATT8SCR': 'attainment_8'}, inplace=True)

            # Force URN to string
            df['urn'] = df['urn'].astype(str)
            df = df[df['urn'] != 'nan']

            for c in ['progress_8', 'attainment_8']:
                df[c] = pd.to_numeric(df[c].replace(SENTINEL_VALUES, np.nan), errors='coerce')
            df['year'] = year
            all_perf.append(df)
        except:
            pass
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
                'area_code': lsoa_code,
                'rank': rank + 1,
                'urn': str(schools_gdf.iloc[idx]['urn']),
                'name': schools_gdf.iloc[idx].get('name', 'Unknown'),
                'nftype': schools_gdf.iloc[idx].get('nftype', 'NA'),
                'distance': dists[i][rank]
            })
    map_df = pd.DataFrame(map_rows)

    # Cross Join
    years_df = pd.DataFrame({'year': YEARS_TO_PROCESS})
    map_expanded = map_df.merge(years_df, how='cross')

    # Merge keys are now strictly strings
    merged_df = map_expanded.merge(perf_df, on=['urn', 'year'], how='left')

    pivot_df = merged_df.pivot_table(index=['area_code', 'year'], columns='rank',
                                     values=['progress_8', 'attainment_8', 'distance', 'name', 'nftype', 'urn'],
                                     aggfunc='first')
    pivot_df.columns = [f"school_{col[1]}_{col[0]}" for col in pivot_df.columns]
    return pivot_df.reset_index()


if __name__ == "__main__":
    schools_gdf = load_school_locations()
    perf_df = load_school_performance()
    if schools_gdf is None or perf_df is None: sys.exit("Error.")

    hist_df = perf_df.copy().rename(
        columns={'urn': 'URN', 'progress_8': 'avg_progress_8', 'attainment_8': 'avg_attainment_8'})
    hist_df.to_parquet(HISTORICAL_OUTPUT, index=False)

    lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE).to_crs("EPSG:27700")
    final_df = process_nearest_schools(lsoa_gdf, schools_gdf, perf_df)
    final_df.to_parquet(OUTPUT_FILE_LSOA, index=False)

    static_df = final_df[final_df['year'] == final_df['year'].max()].copy()
    static_df.to_parquet(STATIC_DETAILS_OUTPUT_LSOA, index=False)
    print("Done.")