import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree
import numpy as np
import re
from datetime import datetime
import sys

print("Starting childcare data processing")

# Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "childcare"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
CHILDCARE_FILE_PATTERN = "Management_information_-_childcare_providers*.csv"
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
POSTCODE_FILE = PROCESSED_DATA_DIR / "sw_postcodes.parquet"
ANNUAL_SCORES_OUTPUT = PROCESSED_DATA_DIR / "lsoa_annual_childcare_scores.parquet"
STATIC_DETAILS_OUTPUT = PROCESSED_DATA_DIR / "lsoa_childcare_details.parquet"
HISTORICAL_OUTPUT = PROCESSED_DATA_DIR / "childcare_historical_data.parquet"
NEAREST_N = 3
# Force these years to exist in the output
YEARS_TO_PROCESS = list(range(2018, 2026))

def extract_date_from_filename(filename):
    match = re.search(r'as_at_(\d{1,2})_(\w+)_(\d{4})', filename)
    if not match: return None, None
    try:
        dt = datetime.strptime(f"{match.group(1)} {match.group(2)} {match.group(3)}", "%d %B %Y")
        return dt, dt.year
    except ValueError:
        return None, None

def load_childcare_data():
    print("  -> Loading childcare files...")
    files = list(RAW_DATA_DIR.glob(CHILDCARE_FILE_PATTERN))
    all_data = []
    target_cols = {
        'provider urn': 'urn', 'provider name': 'name', 'provider postcode': 'postcode',
        'most recent full: overall effectiveness': 'rating_str', 'places': 'places'
    }
    for f in files:
        dt, year = extract_date_from_filename(f.name)
        if not year: continue
        try:
            headers_actual = pd.read_csv(f, nrows=0, encoding='utf-8-sig').columns
            headers_map = {h.lower(): h for h in headers_actual}
            use_cols = []
            rename_map = {}
            for target_low, target_clean in target_cols.items():
                if target_low in headers_map:
                    actual = headers_map[target_low]
                    use_cols.append(actual)
                    rename_map[actual] = target_clean
            if len(use_cols) < len(target_cols):
                if 'urn' not in rename_map.values() or 'postcode' not in rename_map.values():
                    print(f"Skipping {f.name}: Missing URN or Postcode.")
                    continue
            df = pd.read_csv(f, usecols=use_cols, encoding='utf-8-sig', dtype=str, low_memory=False)
            df.rename(columns=rename_map, inplace=True)
            df['year'] = year
            df['file_date'] = dt
            all_data.append(df)
        except Exception as e:
            print(f"Skipping {f.name}: {e}")

    if not all_data: return None
    master_df = pd.concat(all_data, ignore_index=True)
    if 'places' in master_df.columns:
        master_df['places'] = pd.to_numeric(master_df['places'].str.replace(',', ''), errors='coerce').fillna(0).astype(
            int)
    else:
        master_df['places'] = 0
    if 'rating_str' in master_df.columns:
        rating_map = {'1': 4, '2': 3, '3': 2, '4': 1}
        master_df['quality_score'] = master_df['rating_str'].astype(str).str[0].map(rating_map)
    else:
        master_df['quality_score'] = np.nan
    master_df = master_df.sort_values('file_date').drop_duplicates(subset=['urn', 'year'], keep='last')
    return master_df

def geocode_childcare(df):
    print("  -> Geocoding childcare...")
    postcodes = pd.read_parquet(POSTCODE_FILE)
    df['postcode_clean'] = df['postcode'].str.replace(' ', '').str.upper()
    postcodes['Postcode'] = postcodes['Postcode'].str.replace(' ', '').str.upper()
    merged = df.merge(postcodes, left_on='postcode_clean', right_on='Postcode', how='inner')
    return gpd.GeoDataFrame(
        merged, geometry=gpd.points_from_xy(merged.longitude, merged.latitude), crs="EPSG:4326"
    ).to_crs("EPSG:27700")

def process_nearest_childcare(lsoa_gdf, childcare_gdf):
    print("  -> Calculating nearest childcare...")
    lsoa_coords = np.array(list(zip(lsoa_gdf.geometry.centroid.x, lsoa_gdf.geometry.centroid.y)))
    all_years_rows = []

    # Process available years
    available_years = sorted(childcare_gdf['year'].unique())
    print(f"     Found data for years: {available_years}")
    for year in available_years:
        providers = childcare_gdf[childcare_gdf['year'] == year]
        if providers.empty: continue
        prov_coords = np.array(list(zip(providers.geometry.x, providers.geometry.y)))
        tree = cKDTree(prov_coords)
        dists, indices = tree.query(lsoa_coords, k=NEAREST_N)
        for i, lsoa_code in enumerate(lsoa_gdf['area_code']):
            for rank in range(NEAREST_N):
                idx = indices[i][rank]
                dist = dists[i][rank] / 1000.0
                prov = providers.iloc[idx]
                all_years_rows.append({
                    'area_code': lsoa_code,
                    'year': year,
                    'rank': rank + 1,
                    'urn': prov['urn'],
                    'name': prov['name'],
                    'quality_score': prov['quality_score'],
                    'rating_str': prov.get('rating_str', 'N/A'),
                    'places': prov['places'],
                    'distance': dist
                })
    long_df = pd.DataFrame(all_years_rows)

    # Pivot to Wide Format
    pivot_df = long_df.pivot_table(
        index=['area_code', 'year'],
        columns='rank',
        values=['quality_score', 'places', 'distance', 'name', 'rating_str', 'urn'],
        aggfunc='first'
    )
    pivot_df.columns = [f"childcare_{col[1]}_{col[0]}" for col in pivot_df.columns]
    pivot_df = pivot_df.reset_index()

    # Fill Missing Years
    print("  -> Filling missing years (Forward/Back filling)...")
    lsoas = pivot_df['area_code'].unique()
    full_index = pd.MultiIndex.from_product([lsoas, YEARS_TO_PROCESS], names=['area_code', 'year'])
    pivot_df = pivot_df.set_index(['area_code', 'year']).reindex(full_index)
    pivot_df = pivot_df.groupby('area_code').ffill().bfill()
    return pivot_df.reset_index()

if __name__ == "__main__":
    raw_df = load_childcare_data()
    if raw_df is None: sys.exit("Error loading data.")
    gdf = geocode_childcare(raw_df)
    gdf[['urn', 'year', 'name', 'rating_str', 'places']].rename(
        columns={'urn': 'Provider URN', 'name': 'Provider Name', 'rating_str': 'quality_rating'}
    ).to_parquet(HISTORICAL_OUTPUT, index=False)
    lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE).to_crs("EPSG:27700")
    wide_df = process_nearest_childcare(lsoa_gdf, gdf)
    wide_df.to_parquet(ANNUAL_SCORES_OUTPUT, index=False)
    print(f"Saved annual scores to {ANNUAL_SCORES_OUTPUT}")
    latest_year = wide_df['year'].max()
    static_df = wide_df[wide_df['year'] == latest_year].copy()
    rename_map = {}
    for i in range(1, NEAREST_N + 1):
        rename_map[f'childcare_{i}_rating_str'] = f'childcare_{i}_quality_rating'
        rename_map[f'childcare_{i}_distance'] = f'childcare_{i}_distance_km'
    static_df.rename(columns=rename_map, inplace=True)
    static_df.to_parquet(STATIC_DETAILS_OUTPUT, index=False)
    print("Done.")