import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree
import numpy as np
import re
from datetime import datetime
import sys

print("Starting CHILDCARE Deep Dive Data Processing (No Imputation)...")

# Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "childcare"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
CHILDCARE_FILE_PATTERN = "Management_information_-_childcare_providers*.csv"
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
POSTCODE_FILE = PROCESSED_DATA_DIR / "sw_postcodes.parquet"
OUTPUT_FILE = PROCESSED_DATA_DIR / "lsoa_annual_childcare_deepdive.parquet"
NEAREST_N = 3

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
    # Numeric conversion for Places
    if 'places' in master_df.columns:
        master_df['places'] = pd.to_numeric(master_df['places'].str.replace(',', ''), errors='coerce').fillna(0).astype(
            int)
    else:
        master_df['places'] = 0
    # Standardize Quality Score
    if 'rating_str' in master_df.columns:
        rating_map = {'1': 4, '2': 3, '3': 2, '4': 1}
        master_df['quality_score'] = master_df['rating_str'].astype(str).str[0].map(rating_map)
    else:
        master_df['quality_score'] = np.nan
    # Deduplicate: Keep latest file entry for each URN/Year
    master_df = master_df.sort_values('file_date').drop_duplicates(subset=['urn', 'year'], keep='last')
    return master_df

def geocode_childcare(df):
    print("  -> Geocoding childcare...")
    if not POSTCODE_FILE.exists():
        print("Error: Postcode file not found.")
        return gpd.GeoDataFrame()
    postcodes = pd.read_parquet(POSTCODE_FILE)
    df['postcode_clean'] = df['postcode'].str.replace(' ', '').str.upper()
    postcodes['Postcode'] = postcodes['Postcode'].str.replace(' ', '').str.upper()
    merged = df.merge(postcodes, left_on='postcode_clean', right_on='Postcode', how='inner')
    return gpd.GeoDataFrame(
        merged, geometry=gpd.points_from_xy(merged.longitude, merged.latitude), crs="EPSG:4326"
    ).to_crs("EPSG:27700")

def process_nearest_childcare_deepdive(lsoa_gdf, childcare_gdf):
    print("  -> Calculating nearest childcare (Raw Years Only)...")
    lsoa_coords = np.array(list(zip(lsoa_gdf.geometry.centroid.x, lsoa_gdf.geometry.centroid.y)))
    all_years_rows = []
    # Get years present in raw data
    available_years = sorted(childcare_gdf['year'].unique())
    print(f"     Processing years: {available_years}")
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
                # Append raw record
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
    if not all_years_rows:
        return pd.DataFrame()
    long_df = pd.DataFrame(all_years_rows)
    # Pivot to Wide Format
    pivot_df = long_df.pivot_table(
        index=['area_code', 'year'],
        columns='rank',
        values=['quality_score', 'places', 'distance', 'name', 'rating_str', 'urn'],
        aggfunc='first'
    )
    # Flatten columns
    pivot_df.columns = [f"childcare_{col[1]}_{col[0]}" for col in pivot_df.columns]
    # Return as-is (NO FILLING/IMPUTATION)
    return pivot_df.reset_index()

# Main
if __name__ == "__main__":
    if not LSOA_BOUNDARIES_FILE.exists():
        sys.exit(f"Error: LSOA boundaries not found at {LSOA_BOUNDARIES_FILE}")
    raw_df = load_childcare_data()
    if raw_df is None or raw_df.empty:
        sys.exit("Error loading childcare data.")
    gdf = geocode_childcare(raw_df)
    if gdf.empty:
        sys.exit("Error geocoding childcare data.")
    lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE).to_crs("EPSG:27700")
    final_df = process_nearest_childcare_deepdive(lsoa_gdf, gdf)
    print(f"Saving {len(final_df)} rows to {OUTPUT_FILE}...")
    final_df.to_parquet(OUTPUT_FILE, index=False)
    print("Done.")