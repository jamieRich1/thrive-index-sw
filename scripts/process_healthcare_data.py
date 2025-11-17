import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree
import numpy as np
import sys
import re

print("Starting healthcare data processing (RAW EXTRACTION - NO IMPUTATION)...")

# Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "healthcare"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
GP_LOCATIONS_FILE = RAW_DATA_DIR / "epraccur.csv"
SURVEY_FILE_PATTERN = "GPPS_*.csv"
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
WARD_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_ward.geojson"
POSTCODE_FILE = PROCESSED_DATA_DIR / "sw_postcodes.parquet"
STATIC_DETAILS_OUTPUT_LSOA = PROCESSED_DATA_DIR / "lsoa_healthcare_details.parquet"
ANNUAL_SCORES_OUTPUT_LSOA = PROCESSED_DATA_DIR / "lsoa_annual_healthcare_scores.parquet"
STATIC_DETAILS_OUTPUT_WARD = PROCESSED_DATA_DIR / "ward_healthcare_details.parquet"
ANNUAL_SCORES_OUTPUT_WARD = PROCESSED_DATA_DIR / "ward_annual_healthcare_scores.parquet"
HISTORICAL_GP_SCORES_OUTPUT = PROCESSED_DATA_DIR / "gp_historical_satisfaction.parquet"
NEAREST_N_GPS = 3
YEARS_TO_PROCESS = list(range(2018, 2026))  # This is just for the master grid, not for ffill
LATEST_YEAR = 2025  # This is a placeholder, the code will find the real latest year


# (Helper functions extract_year_from_filename, load_all_satisfaction_data, etc. are identical to your original)
# ...
def extract_year_from_filename(filename):
    """Extracts the year from filenames like 'GPPS_2018_...'. Returns 2018."""
    match = re.search(r'_(\d{4})_', filename)
    if match:
        return int(match.group(1))
    match_2025 = re.search(r'GPPS_(\d{4})_Practice_data', filename)
    if match_2025:
        return int(match_2025.group(1))
    return None


def load_all_satisfaction_data():
    """
    Loads and row-binds all annual GP survey files (2018-2025).
    This logic is based on the definitive column names provided by the user.

    - 2024/2025: Uses 'overallexp.pcteval'
    - 2018-2023: Uses 'Q28_12pct'
    """
    print("  -> Loading all historical satisfaction files (User-Verified Logic)...")
    survey_files = list(RAW_DATA_DIR.glob(SURVEY_FILE_PATTERN))
    if not survey_files:
        print(f"ERROR: No GP survey files found matching '{SURVEY_FILE_PATTERN}' in {RAW_DATA_DIR}")
        return None

    all_survey_data = []
    for f in survey_files:
        year = extract_year_from_filename(f.name)
        if not year:
            print(f"Warning: Skipping file, cannot extract year: {f.name}")
            continue

        try:
            # Read headers to find columns
            headers = pd.read_csv(f, nrows=0, encoding='utf-8-sig').columns.str.strip().str.strip('"')
            headers_lower = headers.str.lower()
            org_col_name = None
            sat_col_name = None

            # 2024 & 2025 logic
            if year in [2024, 2025]:
                target_sat_col = 'overallexp.pcteval'
                target_org_col = 'ad_practicecode'

                if target_sat_col in headers_lower and target_org_col in headers_lower:
                    org_col_name = headers[headers_lower == target_org_col][0]
                    sat_col_name = headers[headers_lower == target_sat_col][0]
                    print(f"  -> Processing {f.name} (Year {year}), using col: {sat_col_name}")

            # 2018 - 2023 logic
            elif year in [2018, 2019, 2020, 2021, 2022, 2023]:
                target_sat_col = 'q28_12pct'
                target_org_col = 'practice_code'

                if target_sat_col in headers_lower and target_org_col in headers_lower:
                    org_col_name = headers[headers_lower == target_org_col][0]
                    sat_col_name = headers[headers_lower == target_sat_col][0]
                    print(f"  -> Processing {f.name} (Year {year}), using col: {sat_col_name}")

            # Column Check
            if not org_col_name or not sat_col_name:
                print(f"Warning: Skipping {f.name}. Could not find required columns for year {year}.")
                print(f"   (Looked for Org: '{target_org_col}', Sat: '{target_sat_col}')")
                continue

            # Load and Process Data
            df_full = pd.read_csv(f, encoding='utf-8-sig', low_memory=False)
            df_full.columns = df_full.columns.str.strip().str.strip('"')
            df = df_full[[org_col_name, sat_col_name]].copy()
            df.rename(columns={
                org_col_name: 'org_code',
                sat_col_name: 'sat_val'
            }, inplace=True)

            df['org_code'] = df['org_code'].str.strip()

            # Satisfaction Score
            sat_val = pd.to_numeric(df['sat_val'], errors='coerce')
            df['satisfaction_pct'] = sat_val * 100

            # Clean and Store
            df['satisfaction_pct'] = df['satisfaction_pct'].clip(0, 100)
            df.loc[df['satisfaction_pct'] > 100, 'satisfaction_pct'] = np.nan
            df.loc[df['satisfaction_pct'] < 0, 'satisfaction_pct'] = np.nan
            df['year'] = year
            all_survey_data.append(df[['org_code', 'year', 'satisfaction_pct']])
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    if not all_survey_data:
        print("ERROR: No valid GP survey data was loaded.")
        return None

    master_satisfaction_df = pd.concat(all_survey_data, ignore_index=True)
    master_satisfaction_df.dropna(subset=['org_code', 'satisfaction_pct'], inplace=True)
    print(f"  -> Loaded {len(master_satisfaction_df)} total satisfaction records from {len(survey_files)} files.")
    return master_satisfaction_df


def load_practice_locations():
    """
    Loads the master list of GP practices from epraccur.csv.
    """
    if not GP_LOCATIONS_FILE.exists():
        print(f"ERROR: GP locations file not found: {GP_LOCATIONS_FILE.name}")
        return None

    print("  -> Loading GP practice locations (epraccur.csv)...")
    try:
        gp_locs_df = pd.read_csv(
            GP_LOCATIONS_FILE,
            header=None,
            usecols=[0, 1, 9, 25],
            names=['org_code', 'name', 'postcode', 'setting_code'],
            dtype=str,
            encoding='latin-1'
        )
    except Exception as e:
        print(f"ERROR: Could not read {GP_LOCATIONS_FILE.name}. Check file encoding or format. Error: {e}")
        return None

    for col in ['org_code', 'name', 'postcode', 'setting_code']:
        gp_locs_df[col] = gp_locs_df[col].str.strip().str.strip('"')

    # Filter for active GP practices
    gp_locs_df = gp_locs_df[gp_locs_df['setting_code'] == '4'].copy()
    gp_locs_df = gp_locs_df[['org_code', 'name', 'postcode']]
    gp_locs_df.dropna(subset=['postcode'], inplace=True)
    gp_locs_df['postcode'] = gp_locs_df['postcode'].str.replace(' ', '').str.upper()
    print(f"  -> Loaded {len(gp_locs_df)} active GP practice locations.")
    return gp_locs_df


def geocode_practices(gp_locs_df):
    """
    Geocodes the provided dataframe of practices using the SW postcode file.
    """
    print("  -> Geocoding practices...")
    postcode_df = pd.read_parquet(POSTCODE_FILE, columns=['Postcode', 'latitude', 'longitude'])
    postcode_df.dropna(subset=['Postcode'], inplace=True)
    postcode_df['Postcode'] = postcode_df['Postcode'].str.strip().str.replace(' ', '').str.upper()

    gp_gdf = pd.merge(gp_locs_df, postcode_df, left_on='postcode', right_on='Postcode', how='inner')
    gp_gdf = gpd.GeoDataFrame(
        gp_gdf,
        geometry=gpd.points_from_xy(gp_gdf.longitude, gp_gdf.latitude),
        crs="EPSG:4326"
    )
    print(f"  -> Successfully geocoded {len(gp_gdf)} GP practices in the South West.")
    return gp_gdf.to_crs("EPSG:27700")


def process_area_data(area_gdf, schools_gdf_proj, master_satisfaction_df, area_code_col, latest_year_for_static):
    """
    Processes either LSOA or Ward data to find nearest schools and calculate scores.
    """
    print(f"\nStep 3: Finding {NEAREST_N_GPS} nearest 'existing' GPs for {len(area_gdf)} {area_code_col}s...")

    school_coords = np.array(list(schools_gdf_proj.geometry.apply(lambda p: (p.x, p.y))))
    kdtree = cKDTree(school_coords)

    area_gdf['centroid'] = area_gdf.geometry.centroid
    area_coords = np.array(list(area_gdf.centroid.apply(lambda p: (p.x, p.y))))

    distances, indices = kdtree.query(area_coords, k=NEAREST_N_GPS)

    print(f"Step 4: Generating static 'nearest 3' details file for {area_code_col}...")

    # Use the real latest year from the data for static details
    latest_perf_df = master_satisfaction_df[master_satisfaction_df['year'] == latest_year_for_static]
    gp_gdf_unproj = schools_gdf_proj.to_crs("EPSG:4326")

    static_details_results = []
    area_to_gp_map = []

    for i, area_row in area_gdf.iterrows():
        area_code = area_row[area_code_col]
        nearest_gp_indices = indices[i]
        nearest_gps_info = gp_gdf_unproj.iloc[nearest_gp_indices]

        avg_distance_km = distances[i].mean() / 1000
        result_row = {
            'area_code': area_code,
            'avg_distance_to_gp_km': avg_distance_km,
        }

        for n in range(NEAREST_N_GPS):
            gp_info = nearest_gps_info.iloc[n]
            gp_org_code = gp_info['org_code']

            area_to_gp_map.append({'area_code': area_code, 'org_code': gp_org_code})

            perf_info = latest_perf_df[latest_perf_df['org_code'] == gp_org_code]

            result_row[f'gp_{n + 1}_name'] = gp_info['name']
            result_row[f'gp_{n + 1}_org_code'] = gp_org_code

            if not perf_info.empty:
                result_row[f'gp_{n + 1}_satisfaction'] = perf_info['satisfaction_pct'].iloc[0]
                result_row[f'gp_{n + 1}_data_year'] = perf_info['year'].iloc[0]
            else:
                result_row[f'gp_{n + 1}_satisfaction'] = np.nan
                result_row[f'gp_{n + 1}_data_year'] = np.nan

        static_details_results.append(result_row)

    static_details_df = pd.DataFrame(static_details_results)

    for i in range(1, NEAREST_N_GPS + 1):
        static_details_df[f'gp_{i}_org_code'] = static_details_df[f'gp_{i}_org_code'].astype('string')
        static_details_df[f'gp_{i}_name'] = static_details_df[f'gp_{i}_name'].astype('string')

    print(f"Step 5: Generating annual time-series scores for {area_code_col}...")

    area_urn_map_df = pd.DataFrame(area_to_gp_map).drop_duplicates()
    annual_data = area_urn_map_df.merge(master_satisfaction_df, on='org_code', how='left')

    area_annual_scores = annual_data.groupby(['area_code', 'year'])['satisfaction_pct'].mean().reset_index()
    area_annual_scores = area_annual_scores.rename(columns={'satisfaction_pct': 'avg_gp_satisfaction'})

    master_index = pd.MultiIndex.from_product(
        [area_gdf[area_code_col].unique(), YEARS_TO_PROCESS],
        names=['area_code', 'year']
    )
    final_scores_df = pd.DataFrame(index=master_index).reset_index()
    final_scores_df = final_scores_df.merge(area_annual_scores, on=['area_code', 'year'], how='left')
    return static_details_df, final_scores_df


# Main
if __name__ == "__main__":

    print("Step 1: Loading all GP data...")
    master_satisfaction_df = load_all_satisfaction_data()
    if master_satisfaction_df is None:
        sys.exit(1)

    # Find the real latest year from the loaded data
    LATEST_YEAR_IN_DATA = master_satisfaction_df['year'].max()
    print(f"  -> Latest year found in survey data: {LATEST_YEAR_IN_DATA}")

    gp_locs_df = load_practice_locations()
    if gp_locs_df is None:
        sys.exit(1)

    print("\nStep 2: Filtering, geocoding, and saving GP data...")

    # Use the *real* latest year to filter practices
    latest_org_codes = master_satisfaction_df[
        master_satisfaction_df['year'] == LATEST_YEAR_IN_DATA
        ]['org_code'].unique()

    print(f"  -> Found {len(latest_org_codes)} practices in {LATEST_YEAR_IN_DATA} survey.")

    existing_practices_df = gp_locs_df[
        gp_locs_df['org_code'].isin(latest_org_codes)
    ].copy()

    print(
        f"  -> Found {len(existing_practices_df)} 'existing' practices (in epraccur AND {LATEST_YEAR_IN_DATA} survey).")

    gp_gdf_proj = geocode_practices(existing_practices_df)

    if gp_gdf_proj.empty:
        print("ERROR: No 'existing' GP practices could be geocoded.")
        sys.exit(1)

    invalids = master_satisfaction_df[
        (master_satisfaction_df['satisfaction_pct'] < 0) |
        (master_satisfaction_df['satisfaction_pct'] > 100)
        ]
    if not invalids.empty:
        print(f"Warning: {len(invalids)} invalid satisfaction values found. Replacing with NaN.")
        master_satisfaction_df.loc[invalids.index, 'satisfaction_pct'] = np.nan

    master_satisfaction_df.to_parquet(HISTORICAL_GP_SCORES_OUTPUT, index=False)
    print(f"  -> Saved historical satisfaction data for all GPs to {HISTORICAL_GP_SCORES_OUTPUT.name}")

    print("\n--- Processing LSOA Data ---")
    lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE).to_crs("EPSG:27700")
    lsoa_static_df, lsoa_annual_df = process_area_data(
        area_gdf=lsoa_gdf,
        schools_gdf_proj=gp_gdf_proj,
        master_satisfaction_df=master_satisfaction_df,
        area_code_col='area_code',
        latest_year_for_static=LATEST_YEAR_IN_DATA
    )

    if 'area_code' not in lsoa_annual_df.columns:
        if 'index' in lsoa_annual_df.columns:
            lsoa_annual_df.rename(columns={'index': 'area_code'}, inplace=True)
        else:
            # This is a fallback, but the logic should populate area_code correctly
            lsoa_annual_df['area_code'] = lsoa_annual_df['area_code'].iloc[0:len(lsoa_annual_df)]

    lsoa_static_df.to_parquet(STATIC_DETAILS_OUTPUT_LSOA, index=False)
    lsoa_annual_df.to_parquet(ANNUAL_SCORES_OUTPUT_LSOA, index=False)
    print(f"Success! Saved LSOA healthcare files (sparse).")

    print("\n--- Processing Ward Data ---")
    ward_gdf = gpd.read_file(WARD_BOUNDARIES_FILE).to_crs("EPSG:27700")
    ward_static_df, ward_annual_df = process_area_data(
        area_gdf=ward_gdf,
        schools_gdf_proj=gp_gdf_proj,
        master_satisfaction_df=master_satisfaction_df,
        area_code_col='WD25CD',
        latest_year_for_static=LATEST_YEAR_IN_DATA
    )
    ward_static_df.rename(columns={'area_code': 'ward_code'}, inplace=True)
    ward_annual_df.rename(columns={'area_code': 'ward_code'}, inplace=True)
    ward_static_df.to_parquet(STATIC_DETAILS_OUTPUT_WARD, index=False)
    ward_annual_df.to_parquet(ANNUAL_SCORES_OUTPUT_WARD, index=False)
    print(f"Success! Saved Ward healthcare files (sparse).")
    print("\nHealthcare script finished.")