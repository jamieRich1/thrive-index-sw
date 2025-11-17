import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree
import numpy as np
import sys
import re

print("Starting SECONDARY school (KS4) data processing (RAW EXTRACTION - NO IMPUTATION)...")

# Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "schools"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
WARD_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_ward.geojson"
POSTCODE_FILE = PROCESSED_DATA_DIR / "sw_postcodes.parquet"
ANNUAL_SCORES_OUTPUT_LSOA = PROCESSED_DATA_DIR / "lsoa_annual_secondary_scores.parquet"
STATIC_DETAILS_OUTPUT_LSOA = PROCESSED_DATA_DIR / "lsoa_secondary_education_details.parquet"
ANNUAL_SCORES_OUTPUT_WARD = PROCESSED_DATA_DIR / "ward_annual_secondary_scores.parquet"
STATIC_DETAILS_OUTPUT_WARD = PROCESSED_DATA_DIR / "ward_secondary_education_details.parquet"
NEAREST_N_SCHOOLS = 3
YEARS_TO_PROCESS = list(range(2018, 2026))  # This is just for the master grid, not for ffill
URN_COL = 'URN'
SCHNAME_COL = 'SCHNAME'
PCODE_COL = 'PCODE'
NFTYPE_COL = 'NFTYPE'
PROGRESS_8_COL = 'P8MEA'
ATTAINMENT_8_COL = 'ATT8SCR'
PERF_COLS = [PROGRESS_8_COL, ATTAINMENT_8_COL]
SENTINEL_VALUES = ['SUPP', 'NE', 'LOWCOV', 'NA', '#N/A', -9.99, -2.99, -0.38, 'NEW', 'NP']


# (Helper functions load_school_locations, load_school_performance, etc. are identical to your original)
# ...
def extract_year_from_filename(filename):
    """Extracts the end year from filenames like '2017-2018_...'. Returns 2018."""
    match = re.search(r'(\d{4})-(\d{4})', filename)
    if match:
        return int(match.group(2))
    match_single = re.search(r'(\d{4})', filename)
    if match_single:
        if '-' in filename:
            match_end = re.search(r'-(\d{4})', filename)
            if match_end:
                return int(match_end.group(1))
        return int(match_single.group(1))
    return None


def load_school_locations():
    """
    Loads all FINAL (non-provisional) KS4 files, builds a master list of schools,
    and geocodes their *most recent* final location.
    """
    print("  -> Locating all non-provisional KS4 files...")
    ks4_files = [f for f in RAW_DATA_DIR.glob("*ks4*.csv") if 'provisional' not in f.name.lower()]
    if not ks4_files:
        print(f"ERROR: No FINAL (non-provisional) KS4 files found in {RAW_DATA_DIR}")
        return None
    print(f"  -> Found {len(ks4_files)} final KS4 files for geocoding.")
    all_school_data = []
    cols_to_use = [URN_COL, SCHNAME_COL, PCODE_COL, NFTYPE_COL]

    for f in ks4_files:
        year = extract_year_from_filename(f.name)
        if not year:
            print(f"Warning: Skipping file, cannot extract year: {f.name}")
            continue

        try:
            all_headers = pd.read_csv(f, encoding='latin-1', nrows=0).columns
            cols_to_load = [col for col in cols_to_use if col in all_headers]

            if NFTYPE_COL not in all_headers:
                print(f"  -> INFO: {f.name} does not contain NFTYPE column. Will fill with 'NA'.")

            df = pd.read_csv(f, usecols=cols_to_load, encoding='latin-1', low_memory=False, dtype={URN_COL: str})

            if NFTYPE_COL not in df.columns:
                df[NFTYPE_COL] = 'NA'

            df['year'] = year
            all_school_data.append(df)
        except ValueError:
            print(f"Warning: Could not find required columns in {f.name}.")
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    if not all_school_data:
        print("ERROR: No valid KS4 school data was loaded.")
        return None

    master_school_list = pd.concat(all_school_data, ignore_index=True)
    master_school_list = master_school_list.sort_values(by='year', ascending=False)
    master_school_locations = master_school_list.drop_duplicates(subset=[URN_COL], keep='first')

    print("  -> Geocoding school locations...")
    postcode_df = pd.read_parquet(POSTCODE_FILE)
    postcode_gdf = gpd.GeoDataFrame(
        postcode_df,
        geometry=gpd.points_from_xy(postcode_df.longitude, postcode_df.latitude),
        crs="EPSG:4258"
    )

    postcode_gdf = postcode_gdf.to_crs("EPSG:4326")
    schools_gdf = master_school_locations.merge(
        postcode_gdf[['Postcode', 'geometry']],
        left_on=PCODE_COL,
        right_on='Postcode',
        how='inner'
    )
    schools_gdf = gpd.GeoDataFrame(schools_gdf, geometry='geometry', crs="EPSG:4326")
    print(f"  -> Geocoded {len(schools_gdf)} unique secondary schools.")
    return schools_gdf.to_crs("EPSG:27700")


def load_school_performance():
    """
    Loads performance data (scores) from all FINAL KS4 files
    and adds a 'year' column.
    """
    print("  -> Loading all historical performance data (KS4)...")
    ks4_files = [f for f in RAW_DATA_DIR.glob("*ks4*.csv") if 'provisional' not in f.name.lower()]
    all_perf_data = []
    cols_to_use = [URN_COL] + [col for col in PERF_COLS if col]

    for f in ks4_files:
        year = extract_year_from_filename(f.name)
        if not year:
            continue
        try:
            df = pd.read_csv(f, usecols=lambda c: c in cols_to_use, encoding='latin-1', low_memory=False,
                             dtype={URN_COL: str})
            df['year'] = year
            all_perf_data.append(df)
        except ValueError:
            print(f"Warning: Could not find performance columns in {f.name}.")
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    if not all_perf_data:
        print("ERROR: No valid KS4 performance data was loaded.")
        return None

    master_perf_df = pd.concat(all_perf_data, ignore_index=True)

    for col in PERF_COLS:
        if col in master_perf_df.columns:
            master_perf_df[col] = master_perf_df[col].astype(str).str.replace(r'[^0-9\.-]', '', regex=True)
            master_perf_df[col] = pd.to_numeric(
                master_perf_df[col].replace(SENTINEL_VALUES, np.nan),
                errors='coerce'
            )
    master_perf_df = master_perf_df.rename(columns={
        PROGRESS_8_COL: 'avg_progress_8',
        ATTAINMENT_8_COL: 'avg_attainment_8'
    })
    print(f"  -> Loaded {len(master_perf_df)} valid performance records.")
    return master_perf_df[[URN_COL, 'year', 'avg_progress_8', 'avg_attainment_8']]


def process_area_data(area_gdf, schools_gdf_proj, performance_df, area_code_col):
    """
    Processes either LSOA or Ward data to find nearest schools and calculate scores.
    """
    print(f"\nStep 3: Finding {NEAREST_N_SCHOOLS} nearest secondary schools for {len(area_gdf)} {area_code_col}s...")

    # Create spatial index for schools
    school_coords = np.array(list(schools_gdf_proj.geometry.apply(lambda p: (p.x, p.y))))
    kdtree = cKDTree(school_coords)

    # Get centroids for the areas (LSOAs or Wards)
    area_gdf['centroid'] = area_gdf.geometry.centroid
    area_coords = np.array(list(area_gdf.centroid.apply(lambda p: (p.x, p.y))))

    # Nearest Schools
    distances_m, indices = kdtree.query(area_coords, k=NEAREST_N_SCHOOLS)
    print(f"Step 4: Generating static 'nearest 3' details file for {area_code_col}...")
    latest_perf_df = performance_df.sort_values(by='year').drop_duplicates(subset=[URN_COL], keep='last')
    static_details_results = []

    for i, area_row in area_gdf.iterrows():
        nearest_school_indices = indices[i]
        nearest_schools_info = schools_gdf_proj.iloc[nearest_school_indices]
        result_row = {'area_code': area_row[area_code_col]}

        for n in range(NEAREST_N_SCHOOLS):
            school_info = nearest_schools_info.iloc[n]
            school_urn = school_info[URN_COL]
            perf_info = latest_perf_df[latest_perf_df[URN_COL] == school_urn]
            result_row[f'school_{n + 1}_name'] = school_info[SCHNAME_COL]
            result_row[f'school_{n + 1}_urn'] = school_urn
            result_row[f'school_{n + 1}_nftype'] = school_info[NFTYPE_COL]
            if not perf_info.empty:
                result_row[f'school_{n + 1}_progress_8'] = perf_info['avg_progress_8'].iloc[0]
                result_row[f'school_{n + 1}_attainment_8'] = perf_info['avg_attainment_8'].iloc[0]
                result_row[f'school_{n + 1}_data_year'] = perf_info['year'].iloc[0]
            else:
                result_row[f'school_{n + 1}_progress_8'] = np.nan
                result_row[f'school_{n + 1}_attainment_8'] = np.nan
                result_row[f'school_{n + 1}_data_year'] = np.nan

        static_details_results.append(result_row)
    static_details_df = pd.DataFrame(static_details_results)

    for i in range(1, NEAREST_N_SCHOOLS + 1):
        static_details_df[f'school_{i}_urn'] = static_details_df[f'school_{i}_urn'].astype('string')
        static_details_df[f'school_{i}_name'] = static_details_df[f'school_{i}_name'].astype('string')
        static_details_df[f'school_{i}_nftype'] = static_details_df[f'school_{i}_nftype'].astype('string')

    # Annual Scores for Timeseries and forward fill for covid period missing data
    print(f"Step 5: Generating annual time-series scores for {area_code_col}...")
    area_urn_mapping = []
    area_urn_map_df_base = []
    for i, area_row in area_gdf.iterrows():
        for n in range(NEAREST_N_SCHOOLS):
            school_info = schools_gdf_proj.iloc[indices[i][n]]
            area_urn_map_df_base.append({
                'area_code': area_row[area_code_col],
                URN_COL: school_info[URN_COL],
                NFTYPE_COL: school_info[NFTYPE_COL]
            })
    area_urn_map_df = pd.DataFrame(area_urn_map_df_base).drop_duplicates()

    print(f"  -> Found {len(area_urn_map_df)} total area-school links.")
    area_urn_map_df_state_schools = area_urn_map_df[area_urn_map_df[NFTYPE_COL] != 'IND']
    print(f"  -> Filtering to {len(area_urn_map_df_state_schools)} state-school links for scoring.")
    annual_data = area_urn_map_df_state_schools.merge(performance_df, on=URN_COL, how='left')

    # Aggregate only the raw, non-imputed data
    area_annual_scores = annual_data.groupby(['area_code', 'year'])[
        ['avg_progress_8', 'avg_attainment_8']
    ].mean().reset_index()

    # Create the full 2018-2025 grid
    master_index = pd.MultiIndex.from_product(
        [area_gdf[area_code_col].unique(), YEARS_TO_PROCESS],
        names=['area_code', 'year']
    )
    final_scores_df = pd.DataFrame(index=master_index).reset_index()
    # Merge the sparse scores, which will leave NaNs
    final_scores_df = final_scores_df.merge(area_annual_scores, on=['area_code', 'year'], how='left')
    return static_details_df, final_scores_df


# Main
if __name__ == "__main__":
    print("Step 1: Loading and geocoding all secondary schools...")
    schools_gdf_proj = load_school_locations()
    if schools_gdf_proj is None:
        sys.exit(1)

    print("\nStep 2: Loading all performance data...")
    ks4_performance_df = load_school_performance()
    if ks4_performance_df is None:
        sys.exit(1)

    print("  -> Saving raw (gappy) historical school data...")
    all_school_urns = schools_gdf_proj[URN_COL].unique()
    master_school_index = pd.MultiIndex.from_product(
        [all_school_urns, YEARS_TO_PROCESS],
        names=[URN_COL, 'year']
    )
    historical_df = pd.DataFrame(index=master_school_index).reset_index()
    historical_df = historical_df.merge(
        ks4_performance_df,
        on=[URN_COL, 'year'],
        how='left'
    )
    historical_df = historical_df.sort_values(by=[URN_COL, 'year'])
    historical_df = historical_df.rename(columns={'URN': 'urn'})
    historical_df.to_parquet(
        PROCESSED_DATA_DIR / "secondary_school_historical_data.parquet",
        index=False
    )
    print(f"Success! Saved raw (gappy) historical secondary school data.")
    print("\n--- Processing LSOA Data ---")
    lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE).to_crs("EPSG:27700")
    lsoa_static_df, lsoa_annual_df = process_area_data(
        area_gdf=lsoa_gdf,
        schools_gdf_proj=schools_gdf_proj,
        performance_df=ks4_performance_df,
        area_code_col='area_code'
    )
    lsoa_static_df.to_parquet(STATIC_DETAILS_OUTPUT_LSOA, index=False)
    lsoa_annual_df.to_parquet(ANNUAL_SCORES_OUTPUT_LSOA, index=False)
    print(f"Success! Saved LSOA secondary school files (sparse).")
    print("\n--- Processing Ward Data ---")
    ward_gdf = gpd.read_file(WARD_BOUNDARIES_FILE).to_crs("EPSG:27700")
    ward_static_df, ward_annual_df = process_area_data(
        area_gdf=ward_gdf,
        schools_gdf_proj=schools_gdf_proj,
        performance_df=ks4_performance_df,
        area_code_col='WD25CD'
    )
    ward_static_df.rename(columns={'area_code': 'ward_code'}, inplace=True)
    ward_annual_df.rename(columns={'area_code': 'ward_code'}, inplace=True)
    ward_static_df.to_parquet(STATIC_DETAILS_OUTPUT_WARD, index=False)
    ward_annual_df.to_parquet(ANNUAL_SCORES_OUTPUT_WARD, index=False)
    print(f"Success! Saved Ward secondary school files (sparse).")
    print("\nSecondary school script finished.")