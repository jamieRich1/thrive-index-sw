import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree
import numpy as np
import sys
import re

print("Starting PRIMARY school (KS2) data processing")

#Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "schools"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
WARD_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_ward.geojson"
POSTCODE_FILE = PROCESSED_DATA_DIR / "sw_postcodes.parquet"
ANNUAL_SCORES_OUTPUT_LSOA = PROCESSED_DATA_DIR / "lsoa_annual_primary_scores.parquet"
STATIC_DETAILS_OUTPUT_LSOA = PROCESSED_DATA_DIR / "lsoa_primary_education_details.parquet"
ANNUAL_SCORES_OUTPUT_WARD = PROCESSED_DATA_DIR / "ward_annual_primary_scores.parquet"
STATIC_DETAILS_OUTPUT_WARD = PROCESSED_DATA_DIR / "ward_primary_education_details.parquet"
NEAREST_N_SCHOOLS = 3
YEARS_TO_PROCESS = list(range(2018, 2026))
URN_COL = 'URN'
SCHNAME_COL = 'SCHNAME'
PCODE_COL = 'PCODE'
PASS_RATE_COL = 'PTRWM_EXP'
READ_SCORE_COL = 'READ_AVERAGE'
MATH_SCORE_COL = 'MAT_AVERAGE'
PERF_COLS = [PASS_RATE_COL, READ_SCORE_COL, MATH_SCORE_COL]
SENTINEL_VALUES = ['SUPP', 'NE', 'LOWCOV', 'NA', '#N/A', -9.99, -2.99, -0.38, 'NEW', 'NP']

#Helpers
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
    Loads all FINAL (non-provisional) KS2 files, builds a master list of schools,
    and geocodes their *most recent* final location.
    """
    print("  -> Locating all non-provisional KS2 files...")
    ks2_files = [f for f in RAW_DATA_DIR.glob("*ks2*.csv") if 'provisional' not in f.name.lower()]
    if not ks2_files:
        print(f"ERROR: No FINAL (non-provisional) KS2 files found in {RAW_DATA_DIR}")
        return None
    print(f"  -> Found {len(ks2_files)} final KS2 files for geocoding.")

    all_school_data = []
    cols_to_use = [URN_COL, SCHNAME_COL, PCODE_COL]

    for f in ks2_files:
        year = extract_year_from_filename(f.name)
        if not year:
            print(f"Warning: Skipping file, cannot extract year: {f.name}")
            continue
        try:
            df = pd.read_csv(f, usecols=cols_to_use, encoding='latin-1', low_memory=False, dtype={URN_COL: str})
            df['year'] = year
            all_school_data.append(df)
        except ValueError:
            print(f"Warning: Could not find required columns in {f.name}. Check {cols_to_use}.")
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    if not all_school_data:
        print("ERROR: No valid KS2 school data was loaded.")
        return None

    master_school_list = pd.concat(all_school_data, ignore_index=True)
    master_school_list = master_school_list.sort_values(by='year', ascending=False)
    master_school_locations = master_school_list.drop_duplicates(subset=[URN_COL], keep='first')

    print("  -> Geocoding school locations...")
    postcode_df = pd.read_parquet(POSTCODE_FILE)
    postcode_gdf = gpd.GeoDataFrame(
        postcode_df,
        geometry=gpd.points_from_xy(postcode_df.longitude, postcode_df.latitude),
        crs="EPSG:4326"
    )

    schools_gdf = master_school_locations.merge(
        postcode_gdf[['Postcode', 'geometry']],
        left_on=PCODE_COL,
        right_on='Postcode',
        how='inner'
    )
    schools_gdf = gpd.GeoDataFrame(schools_gdf, geometry='geometry', crs="EPSG:4326")

    print(f"  -> Geocoded {len(schools_gdf)} unique primary schools.")
    return schools_gdf.to_crs("EPSG:27700")


def load_school_performance():
    """
    Loads performance data (scores) from all FINAL KS2 files
    and adds a 'year' column.
    """
    print("  -> Loading all historical performance data (KS2)...")
    ks2_files = [f for f in RAW_DATA_DIR.glob("*ks2*.csv") if 'provisional' not in f.name.lower()]
    all_perf_data = []

    cols_to_use = [URN_COL] + [col for col in PERF_COLS if col]

    for f in ks2_files:
        year = extract_year_from_filename(f.name)
        if not year:
            continue
        try:
            # *** THIS IS THE FIX: Read URN as string ***
            df = pd.read_csv(f, usecols=lambda c: c in cols_to_use, encoding='latin-1', low_memory=False,
                             dtype={URN_COL: str})
            df['year'] = year
            all_perf_data.append(df)
        except ValueError:
            print(f"Warning: Could not find performance columns in {f.name}.")
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    if not all_perf_data:
        print("ERROR: No valid KS2 performance data was loaded.")
        return None

    master_perf_df = pd.concat(all_perf_data, ignore_index=True)

    for col in PERF_COLS:
        if col in master_perf_df.columns:
            master_perf_df[col] = master_perf_df[col].astype(str).str.replace(r'[^0-9\.-]', '', regex=True)
            master_perf_df[col] = pd.to_numeric(
                master_perf_df[col].replace(SENTINEL_VALUES, np.nan),
                errors='coerce'
            )

    master_perf_df['avg_primary_scaled_score'] = master_perf_df[[READ_SCORE_COL, MATH_SCORE_COL]].mean(axis=1)
    master_perf_df['avg_ks2_pass_rate'] = master_perf_df[PASS_RATE_COL]

    print(f"  -> Loaded {len(master_perf_df)} valid performance records.")
    return master_perf_df[[URN_COL, 'year', 'avg_primary_scaled_score', 'avg_ks2_pass_rate']]


def process_area_data(area_gdf, schools_gdf_proj, performance_df, area_code_col):
    """
    Processes either LSOA or Ward data to find nearest schools and calculate scores.
    """
    print(f"\nStep 3: Finding {NEAREST_N_SCHOOLS} nearest primary schools for {len(area_gdf)} {area_code_col}s...")

    #Create spatial index for schools
    school_coords = np.array(list(schools_gdf_proj.geometry.apply(lambda p: (p.x, p.y))))
    kdtree = cKDTree(school_coords)

    #Get centroids for the areas (LSOAs or Wards)
    area_gdf['centroid'] = area_gdf.geometry.centroid
    area_coords = np.array(list(area_gdf.centroid.apply(lambda p: (p.x, p.y))))

    #Query the tree to find the indices of the N nearest schools for each area
    distances_m, indices = kdtree.query(area_coords, k=NEAREST_N_SCHOOLS)

    #Nearest Schools (for App UI)
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
            result_row[f'primary_school_{n + 1}_name'] = school_info[SCHNAME_COL]
            result_row[f'primary_school_{n + 1}_urn'] = school_urn

            if not perf_info.empty:
                result_row[f'primary_school_{n + 1}_pass_rate'] = perf_info['avg_ks2_pass_rate'].iloc[0]
                result_row[f'primary_school_{n + 1}_avg_scaled_score'] = perf_info['avg_primary_scaled_score'].iloc[0]
                result_row[f'primary_school_{n + 1}_data_year'] = perf_info['year'].iloc[0]
            else:
                result_row[f'primary_school_{n + 1}_pass_rate'] = np.nan
                result_row[f'primary_school_{n + 1}_avg_scaled_score'] = np.nan
                result_row[f'primary_school_{n + 1}_data_year'] = np.nan

        static_details_results.append(result_row)

    static_details_df = pd.DataFrame(static_details_results)

    for i in range(1, NEAREST_N_SCHOOLS + 1):
        static_details_df[f'primary_school_{i}_urn'] = static_details_df[f'primary_school_{i}_urn'].astype('string')
        static_details_df[f'primary_school_{i}_name'] = static_details_df[f'primary_school_{i}_name'].astype('string')

    #Annual Scores for Timeseries - Forward Fill for Covid Years Missing Data
    print(f"Step 5: Generating annual time-series scores for {area_code_col}...")
    area_urn_mapping = []
    for i, area_row in area_gdf.iterrows():
        for n in range(NEAREST_N_SCHOOLS):
            school_urn = schools_gdf_proj.iloc[indices[i][n]][URN_COL]
            area_urn_mapping.append({'area_code': area_row[area_code_col], URN_COL: school_urn})

    area_urn_map_df = pd.DataFrame(area_urn_mapping).drop_duplicates()
    annual_data = area_urn_map_df.merge(performance_df, on=URN_COL, how='left')
    annual_data = annual_data.sort_values(by=['area_code', URN_COL, 'year'])
    perf_cols = ['avg_primary_scaled_score', 'avg_ks2_pass_rate']
    annual_data[perf_cols] = annual_data.groupby(['area_code', URN_COL])[perf_cols].ffill()
    area_annual_scores = annual_data.groupby(['area_code', 'year'])[perf_cols].mean().reset_index()
    master_index = pd.MultiIndex.from_product(
        [area_gdf[area_code_col].unique(), YEARS_TO_PROCESS],
        names=['area_code', 'year']
    )
    final_scores_df = pd.DataFrame(index=master_index).reset_index()
    final_scores_df = final_scores_df.merge(area_annual_scores, on=['area_code', 'year'], how='left')
    print(f"  -> Forward-filling missing years for {area_code_col}s (e.g., COVID gaps)...")
    final_scores_df[perf_cols] = final_scores_df.groupby('area_code')[perf_cols].ffill()
    final_scores_df[perf_cols] = final_scores_df.groupby('area_code')[perf_cols].bfill()
    return static_details_df, final_scores_df


#Main
if __name__ == "__main__":
    print("Step 1: Loading and geocoding all primary schools...")
    schools_gdf_proj = load_school_locations()
    if schools_gdf_proj is None:
        sys.exit(1)

    print("\nStep 2: Loading all performance data...")
    ks2_performance_df = load_school_performance()
    if ks2_performance_df is None:
        sys.exit(1)

    historical_df = ks2_performance_df.rename(columns={'URN': 'urn'})
    historical_df.to_parquet(
        PROCESSED_DATA_DIR / "primary_school_historical_data.parquet",
        index=False
    )
    print(f"Success! Saved historical primary school data.")
    print("\n--- Processing LSOA Data ---")
    lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE).to_crs("EPSG:27700")
    lsoa_static_df, lsoa_annual_df = process_area_data(
        area_gdf=lsoa_gdf,
        schools_gdf_proj=schools_gdf_proj,
        performance_df=ks2_performance_df,
        area_code_col='area_code'
    )
    lsoa_static_df.to_parquet(STATIC_DETAILS_OUTPUT_LSOA, index=False)
    lsoa_annual_df.to_parquet(ANNUAL_SCORES_OUTPUT_LSOA, index=False)
    print(f"Success! Saved LSOA primary school files.")
    print("\n--- Processing Ward Data ---")
    ward_gdf = gpd.read_file(WARD_BOUNDARIES_FILE).to_crs("EPSG:27700")
    ward_static_df, ward_annual_df = process_area_data(
        area_gdf=ward_gdf,
        schools_gdf_proj=schools_gdf_proj,
        performance_df=ks2_performance_df,
        area_code_col='WD25CD'
    )
    ward_static_df.rename(columns={'area_code': 'ward_code'}, inplace=True)
    ward_annual_df.rename(columns={'area_code': 'ward_code'}, inplace=True)
    ward_static_df.to_parquet(STATIC_DETAILS_OUTPUT_WARD, index=False)
    ward_annual_df.to_parquet(ANNUAL_SCORES_OUTPUT_WARD, index=False)
    print(f"Success! Saved Ward primary school files.")
    print("\nPrimary school script finished.")