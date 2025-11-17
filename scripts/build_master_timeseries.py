import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys
import numpy as np

print("Starting master annual indicator table build (IMPUTATION ENGINE)...")

#Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

#Base Geographies
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"

# Sparse Annual Data (To be loaded and imputed)
ANNUAL_POP_FILE = PROCESSED_DATA_DIR / "lsoa_annual_population.parquet"
ANNUAL_CRIME_FILE = PROCESSED_DATA_DIR / "lsoa_annual_crime.parquet"
ANNUAL_AIR_QUALITY_FILE = PROCESSED_DATA_DIR / "lsoa_annual_air_quality.parquet"
ANNUAL_PRIMARY_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_primary_scores.parquet"
ANNUAL_SECONDARY_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_secondary_scores.parquet"
ANNUAL_HEALTHCARE_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_healthcare_scores.parquet"
ANNUAL_CHILDCARE_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_childcare_scores.parquet"

#Static Data
STATIC_GREENSPACE_FILE = PROCESSED_DATA_DIR / "lsoa_greenspace.parquet"
STATIC_IMD_FILE = PROCESSED_DATA_DIR / "lsoa_imd.parquet"
STATIC_LATEST_HOUSE_PRICE_FILE = PROCESSED_DATA_DIR / "lsoa_latest_house_prices_imputed.parquet"

#Static Details
STATIC_SECONDARY_EDU_FILE = PROCESSED_DATA_DIR / "lsoa_secondary_education_details.parquet"
STATIC_PRIMARY_EDU_FILE = PROCESSED_DATA_DIR / "lsoa_primary_education_details.parquet"
STATIC_HEALTHCARE_FILE = PROCESSED_DATA_DIR / "lsoa_healthcare_details.parquet"
STATIC_CHILDCARE_FILE = PROCESSED_DATA_DIR / "lsoa_childcare_details.parquet"

#Final Output
OUTPUT_FILE = PROCESSED_DATA_DIR / "lsoa_annual_indicators.parquet"

#Define Years for the complete 2018-2025 grid
YEARS = list(range(2018, 2026))


#Helpers
def load_and_merge_file(master_df, file_path, on_cols, file_desc, cols_to_drop=None):
    """
    Helper function to load a parquet file, optionally drop columns, and merge it.
    """
    if file_path.exists():
        print(f"  -> Loading {file_desc}...")
        df_to_merge = pd.read_parquet(file_path)
        if cols_to_drop:
            cols_to_drop_existing = [col for col in cols_to_drop if col in df_to_merge.columns]
            if cols_to_drop_existing:
                df_to_merge = df_to_merge.drop(columns=cols_to_drop_existing)
                print(f"     -> Dropped conflicting columns: {cols_to_drop_existing}")

        master_df = master_df.merge(df_to_merge, on=on_cols, how='left')
    else:
        print(f"ERROR: {file_path.name} not found. Stopping build.")
        sys.exit(1)
    return master_df


#Main Processing
def main():
    try:
        #LSOAs
        print("Step 1: Loading LSOA boundaries...")
        if not LSOA_BOUNDARIES_FILE.exists():
            print(f"ERROR: Base LSOA boundary file not found: {LSOA_BOUNDARIES_FILE}")
            print("Please run 'scripts/build_boundaries_sw.py' first.")
            sys.exit(1)
        lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE)
        lsoa_codes = lsoa_gdf['area_code'].unique()
        print(f"Loaded {len(lsoa_codes)} LSOAs.")

        #Master LSOA Index
        print("Step 2: Creating master LSOA-Year index for 2018-2025...")
        master_index = pd.MultiIndex.from_product([lsoa_codes, YEARS], names=['area_code', 'year'])
        master_df = pd.DataFrame(index=master_index).reset_index()
        master_df = master_df.sort_values(by=['area_code', 'year'])

        #Merging Sparse Annual Data
        print("Step 3: Merging sparse annual data (will create NaNs)...")
        master_df = load_and_merge_file(master_df, ANNUAL_POP_FILE, ['area_code', 'year'], "population")
        master_df = load_and_merge_file(master_df, ANNUAL_CRIME_FILE, ['area_code', 'year'], "crime",
                                        cols_to_drop=['population', 'raw_crime_count', 'months_of_data',
                                                      'is_full_year'])

        master_df = load_and_merge_file(master_df, ANNUAL_AIR_QUALITY_FILE, ['area_code', 'year'], "air quality")
        master_df = load_and_merge_file(master_df, ANNUAL_PRIMARY_SCORES_FILE, ['area_code', 'year'],
                                        "primary school scores")
        master_df = load_and_merge_file(master_df, ANNUAL_SECONDARY_SCORES_FILE, ['area_code', 'year'],
                                        "secondary school scores")
        master_df = load_and_merge_file(master_df, ANNUAL_HEALTHCARE_SCORES_FILE, ['area_code', 'year'],
                                        "healthcare scores")
        master_df = load_and_merge_file(master_df, ANNUAL_CHILDCARE_SCORES_FILE, ['area_code', 'year'],
                                        "childcare scores")

        #Merge Static Data
        print("Step 4: Merging static data (will be propagated across all years)...")
        master_df = load_and_merge_file(master_df, STATIC_GREENSPACE_FILE, ['area_code'], "greenspace")
        master_df = load_and_merge_file(master_df, STATIC_IMD_FILE, ['area_code'], "IMD")
        master_df = load_and_merge_file(master_df, STATIC_LATEST_HOUSE_PRICE_FILE, ['area_code'], "latest house prices")
        master_df = load_and_merge_file(master_df, STATIC_SECONDARY_EDU_FILE, ['area_code'], "secondary school details")
        master_df = load_and_merge_file(master_df, STATIC_PRIMARY_EDU_FILE, ['area_code'], "primary school details")
        master_df = load_and_merge_file(master_df, STATIC_HEALTHCARE_FILE, ['area_code'], "healthcare details")
        master_df = load_and_merge_file(master_df, STATIC_CHILDCARE_FILE, ['area_code'], "childcare details")

        #Imputation
        print("Step 5: Running imputation strategies...")

        #Strategy 1: Propagate static data (IMD, Greenspace, etc.) across all years
        # These indicators are assumed to be constant for the study period.
        print("  -> Strategy 1: Propagating static data (IMD, Greenspace, etc.) across all years...")
        static_cols = [
            'greenspace_percentage', 'IMD_Decile', 'Income_Decile', 'Employment_Decile', 'Health_Decile',
            'latest_median_house_price', 'avg_distance_to_gp_km'
        ]
        #Add static details columns (school names, URNs, etc.)
        static_cols += [col for col in master_df.columns if '_name' in col or '_urn' in col or '_org_code' in col]
        static_cols += [col for col in master_df.columns if '_nftype' in col or '_quality_rating' in col]
        static_cols += [col for col in master_df.columns if '_places' in col or '_distance_km' in col]
        static_cols += [col for col in master_df.columns if '_data_year' in col or '_satisfaction' in col]
        static_cols += [col for col in master_df.columns if '_pass_rate' in col or '_avg_scaled_score' in col]
        static_cols += [col for col in master_df.columns if '_progress_8' in col or '_attainment_8' in col]

        #Remove duplicates and ensure they exist in the dataframe
        static_cols = list(set(col for col in static_cols if col in master_df.columns))
        master_df[static_cols] = master_df.groupby('area_code')[static_cols].ffill().bfill()

        #Strategy 2: Forward-fill Population data (LOCF)
        # Assumes population from the last known year (2024) is the best estimate for 2025.
        print("  -> Strategy 2: Forward-filling Population data (LOCF)...")
        master_df['population'] = master_df.groupby('area_code')['population'].ffill()

        #Strategy 3: Forward/Backward-fill Time-Series Data (LOCF/NOCB)
        #Fills gaps (like 2020-21 for schools) using the nearest available data point.
        print("  -> Strategy 3: Filling gaps in time-series (Education, Health, Childcare, Air Quality)...")
        timeseries_cols = [
            'annualized_crime_count',
            'no2_mean_concentration', 'pm25_mean_concentration', 'air_quality_score',
            'avg_primary_scaled_score', 'avg_ks2_pass_rate',
            'avg_progress_8', 'avg_attainment_8',
            'avg_gp_satisfaction',
            'avg_childcare_quality_score', 'avg_distance_to_childcare_km', 'total_childcare_places_nearby'
        ]
        #Rename crime column if it exists
        if 'annualized_crime_count' in master_df.columns:
            master_df = master_df.rename(columns={'annualized_crime_count': 'crime_count'})
            timeseries_cols[0] = 'crime_count'

        #Ensure we only try to fill columns that were successfully loaded
        timeseries_cols_exist = [col for col in timeseries_cols if col in master_df.columns]

        #ffill() fills forward (e.g., 2019 data fills 2020, 2021)
        #bfill() fills backward (e.g., 2021 data fills 2018, 2019, 2020)
        master_df[timeseries_cols_exist] = master_df.groupby('area_code')[timeseries_cols_exist].ffill().bfill()

        #Clean and Save
        print("Step 6: Final cleanup and save...")

        #Fill any remaining NaNs (e.g., LSOAs with NO data at all) with 0
        numeric_cols = master_df.select_dtypes(include=np.number).columns.drop(['year'])
        master_df[numeric_cols] = master_df[numeric_cols].fillna(0)

        #Fill non-numeric columns with 'N/A'
        non_numeric_cols = master_df.select_dtypes(exclude=np.number).columns.drop(['area_code'])
        master_df[non_numeric_cols] = master_df[non_numeric_cols].fillna('N/A')

        #Ensure correct data types for integers
        int_cols = [
            'population', 'IMD_Decile', 'Income_Decile', 'Employment_Decile', 'Health_Decile',
            'latest_median_house_price'
        ]
        int_cols += [col for col in master_df.columns if '_data_year' in col or '_places' in col]

        for col in int_cols:
            if col in master_df.columns:
                master_df[col] = pd.to_numeric(master_df[col], errors='coerce').fillna(0).astype(int)

        master_df.to_parquet(OUTPUT_FILE, index=False)

        print(f"Success! Master annual indicator file built with {len(master_df)} LSOA/year records.")
        print(f"File saved to {OUTPUT_FILE}")

    except FileNotFoundError as e:
        print(f"ERROR: A required file was not found. {e.filename}")
        print("Please check your file paths and run rebuild_data.sh")
        sys.exit(1)
    except KeyError as e:
        print(f"ERROR: A required column was not found. {e}")
        print("This likely means a 'process_*.py' script failed to create the correct columns.")
        print("Run 'scripts/debug_check_processed_files.py' to investigate.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()