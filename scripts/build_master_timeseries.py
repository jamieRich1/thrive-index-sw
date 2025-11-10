import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys

print("Starting master annual indicator table build (IMPUTATION ENGINE)...")

# --- 1. DEFINE FILE PATHS ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

# Base geometry
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"

# Annual (Time-Series) Data Files (Now "sparse" with gaps)
ANNUAL_POP_FILE = PROCESSED_DATA_DIR / "lsoa_annual_population.parquet"
ANNUAL_CRIME_FILE = PROCESSED_DATA_DIR / "lsoa_annual_crime.parquet"
ANNUAL_AIR_QUALITY_FILE = PROCESSED_DATA_DIR / "lsoa_annual_air_quality.parquet"
ANNUAL_PRIMARY_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_primary_scores.parquet"
ANNUAL_SECONDARY_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_secondary_scores.parquet"
ANNUAL_HEALTHCARE_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_healthcare_scores.parquet"
ANNUAL_CHILDCARE_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_childcare_scores.parquet"
# Note: House price time-series is loaded for context, not merged into this file

# Static (Single-Snapshot) Data Files
STATIC_GREENSPACE_FILE = PROCESSED_DATA_DIR / "lsoa_greenspace.parquet"
STATIC_IMD_FILE = PROCESSED_DATA_DIR / "lsoa_imd.parquet"
STATIC_LATEST_HOUSE_PRICE_FILE = PROCESSED_DATA_DIR / "lsoa_latest_house_prices_imputed.parquet"  # Contextual
STATIC_SECONDARY_EDU_FILE = PROCESSED_DATA_DIR / "lsoa_secondary_education_details.parquet"
STATIC_PRIMARY_EDU_FILE = PROCESSED_DATA_DIR / "lsoa_primary_education_details.parquet"
STATIC_HEALTHCARE_FILE = PROCESSED_DATA_DIR / "lsoa_healthcare_details.parquet"
STATIC_CHILDCARE_FILE = PROCESSED_DATA_DIR / "lsoa_childcare_details.parquet"

# Final Output File
OUTPUT_FILE = PROCESSED_DATA_DIR / "lsoa_annual_indicators.parquet"

# Define the full time range for the index
YEARS = list(range(2018, 2026))


# --- 2. HELPER FUNCTION ---
def load_file(file_path, name):
    """Helper to load a parquet file and exit if it's missing."""
    if not file_path.exists():
        print(f"ERROR: Missing required file: {file_path.name}")
        print(f"Please run the corresponding 'process_*.py' script first.")
        sys.exit(1)
    print(f"  -> Loading {name}...")
    return pd.read_parquet(file_path)


# --- 3. MAIN BUILD PROCESS ---
def main():
    print("Step 1: Loading LSOA boundaries...")
    lsoa_gdf = load_file(LSOA_BOUNDARIES_FILE, "LSOA boundaries")
    lsoa_codes = lsoa_gdf['area_code'].unique()
    print(f"Loaded {len(lsoa_codes)} LSOAs.")

    print(f"Step 2: Creating master LSOA-Year index for {YEARS[0]}-{YEARS[-1]}...")
    master_index = pd.MultiIndex.from_product([lsoa_codes, YEARS], names=['area_code', 'year'])
    master_df = pd.DataFrame(index=master_index).reset_index()

    # --- 3.1. MERGE ANNUAL (SPARSE) DATA ---
    print("Step 3: Merging sparse annual data (will create NaNs)...")

    annual_files = {
        ANNUAL_POP_FILE: "population",
        ANNUAL_CRIME_FILE: "crime",
        ANNUAL_AIR_QUALITY_FILE: "air quality",
        ANNUAL_PRIMARY_SCORES_FILE: "primary school scores",
        ANNUAL_SECONDARY_SCORES_FILE: "secondary school scores",
        ANNUAL_HEALTHCARE_SCORES_FILE: "healthcare scores",
        ANNUAL_CHILDCARE_SCORES_FILE: "childcare scores",
    }

    for file_path, name in annual_files.items():
        df_annual = load_file(file_path, name)
        # Rename crime count for clarity
        if file_path == ANNUAL_CRIME_FILE:
            df_annual = df_annual.rename(columns={'annualized_crime_count': 'crime_count'})
            df_annual = df_annual.drop(columns=['raw_crime_count', 'months_of_data', 'is_full_year'], errors='ignore')

        master_df = master_df.merge(df_annual, on=['area_code', 'year'], how='left')

    # --- 3.2. MERGE STATIC (SINGLE-SNAPSHOT) DATA ---
    print("Step 4: Merging static data (will be propagated to all years)...")

    static_files = {
        STATIC_GREENSPACE_FILE: "greenspace",
        STATIC_IMD_FILE: "IMD",
        STATIC_LATEST_HOUSE_PRICE_FILE: "latest house prices",  # Contextual
        STATIC_SECONDARY_EDU_FILE: "secondary school details",
        STATIC_PRIMARY_EDU_FILE: "primary school details",
        STATIC_HEALTHCARE_FILE: "healthcare details",
        STATIC_CHILDCARE_FILE: "childcare details",
    }

    # Store column names to be propagated
    static_cols = []
    for file_path, name in static_files.items():
        df_static = load_file(file_path, name)
        new_cols = [col for col in df_static.columns if col != 'area_code']
        static_cols.extend(new_cols)
        master_df = master_df.merge(df_static, on='area_code', how='left')

    # --- 3.3. START IMPUTATION & GAP-FILLING ---
    print("Step 5: Running imputation strategies...")

    # Sort by area_code and year, which is ESSENTIAL for ffill/bfill to work correctly
    master_df = master_df.sort_values(by=['area_code', 'year'])

    # --- Imputation Strategy 1: Static Data Propagation ---
    # Propagate all static data (IMD, Greenspace, static school/GP names)
    # across all years for each LSOA.
    print("  -> Strategy 1: Propagating static data (IMD, Greenspace, etc.) across all years...")
    # Ensure we only propagate columns that actually exist
    static_cols_to_propagate = [col for col in static_cols if col in master_df.columns]
    master_df[static_cols_to_propagate] = master_df.groupby('area_code')[static_cols_to_propagate].ffill().bfill()

    # --- Imputation Strategy 2: Annual Contextual Data (Population) ---
    # Population is not a scoring indicator, but is a denominator for crime.
    # A simple LOCF (Last Observation Carried Forward) is academically acceptable.
    # This will now use your 2024 data to fill 2025.
    print("  -> Strategy 2: Forward-filling Population data (LOCF)...")
    master_df['population'] = master_df.groupby('area_code')['population'].ffill()

    # --- Imputation Strategy 3: Annual SCORING Data (The Core Logic) ---
    # This block fills the gaps for the key scoring indicators.
    # The default method is ffill().bfill() - Last Observation Carried Forward, then Backward.
    # This fills the COVID gap (2020-2021) for schools with 2019/2022 data.
    # It back-fills childcare (2018-2020) with 2021 data.
    # It forward-fills Air Quality (2025) with 2024 data.
    print("  -> Strategy 3: Filling gaps for scoring indicators (ffill/bfill)...")

    scoring_cols = [
        'crime_count',  # Fill NaNs with 0 instead of ffill
        'no2_mean_concentration', 'pm25_mean_concentration', 'air_quality_score',
        'avg_primary_scaled_score', 'avg_ks2_pass_rate',
        'avg_progress_8', 'avg_attainment_8',
        'avg_gp_satisfaction',
        'avg_childcare_quality_score', 'avg_distance_to_childcare_km', 'total_childcare_places_nearby'
    ]
    # Ensure we only process columns that were successfully loaded
    scoring_cols_to_fill = [col for col in scoring_cols if col in master_df.columns and col != 'crime_count']

    # === METHOD 1: Forward-fill then Back-fill (Default) ===
    master_df[scoring_cols_to_fill] = master_df.groupby('area_code')[scoring_cols_to_fill].ffill().bfill()

    # === METHOD 2: Linear Interpolation (Alternative for testing) ===
    # To use this, comment out the `ffill().bfill()` line above and uncomment the line below.
    # This will fill gaps by drawing a straight line between the two nearest points.
    # master_df[scoring_cols_to_fill] = master_df.groupby('area_code')[scoring_cols_to_fill].interpolate(method='linear', limit_direction='both', axis=0)

    # Specific fill for crime: missing crime data means 0 crimes, not an imputed value.
    if 'crime_count' in master_df.columns:
        master_df['crime_count'] = master_df['crime_count'].fillna(0)

    # --- 3.4. FINAL CLEANUP & SAVE ---
    print("Step 6: Final cleanup and save...")

    # Identify all numeric columns for final fill
    numeric_cols = master_df.select_dtypes(include=np.number).columns.tolist()
    if 'year' in numeric_cols: numeric_cols.remove('year')  # Don't fill year
    if 'area_code' in numeric_cols: numeric_cols.remove('area_code')  # This shouldn't be numeric, but good to check

    master_df.fillna({col: 0 for col in numeric_cols}, inplace=True)
    master_df.fillna('N/A', inplace=True)

    # Ensure key ID/name columns are strings
    for col in ['area_code', 'WD25CD', 'WD25NM', 'LAD25CD', 'LAD25NM']:
        if col in master_df.columns:
            master_df[col] = master_df[col].astype(str).fillna('N/A')

    # Save the master file
    master_df.to_parquet(OUTPUT_FILE, index=False)

    print("-" * 50)
    print(f"✅ Success! Master annual indicator file built with {len(master_df)} LSOA/year records.")
    print(f"All imputation is now centralized in this script.")
    print(f"File saved to: {OUTPUT_FILE}")
    print("-" * 50)


if __name__ == "__main__":
    main()