import pandas as pd
import geopandas as gpd
from pathlib import Path
import re
import sys

print("Starting ANNUAL population data processing (RAW EXTRACTION - NO IMPUTATION)...")

#Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "population"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
RAW_POP_DIR = RAW_DATA_DIR
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
OUTPUT_FILE = PROCESSED_DATA_DIR / "lsoa_annual_population.parquet"

#Data Processing
try:
    print("Step 1: Loading the list of required LSOAs for the app...")
    lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE)
    required_lsoa_codes = set(lsoa_gdf['area_code'].unique())
    print(f"Found {len(required_lsoa_codes)} unique LSOAs used in the app.")

    print(f"Step 2: Loading ALL raw population files from {RAW_POP_DIR}...")
    pop_files = list(RAW_POP_DIR.glob("*.csv"))

    if not pop_files:
        print(f"ERROR: No raw population CSV files found in {RAW_POP_DIR}. Exiting.")
        sys.exit(1)
    all_pop_data = []

    for f in pop_files:
        match = re.search(r'(\d{4})', f.stem)
        if not match:
            print(f"Warning: Could not extract year from filename {f.name}. Skipping.")
            continue

        year = int(match.group(1))
        print(f"  -> Processing {f.name} for year {year}...")

        # Add encoding='utf-8-sig' to handle the 'ï»¿' Byte Order Mark (BOM)
        pop_df = pd.read_csv(f, thousands=',', encoding='utf-8-sig')

        # Normalize all column names IN-PLACE first.
        pop_df.columns = [col.lower().strip() for col in pop_df.columns]
        if 'lsoa 2021 code' in pop_df.columns:
            lsoa_col = 'lsoa 2021 code'
        elif 'lsoa21cd' in pop_df.columns:
            lsoa_col = 'lsoa21cd'
        else:
            print(f"  -> ERROR: Could not find 'lsoa 2021 code' or 'lsoa21cd' in {f.name}. Skipping.")
            print(f"     Available columns: {pop_df.columns.tolist()}")
            continue

        if 'total' in pop_df.columns:
            total_col = 'total'
        elif 'all ages' in pop_df.columns:  # Another common name
            total_col = 'all ages'
        else:
            print(f"  -> ERROR: Could not find 'total' or 'all ages' column in {f.name}. Skipping.")
            print(f"     Available columns: {pop_df.columns.tolist()}")
            continue

        print(f"  -> Found columns: '{lsoa_col}' and '{total_col}'")

        # Select and rename the correct columns
        pop_df = pop_df[[lsoa_col, total_col]].copy()
        pop_df = pop_df.rename(columns={lsoa_col: 'area_code', total_col: 'population'})

        pop_df['year'] = year
        all_pop_data.append(pop_df)

    if not all_pop_data:
        print("ERROR: No population data was successfully processed. Exiting.")
        sys.exit(1)

    # Combine all found years into one dataframe
    print("Step 3: Combining all found years into one dataframe...")
    combined_pop_df = pd.concat(all_pop_data, ignore_index=True)
    print(f"Loaded {len(combined_pop_df):,} total population records from {len(all_pop_data)} file(s).")
    print("Step 4: Trimming population data to match the app's LSOAs...")
    trimmed_pop_df = combined_pop_df[combined_pop_df['area_code'].isin(required_lsoa_codes)].copy()
    print(f"Trimmed to {len(trimmed_pop_df):,} records for the South West.")

    if trimmed_pop_df.empty:
        print("WARNING: No matching LSOAs found. The output file will be empty.")

    # Save
    print("Step 5: Cleaning and saving the processed file...")
    trimmed_pop_df['population'] = pd.to_numeric(trimmed_pop_df['population'], errors='coerce').fillna(0).astype(int)
    trimmed_pop_df['area_code'] = trimmed_pop_df['area_code'].astype(str).str.strip()
    trimmed_pop_df['year'] = trimmed_pop_df['year'].astype(int)
    final_df = trimmed_pop_df[['area_code', 'year', 'population']]
    final_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Success! Processed RAW population data saved to {OUTPUT_FILE}")

except FileNotFoundError as e:
    print(f"ERROR: A required file was not found. Please check your file paths. Details: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
print("Script finished.")