import pandas as pd
from pathlib import Path

print("Starting IMD data processing")

# Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
IMD_FILE = RAW_DATA_DIR / "imd" / "imd_2019.csv"
LSOA_LOOKUP_FILE = RAW_DATA_DIR / "lookups" / "LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Best_Fit_Lookup_for_EW_(V2).csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "lsoa_imd.parquet"

# File Check
if not IMD_FILE.exists() or not LSOA_LOOKUP_FILE.exists():
    print("ERROR: IMD file or LSOA lookup file not found.")
    exit()

# Load Data
print("Loading IMD 2019 scores from CSV...")
try:
    df_imd = pd.read_csv(IMD_FILE)
except Exception as e:
    print(f"ERROR: Could not read IMD CSV file. Error: {e}")
    exit()

# Column Mapping
raw_column_map = {
    'Index of Multiple Deprivation (IMD) Score': 'IMD_Score',
    'Income Score (rate)': 'Income_Rate',
    'Employment Score (rate)': 'Employment_Rate',
    'Health Deprivation and Disability Score': 'Health_Score',
    'Income Deprivation Affecting Children Index (IDACI) Score (rate)': 'IDACI_Rate'
}
for col in raw_column_map.keys():
    if col not in df_imd.columns:
        print(f"ERROR: Required column '{col}' not found in CSV.")
        exit()

# Extract Raw VAlues
print("Extracting raw scores...")
cols_to_select = ['LSOA code (2011)'] + list(raw_column_map.keys())
df_imd_clean = df_imd[cols_to_select].copy()
# Rename
df_imd_clean.rename(columns=raw_column_map, inplace=True)
df_imd_clean.rename(columns={'LSOA code (2011)': 'LSOA11CD'}, inplace=True)

# Geometries and Save
print("Loading LSOA lookup...")
df_lookup = pd.read_csv(LSOA_LOOKUP_FILE)
df_lookup = df_lookup[['LSOA11CD', 'LSOA21CD']].drop_duplicates()
print("Merging data...")
df_merged = pd.merge(df_lookup, df_imd_clean, on='LSOA11CD', how='left')
df_final = df_merged.drop(columns=['LSOA11CD'])
df_final.rename(columns={'LSOA21CD': 'area_code'}, inplace=True)
df_final['area_code'] = df_final['area_code'].astype(str).str.strip()
initial_count = len(df_final)
df_final.dropna(subset=['IMD_Score'], inplace=True)
final_count = len(df_final)
if initial_count != final_count:
    print(f"Dropped {initial_count - final_count} rows due to geography mismatch.")
df_final.to_parquet(OUTPUT_FILE, index=False)
print(f"Success! Saved {len(df_final)} raw IMD records to {OUTPUT_FILE}")