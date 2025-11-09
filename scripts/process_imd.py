# process_imd.py
import pandas as pd
from pathlib import Path

print("Starting IMD data processing...")

#Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
IMD_FILE = RAW_DATA_DIR / "imd" / "imd_2019.csv"
LSOA_LOOKUP_FILE = RAW_DATA_DIR / "lookups" / "LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Best_Fit_Lookup_for_EW_(V2).csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "lsoa_imd.parquet"

#Data Load
if not IMD_FILE.exists() or not LSOA_LOOKUP_FILE.exists():
    print("ERROR: IMD file or LSOA lookup file not found.")
    print(f"Check for IMD at: {IMD_FILE}")
    print(f"Check for Lookup at: {LSOA_LOOKUP_FILE}")
    exit()

print("Loading IMD 2019 scores from CSV...")
try:
    df_imd = pd.read_csv(IMD_FILE)
except Exception as e:
    print(f"ERROR: Could not read IMD CSV file. Error: {e}")
    exit()

#Calc deciles
print("Calculating deprivation deciles from scores...")
score_columns = {
    'Index of Multiple Deprivation (IMD) Score': 'IMD_Decile',
    'Income Score (rate)': 'Income_Decile',
    'Employment Score (rate)': 'Employment_Decile',
    'Health Deprivation and Disability Score': 'Health_Decile'
}

for score_col in score_columns.keys():
    if score_col not in df_imd.columns:
        print(f"ERROR: Required column '{score_col}' not found in {IMD_FILE.name}")
        exit()

for score_col, decile_col in score_columns.items():
    # The rank method='first' handles duplicate scores gracefully
    df_imd[decile_col] = pd.qcut(df_imd[score_col].rank(method='first'), 10, labels=False) + 1

#Merge Data
df_imd_final = df_imd[['LSOA code (2011)'] + list(score_columns.values())].copy()
df_imd_final.rename(columns={'LSOA code (2011)': 'LSOA11CD'}, inplace=True)
print("Loading LSOA 2011 to 2021 lookup...")
df_lookup = pd.read_csv(LSOA_LOOKUP_FILE)
df_lookup = df_lookup[['LSOA11CD', 'LSOA21CD']].drop_duplicates()
print("Merging IMD data onto 2021 LSOA codes...")
df_merged = pd.merge(df_lookup, df_imd_final, on='LSOA11CD', how='left')
df_final = df_merged.drop(columns=['LSOA11CD'])
df_final.rename(columns={'LSOA21CD': 'area_code'}, inplace=True)
df_final['area_code'] = df_final['area_code'].astype(str).str.strip()
df_final.dropna(subset=['IMD_Decile'], inplace=True)
df_final[list(score_columns.values())] = df_final[list(score_columns.values())].astype(int)

#Save
df_final.to_parquet(OUTPUT_FILE, index=False)
print(f"Success! Saved {len(df_final)} IMD records for 2021 LSOAs to {OUTPUT_FILE}")
print("IMD data processing finished.")