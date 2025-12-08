import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys

print("Starting DUAL crime data aggregation (Annual and Monthly)...")

#Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
CRIME_DATA_FOLDER = RAW_DATA_DIR / "crime" / "Crime_2018-2025"
POPULATION_FILE = PROCESSED_DATA_DIR / "lsoa_annual_population.parquet"
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
OUTPUT_FILE_ANNUAL = PROCESSED_DATA_DIR / "lsoa_annual_crime.parquet"
OUTPUT_FILE_MONTHLY = PROCESSED_DATA_DIR / "lsoa_monthly_crime.parquet"
COMMUNITY_SAFETY_CRIMES = [
    'Anti-social behaviour', 'Burglary', 'Robbery',
    'Criminal damage and arson', 'Violence and sexual offences', 'Public order'
]

#Load Data
print(f"Loading all crime CSVs from subfolders in {CRIME_DATA_FOLDER}...")
crime_files = list(CRIME_DATA_FOLDER.rglob("*.csv"))
if not crime_files:
    print(f"ERROR: No crime CSV files found. Exiting.")
    sys.exit(1)

df_list = []
for f in crime_files:
    try:
        df = pd.read_csv(
            f,
            usecols=['Month', 'LSOA code', 'Crime type'],
            parse_dates=['Month']
        )
        df_list.append(df)
    except Exception as e:
        print(f"Warning: Could not process file {f.name}. Error: {e}")

if not df_list:
    print(f"ERROR: No crime data was successfully loaded. Exiting.")
    sys.exit(1)

crime_df = pd.concat(df_list, ignore_index=True)
crime_df.dropna(subset=['LSOA code', 'Crime type', 'Month'], inplace=True)
crime_df = crime_df.rename(columns={'LSOA code': 'area_code'})

#Filter to relevant crimes
crime_df = crime_df[crime_df['Crime type'].isin(COMMUNITY_SAFETY_CRIMES)]
print(f"Filtered to {len(crime_df):,} relevant incidents.")

#Create year and month columns
crime_df['year'] = crime_df['Month'].dt.year
crime_df['month'] = crime_df['Month'].dt.month
crime_df['period'] = crime_df['Month'].dt.to_period('M')

#Create monthly totals
print(f"Aggregating monthly data...")
monthly_agg = crime_df.groupby(['area_code', 'period']).size().reset_index(name='monthly_crime_count')

#By Crime Type
monthly_by_category = crime_df.groupby(['area_code', 'period', 'Crime type']).size().unstack(fill_value=0)
monthly_final_df = monthly_agg.merge(monthly_by_category, on=['area_code', 'period'], how='left')
monthly_final_df['period'] = monthly_final_df['period'].astype(str)

#Save Monthly
monthly_final_df.to_parquet(OUTPUT_FILE_MONTHLY, index=False)
print(f"Success! Saved MONTHLY data for {len(monthly_final_df)} LSOA/month combinations to {OUTPUT_FILE_MONTHLY}")


#Create Annual Totals
print(f"Aggregating annual data...")
annual_crimes = crime_df.groupby(['area_code', 'year']).size().reset_index(name='raw_crime_count')
months_per_year = crime_df.groupby(['area_code', 'year'])['Month'].nunique().reset_index(name='months_of_data')
annual_df = annual_crimes.merge(months_per_year, on=['area_code', 'year'], how='outer')
annual_df['annualized_crime_count'] = (annual_df['raw_crime_count'] / annual_df['months_of_data']) * 12
annual_df['is_full_year'] = annual_df['months_of_data'] == 12

#Merge Population & Boundaries
print("Merging annual data with LSOA boundaries and annual population...")
lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE)[['area_code']]

if not POPULATION_FILE.exists():
    print(f"ERROR: Annual population file not found at {POPULATION_FILE}.")
    sys.exit(1)
population_df = pd.read_parquet(POPULATION_FILE)

#Master DF
all_years = annual_df['year'].unique()
master_index = pd.MultiIndex.from_product([lsoa_gdf['area_code'].unique(), all_years], names=['area_code', 'year'])
master_df = pd.DataFrame(index=master_index).reset_index()
master_df = master_df.merge(population_df, on=['area_code', 'year'], how='left')
master_df = master_df.merge(annual_df, on=['area_code', 'year'], how='left')

#Fill NaNs for LSOAs twith no crime
fill_cols = ['raw_crime_count', 'months_of_data', 'annualized_crime_count']
master_df.fillna({col: 0 for col in fill_cols}, inplace=True)
master_df['is_full_year'] = master_df['months_of_data'].apply(lambda x: x == 12)

#Save Annual
annual_output_df = master_df.drop(columns=['geometry'], errors='ignore')
annual_output_df.to_parquet(OUTPUT_FILE_ANNUAL, index=False)
print(f"Success! Saved ANNUALLY aggregated data for {len(annual_output_df)} LSOA/year combinations to {OUTPUT_FILE_ANNUAL}")
print("Script finished.")