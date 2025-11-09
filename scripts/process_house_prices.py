import pandas as pd
from pathlib import Path
import numpy as np

print("Starting house price time series data processing (with monthly granularity)...")

#Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "housing"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
HOUSE_PRICE_FILE = RAW_DATA_DIR / "median_house_prices.csv"
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
AREA_CODES_FILE = PROCESSED_DATA_DIR / "area_codes.csv"
TIMESERIES_OUTPUT_FILE = PROCESSED_DATA_DIR / "lsoa_house_prices_timeseries.parquet"
LATEST_PRICE_OUTPUT_FILE = PROCESSED_DATA_DIR / "lsoa_latest_house_prices_imputed.parquet"
WARD_TIMESERIES_OUTPUT_FILE = PROCESSED_DATA_DIR / "ward_house_prices_timeseries.parquet"
SW_TIMESERIES_OUTPUT_FILE = PROCESSED_DATA_DIR / "sw_house_prices_timeseries.parquet"


#Load Data
if not HOUSE_PRICE_FILE.exists():
    print(f"ERROR: House price file not found at {HOUSE_PRICE_FILE}")
    exit()

print(f"Loading data from {HOUSE_PRICE_FILE.name}...")
df = pd.read_csv(HOUSE_PRICE_FILE, low_memory=False)

lsoa_code_col_name = 'LSOA code'
year_columns = [col for col in df.columns if isinstance(col, str) and col.startswith('Year ending')]
if not year_columns:
    print("ERROR: No 'Year ending...' columns found. Check CSV headers.")
    exit()

#LSOA Codes
cols_to_keep = [lsoa_code_col_name] + year_columns
df_selected = df[cols_to_keep].copy()
df_selected.rename(columns={lsoa_code_col_name: 'area_code'}, inplace=True)
df_selected['area_code'] = df_selected['area_code'].astype(str).str.strip()
df_selected.dropna(subset=['area_code'], inplace=True)

#Melt to long
print("Melting data to long format...")
df_long = pd.melt(
    df_selected,
    id_vars=['area_code'],
    value_vars=year_columns,
    var_name='period_text',
    value_name='median_house_price'
)

#Dates
print("Extracting full date from period text...")
date_parts = df_long['period_text'].str.extract(r'Year ending (\w{3}) (\d{4})')
df_long['date'] = pd.to_datetime(date_parts[1] + '-' + date_parts[0] + '-01', errors='coerce') + pd.offsets.MonthEnd(0)


#Prices
print("Cleaning price data...")
df_long['median_house_price'] = df_long['median_house_price'].astype(str).str.replace(',', '', regex=False)
df_long['median_house_price'] = pd.to_numeric(df_long['median_house_price'], errors='coerce')
df_long.dropna(subset=['date', 'median_house_price'], inplace=True)
df_long['median_house_price'] = df_long['median_house_price'].astype(int)
print(f"Processed {len(df_long):,} LSOA-period records from raw file.")

#LSOAs
if not LSOA_BOUNDARIES_FILE.exists() or not AREA_CODES_FILE.exists():
    print(f"ERROR: Boundaries or Area Codes file not found.")
    exit()
print("Loading SW LSOA boundaries and area codes...")
lsoa_geo_df = pd.read_parquet(LSOA_BOUNDARIES_FILE, columns=['area_code'])
sw_lsoa_codes = lsoa_geo_df['area_code'].astype(str).str.strip().unique()
area_codes_df = pd.read_csv(AREA_CODES_FILE)
area_codes_df = area_codes_df[["LSOA21CD", "WD25CD", "LAD25CD"]].copy()
area_codes_df.rename(columns={'LSOA21CD': 'area_code'}, inplace=True)
area_codes_df['area_code'] = area_codes_df['area_code'].astype(str).str.strip()
base_sw_lsoas_df = pd.DataFrame({'area_code': sw_lsoa_codes})
base_sw_lsoas_df = base_sw_lsoas_df.merge(area_codes_df, on='area_code', how='left')
print(f"Loaded {len(base_sw_lsoas_df):,} total LSOAs for South West.")


#Save LSOA Timeseries
df_sw_timeseries = df_long[df_long['area_code'].isin(sw_lsoa_codes)].copy()
df_final_timeseries = df_sw_timeseries[['area_code', 'date', 'median_house_price']]
df_final_timeseries.to_parquet(TIMESERIES_OUTPUT_FILE, index=False)
print(f"Success! Saved LSOA time series data to {TIMESERIES_OUTPUT_FILE}")

#Save latest price and impute if needed
if df_sw_timeseries.empty:
    print("Warning: No house price data for any SW LSOAs. Cannot create latest price file.")
    exit()

print("Starting imputation for latest period's prices...")
latest_date = df_sw_timeseries['date'].max()
print(f"Latest house price period found: {latest_date.strftime('%Y-%m')}")
latest_prices_df = df_sw_timeseries[df_sw_timeseries['date'] == latest_date][['area_code', 'median_house_price']]

#Imputation logic
df_imputed = base_sw_lsoas_df.merge(latest_prices_df, on='area_code', how='left')
df_imputed.rename(columns={'median_house_price': 'latest_median_house_price'}, inplace=True)
df_imputed['latest_median_house_price'] = df_imputed['latest_median_house_price'].replace(0, np.nan)
print(f"Imputing {df_imputed['latest_median_house_price'].isna().sum():,} missing latest house prices...")
if 'WD25CD' in df_imputed.columns:
    ward_median_prices = df_imputed.groupby('WD25CD')['latest_median_house_price'].transform('median')
    df_imputed['latest_median_house_price'] = df_imputed['latest_median_house_price'].fillna(ward_median_prices)
if df_imputed['latest_median_house_price'].isna().any() and 'LAD25CD' in df_imputed.columns:
    lad_median_prices = df_imputed.groupby('LAD25CD')['latest_median_house_price'].transform('median')
    df_imputed['latest_median_house_price'] = df_imputed['latest_median_house_price'].fillna(lad_median_prices)
if df_imputed['latest_median_house_price'].isna().any():
    sw_median_price = df_imputed['latest_median_house_price'].median()
    if pd.notna(sw_median_price):
        df_imputed['latest_median_house_price'] = df_imputed['latest_median_house_price'].fillna(sw_median_price)
df_imputed['latest_median_house_price'] = df_imputed['latest_median_house_price'].fillna(0).astype(int)
print("Imputation complete.")
df_final_latest = df_imputed[['area_code', 'latest_median_house_price']]
df_final_latest.to_parquet(LATEST_PRICE_OUTPUT_FILE, index=False)
print(f"Success! Saved LSOA latest imputed prices to {LATEST_PRICE_OUTPUT_FILE}")

#Ward Timeseries
print("Aggregating time series for Ward and SW levels...")
df_sw_timeseries_with_wards = df_sw_timeseries.merge(
    base_sw_lsoas_df[['area_code', 'WD25CD']],
    on='area_code',
    how='left'
)

#Calculate Ward median
ward_history = df_sw_timeseries_with_wards.groupby(['WD25CD', 'date'])['median_house_price'].median().reset_index()
ward_history['median_house_price'] = ward_history['median_house_price'].astype(int)
ward_history.rename(columns={'WD25CD': 'area_code'}, inplace=True)
ward_history.to_parquet(WARD_TIMESERIES_OUTPUT_FILE, index=False)
print(f"Success! Saved Ward time series data to {WARD_TIMESERIES_OUTPUT_FILE}")

#Whole SW Median
sw_history = df_sw_timeseries['median_house_price'].median()
sw_history = df_sw_timeseries.groupby('date')['median_house_price'].median().reset_index()
sw_history['median_house_price'] = sw_history['median_house_price'].astype(int)
sw_history.to_parquet(SW_TIMESERIES_OUTPUT_FILE, index=False)
print(f"Success! Saved SW time series data to {SW_TIMESERIES_OUTPUT_FILE}")
print("Script finished.")