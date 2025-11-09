import pandas as pd
import geopandas as gpd
from pathlib import Path
from scipy.spatial import cKDTree
import numpy as np
import re
from datetime import datetime

print("Starting childcare provider data processing (Time-Series v3 - Corrected)...")

#Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw" / "childcare"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
CHILDCARE_DATA_PATTERN = "Management_information_-_childcare_providers*.csv"
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
POSTCODE_LOOKUP_FILE = PROCESSED_DATA_DIR / "sw_postcodes.parquet"
ANNUAL_SCORES_OUTPUT = PROCESSED_DATA_DIR / "lsoa_annual_childcare_scores.parquet"
STATIC_DETAILS_OUTPUT = PROCESSED_DATA_DIR / "lsoa_childcare_details.parquet"
NEAREST_N_PROVIDERS = 3
YEARS_TO_PROCESS = list(range(2018, 2026))


#Helpers
def extract_date_from_filename(filename):
    """
    Extracts the date and year from Ofsted filenames.
    """
    match = re.search(r'as_at_(\d{1,2})_(\w+)_(\d{4})', filename)
    if not match:
        return None, None

    day = int(match.group(1))
    month_str = match.group(2)
    year = int(match.group(3))

    try:
        month = datetime.strptime(month_str, "%B").month
    except ValueError:
        return None, None

    return datetime(year, month, day), year


#Load Data
print("Loading and preparing all Ofsted childcare provider files...")
all_files = list(RAW_DATA_DIR.glob(CHILDCARE_DATA_PATTERN))
if not all_files:
    print(f"ERROR: No Ofsted childcare files found matching '{CHILDCARE_DATA_PATTERN}' in {RAW_DATA_DIR}")
    exit()

cols_to_use = [
    'Provider URN',
    'Provider Name',
    'Provider Postcode',
    'Most Recent Full: Overall Effectiveness',
    'Places'
]

all_childcare_data = []

for file_path in all_files:
    file_date, file_year = extract_date_from_filename(file_path.name)
    if not file_date:
        print(f"Warning: Could not parse date from {file_path.name}. Skipping.")
        continue

    try:
        df = pd.read_csv(
            file_path,
            encoding='utf-8-sig',
            low_memory=False,
            dtype=str
        )
        if not all(col in df.columns for col in cols_to_use):
            print(f"Warning: {file_path.name} is missing one or more key columns. Skipping.")
            print(f"   Missing: {[col for col in cols_to_use if col not in df.columns]}")
            continue

        df = df[cols_to_use].copy()
        df['file_date'] = file_date
        df['year'] = file_year
        all_childcare_data.append(df)

    except Exception as e:
        print(f"ERROR: Could not read {file_path.name}. Error: {e}")

if not all_childcare_data:
    print("ERROR: No valid childcare data was loaded. Exiting.")
    exit()

master_df = pd.concat(all_childcare_data, ignore_index=True)
print(f"Loaded {len(master_df):,} total rows from {len(all_files)} files.")

#Rename columns
master_df = master_df.rename(columns={
    'Provider URN': 'provider_urn',
    'Provider Name': 'provider_name',
    'Provider Postcode': 'PCODE',
    'Most Recent Full: Overall Effectiveness': 'quality_rating',
    'Places': 'places'
})

#Clean Data and Create master DF
for col in ['provider_urn', 'provider_name', 'PCODE', 'quality_rating', 'places']:
    master_df[col] = master_df[col].str.strip().str.strip('"')
master_df.dropna(subset=['PCODE'], inplace=True)
master_df = master_df[master_df['PCODE'].str.upper() != 'REDACTED']
master_df['places'] = pd.to_numeric(
    master_df['places'].str.replace(',', '', regex=False),
    errors='coerce'
)
master_df.dropna(subset=['places'], inplace=True)
master_df['places'] = master_df['places'].astype(int)
master_df['quality_score_raw'] = pd.to_numeric(master_df['quality_rating'], errors='coerce')
master_df.dropna(subset=['quality_score_raw'], inplace=True)

#Remap Scores
quality_remapping = {
    1.0: 4,  # Outstanding
    2.0: 3,  # Good
    3.0: 2,  # Requires improvement
    4.0: 1  # Inadequate
}
master_df['quality_score'] = master_df['quality_score_raw'].map(quality_remapping)
master_df.dropna(subset=['quality_score'], inplace=True)
master_df['quality_score'] = master_df['quality_score'].astype(int)

#Keep latest record per provider per year
print("Deduplicating data: keeping latest entry per provider per year...")
master_df = master_df.sort_values(by='file_date', ascending=True)
master_df = master_df.drop_duplicates(subset=['provider_urn', 'year'], keep='last')
print(f"Loaded {len(master_df):,} valid, deduplicated provider records.")

#Lad Geospatial Data and locate providers
print("Loading LSOA boundaries and postcode locations...")
lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE)
postcode_df = pd.read_parquet(POSTCODE_LOOKUP_FILE)
postcode_gdf = gpd.GeoDataFrame(
    postcode_df,
    geometry=gpd.points_from_xy(postcode_df.longitude, postcode_df.latitude),
    crs="EPSG:4326"
)
master_df['PCODE_clean'] = master_df['PCODE'].astype(str).str.replace(' ', '').str.upper()
postcode_gdf['Postcode_clean'] = postcode_gdf['Postcode'].astype(str).str.replace(' ', '').str.upper()

childcare_gdf = master_df.merge(
    postcode_gdf,
    left_on='PCODE_clean',
    right_on='Postcode_clean',
    how='inner'
)
childcare_gdf = gpd.GeoDataFrame(childcare_gdf, geometry='geometry', crs="EPSG:4326")
print(f"Successfully geocoded {len(childcare_gdf)} provider records in the South West.")

if childcare_gdf.empty:
    print(f"ERROR: No matching providers found after geocoding.")
    exit()

#Time Series
print("Re-projecting coordinates for accurate calculations (EPSG:27700)...")
lsoa_gdf_proj = lsoa_gdf.to_crs("EPSG:27700")
childcare_gdf_proj = childcare_gdf.to_crs("EPSG:27700")
lsoa_gdf_proj['centroid'] = lsoa_gdf_proj.geometry.centroid
lsoa_coords = np.array(list(lsoa_gdf_proj.centroid.apply(lambda p: (p.x, p.y))))

all_annual_scores = []
static_details_results = []
available_years = sorted(childcare_gdf_proj['year'].unique())
latest_year = available_years[-1]

print(f"Found data for years: {available_years}. Latest year is {latest_year}.")
print(f"Calculating metrics from the {NEAREST_N_PROVIDERS} nearest providers for each LSOA, for each year...")

for year in available_years:
    print(f"  -> Processing year: {year}")

    providers_this_year = childcare_gdf_proj[childcare_gdf_proj['year'] == year]

    if providers_this_year.empty:
        print(f"  -> No providers found for {year}. Skipping.")
        continue

    provider_coords = np.array(list(providers_this_year.geometry.apply(lambda p: (p.x, p.y))))
    kdtree = cKDTree(provider_coords)

    distances_m, indices = kdtree.query(lsoa_coords, k=NEAREST_N_PROVIDERS)

    for i, lsoa_row in lsoa_gdf.iterrows():
        nearest_provider_indices = indices[i]
        nearest_distances_m = distances_m[i]

        nearest_providers_df = providers_this_year.iloc[nearest_provider_indices]

        #Key Metrics
        avg_quality = nearest_providers_df['quality_score'].mean()
        avg_distance_km = (nearest_distances_m.mean()) / 1000.0
        total_places_nearby = nearest_providers_df['places'].sum()

        annual_row = {
            'area_code': lsoa_row['area_code'],
            'year': year,
            'avg_childcare_quality_score': avg_quality,
            'avg_distance_to_childcare_km': avg_distance_km,
            'total_childcare_places_nearby': total_places_nearby
        }
        all_annual_scores.append(annual_row)

        if year == latest_year:
            static_row = {'area_code': lsoa_row['area_code']}
            for n in range(NEAREST_N_PROVIDERS):
                provider_info = nearest_providers_df.iloc[n]
                distance_km = nearest_distances_m[n] / 1000.0

                static_row[f'childcare_{n + 1}_name'] = provider_info['provider_name']
                static_row[f'childcare_{n + 1}_quality_rating'] = provider_info['quality_rating']
                static_row[f'childcare_{n + 1}_places'] = provider_info['places']
                static_row[f'childcare_{n + 1}_distance_km'] = distance_km
                static_row[f'childcare_{n + 1}_urn'] = provider_info['provider_urn']
            static_details_results.append(static_row)

annual_scores_df = pd.DataFrame(all_annual_scores)
static_details_df = pd.DataFrame(static_details_results)

#Back Fill Missing Years
print("Handling missing years (2018-2020) by back-filling...")

master_index = pd.MultiIndex.from_product(
    [lsoa_gdf['area_code'].unique(), YEARS_TO_PROCESS],
    names=['area_code', 'year']
)
final_scores_df = pd.DataFrame(index=master_index).reset_index()

final_scores_df = final_scores_df.merge(
    annual_scores_df,
    on=['area_code', 'year'],
    how='left'
)

final_scores_df = final_scores_df.sort_values(by=['area_code', 'year'])
score_cols = ['avg_childcare_quality_score', 'avg_distance_to_childcare_km', 'total_childcare_places_nearby']
final_scores_df[score_cols] = final_scores_df.groupby('area_code')[score_cols].bfill()

#Save Files
historical_cols = {
    'provider_urn': 'provider_urn',
    'year': 'year',
    'provider_name': 'provider_name',
    'quality_rating': 'quality_rating',
    'places': 'places'
}
historical_df = childcare_gdf[historical_cols.keys()].rename(columns=historical_cols)
historical_df = historical_df.drop_duplicates(subset=['provider_urn', 'year'])
historical_df.to_parquet(PROCESSED_DATA_DIR / "childcare_historical_data.parquet", index=False)
print(f"✅ Success! Saved HISTORICAL provider data to childcare_historical_data.parquet")
final_scores_df.to_parquet(ANNUAL_SCORES_OUTPUT, index=False)
print(
    f"Success! Saved ANNUAL scores for {len(final_scores_df['area_code'].unique())} LSOAs (2018-2025) to {ANNUAL_SCORES_OUTPUT.name}")
static_details_df.to_parquet(STATIC_DETAILS_OUTPUT, index=False)
print(
    f"Success! Saved STATIC details (from {latest_year}) for {len(static_details_df)} LSOAs to {STATIC_DETAILS_OUTPUT.name}")
print("Script finished.")