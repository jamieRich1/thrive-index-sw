import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys

print("Starting master annual indicator table build (v-all-timeseries)...")

#Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
ANNUAL_POP_FILE = PROCESSED_DATA_DIR / "lsoa_annual_population.parquet"
ANNUAL_CRIME_FILE = PROCESSED_DATA_DIR / "lsoa_annual_crime.parquet"
ANNUAL_AIR_QUALITY_FILE = PROCESSED_DATA_DIR / "lsoa_annual_air_quality.parquet"
ANNUAL_PRIMARY_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_primary_scores.parquet"
ANNUAL_SECONDARY_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_secondary_scores.parquet"
ANNUAL_HEALTHCARE_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_healthcare_scores.parquet"
ANNUAL_CHILDCARE_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_childcare_scores.parquet"
STATIC_GREENSPACE_FILE = PROCESSED_DATA_DIR / "lsoa_greenspace.parquet"
STATIC_SECONDARY_EDU_FILE = PROCESSED_DATA_DIR / "lsoa_secondary_education_details.parquet"
STATIC_PRIMARY_EDU_FILE = PROCESSED_DATA_DIR / "lsoa_primary_education_details.parquet"
STATIC_HEALTHCARE_FILE = PROCESSED_DATA_DIR / "lsoa_healthcare_details.parquet"
STATIC_CHILDCARE_FILE = PROCESSED_DATA_DIR / "lsoa_childcare_details.parquet"
STATIC_IMD_FILE = PROCESSED_DATA_DIR / "lsoa_imd.parquet"
STATIC_LATEST_HOUSE_PRICE_FILE = PROCESSED_DATA_DIR / "lsoa_latest_house_prices_imputed.parquet"
OUTPUT_FILE = PROCESSED_DATA_DIR / "lsoa_annual_indicators.parquet"

#Define Years
YEARS = list(range(2018, 2026))

#Load and Validate
try:
    print("Step 1: Loading LSOA boundaries...")
    lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE)
    lsoa_codes = lsoa_gdf['area_code'].unique()
    print(f"Loaded {len(lsoa_codes)} LSOAs.")

    print("Step 2: Creating master LSOA-Year index...")
    master_index = pd.MultiIndex.from_product([lsoa_codes, YEARS], names=['area_code', 'year'])
    master_df = pd.DataFrame(index=master_index).reset_index()

    #Mergin Annual Timeseries
    print("Step 3: Merging annual (time-aware) data...")


    def merge_annual_file(master_df, file_path, file_desc):
        if file_path.exists():
            df_annual = pd.read_parquet(file_path)
            master_df = master_df.merge(df_annual, on=['area_code', 'year'], how='left')
            print(f"  -> Merged {len(df_annual['area_code'].unique())} LSOAs for {file_desc}.")
        else:
            print(f"ERROR: {file_path.name} not found.")
            sys.exit(1)
        return master_df


    master_df = merge_annual_file(master_df, ANNUAL_POP_FILE, "population")

    #Crime rename and file merges
    if ANNUAL_CRIME_FILE.exists():
        df_crime = pd.read_parquet(ANNUAL_CRIME_FILE)
        df_crime = df_crime.rename(columns={'annualized_crime_count': 'crime_count'})
        master_df = master_df.merge(
            df_crime[['area_code', 'year', 'crime_count']],
            on=['area_code', 'year'],
            how='left'
        )
        print(f"  -> Merged {len(df_crime['area_code'].unique())} LSOAs for crime.")
    else:
        print(f"ERROR: {ANNUAL_CRIME_FILE.name} not found.")
        sys.exit(1)

    master_df = merge_annual_file(master_df, ANNUAL_AIR_QUALITY_FILE, "air quality")
    master_df = merge_annual_file(master_df, ANNUAL_PRIMARY_SCORES_FILE, "primary school")
    master_df = merge_annual_file(master_df, ANNUAL_SECONDARY_SCORES_FILE, "secondary school")
    master_df = merge_annual_file(master_df, ANNUAL_HEALTHCARE_SCORES_FILE, "healthcare")
    master_df = merge_annual_file(master_df, ANNUAL_CHILDCARE_SCORES_FILE, "childcare")

    #Merge Static Data
    print("Step 4: Merging static data and filling time gaps...")

    static_files_to_merge = {
        STATIC_GREENSPACE_FILE: ['greenspace_percentage'],
        STATIC_SECONDARY_EDU_FILE: [
                                       f'school_{i}_name' for i in range(1, 4)
                                   ] + [
                                       f'school_{i}_progress_8' for i in range(1, 4)
                                   ] + [
                                       f'school_{i}_attainment_8' for i in range(1, 4)
                                   ] + [
                                       f'school_{i}_data_year' for i in range(1, 4)
                                   ] + [
                                       f'school_{i}_nftype' for i in range(1, 4)
                                   ]+  [
                                       f'school_{i}_urn' for i in range(1, 4)
                                   ],
        STATIC_PRIMARY_EDU_FILE: [
                                     f'primary_school_{i}_name' for i in range(1, 4)
                                 ] + [
                                     f'primary_school_{i}_pass_rate' for i in range(1, 4)
                                 ] + [
                                     f'primary_school_{i}_avg_scaled_score' for i in range(1, 4)
                                 ] + [
                                     f'primary_school_{i}_data_year' for i in range(1, 4)
                                 ]+  [
                                     f'primary_school_{i}_urn' for i in range(1, 4)
                                 ],
        STATIC_HEALTHCARE_FILE: [
                                    'avg_distance_to_gp_km',
                                ] + [
                                    f'gp_{i}_org_code' for i in range(1, 4)
                                ] + [
                                    f'gp_{i}_name' for i in range(1, 4)
                                ] + [
                                    f'gp_{i}_satisfaction' for i in range(1, 4)
                                ] + [
                                    f'gp_{i}_data_year' for i in range(1, 4)
                                ],

        STATIC_CHILDCARE_FILE: [
                                   f'childcare_{i}_name' for i in range(1, 4)
                               ] + [
                                   f'childcare_{i}_quality_rating' for i in range(1, 4)
                               ] + [
                                   f'childcare_{i}_places' for i in range(1, 4)
                               ] + [
                                   f'childcare_{i}_distance_km' for i in range(1, 4)
                               ]+  [
                                   f'childcare_{i}_urn' for i in range(1, 4)
                               ],
        STATIC_IMD_FILE: [
            'IMD_Decile', 'Income_Decile', 'Employment_Decile', 'Health_Decile'
        ],
        STATIC_LATEST_HOUSE_PRICE_FILE: ['latest_median_house_price']
    }

    #Sort master_df for correct ffill/bfill
    master_df = master_df.sort_values(by=['area_code', 'year'])

    for file_path, cols in static_files_to_merge.items():
        if file_path.exists():
            try:
                df_static = pd.read_parquet(file_path)
                cols_to_keep = ['area_code'] + [col for col in cols if col in df_static.columns]
                missing_cols = [col for col in cols if col not in df_static.columns]
                if missing_cols:
                    print(f"  -> Warning: File {file_path.name} is missing expected columns: {missing_cols}")
                df_static = df_static[cols_to_keep]
                master_df = master_df.merge(df_static, on='area_code', how='left')
                fill_cols = [col for col in cols_to_keep if col != 'area_code']
                fill_cols = [c for c in fill_cols if not c.endswith('_satisfaction')]
                master_df[fill_cols] = master_df.groupby('area_code')[fill_cols].ffill().bfill()
                print(f"  -> Merged and time-filled {file_path.name}")
            except Exception as e:
                print(f"Warning: Failed to merge {file_path.name}. Error: {e}")
        else:
            print(f"Warning: Static file not found, skipping: {file_path.name}")

    #Clean and Save
    print("Step 5: Final cleanup and save...")
    master_df.dropna(subset=['population'], inplace=True)

    numeric_cols = [
        'crime_count', 'no2_mean_concentration', 'pm25_mean_concentration',
        'avg_primary_scaled_score', 'avg_ks2_pass_rate', 'avg_progress_8', 'avg_attainment_8',
        'avg_gp_satisfaction',
        'avg_distance_to_childcare_km', 'avg_childcare_quality_score', 'total_childcare_places_nearby',
        'greenspace_percentage', 'avg_distance_to_gp_km',
        'IMD_Decile', 'Income_Decile', 'Employment_Decile', 'Health_Decile', 'latest_median_house_price',
        'school_1_progress_8', 'school_1_attainment_8', 'school_1_data_year',
        'school_2_progress_8', 'school_2_attainment_8', 'school_2_data_year',
        'school_3_progress_8', 'school_3_attainment_8', 'school_3_data_year',
        'primary_school_1_pass_rate', 'primary_school_1_avg_scaled_score', 'primary_school_1_data_year',
        'primary_school_2_pass_rate', 'primary_school_2_avg_scaled_score', 'primary_school_2_data_year',
        'primary_school_3_pass_rate', 'primary_school_3_avg_scaled_score', 'primary_school_3_data_year',
        'gp_1_satisfaction', 'gp_1_data_year',
        'gp_2_satisfaction', 'gp_2_data_year',
        'gp_3_satisfaction', 'gp_3_data_year',
        'childcare_1_places', 'childcare_1_distance_km',
        'childcare_2_places', 'childcare_2_distance_km',
        'childcare_3_places', 'childcare_3_distance_km',
    ]

    fill_values = {col: 0 for col in numeric_cols if col in master_df.columns}
    master_df.fillna(fill_values, inplace=True)
    master_df.fillna('N/A', inplace=True)

    int_cols = [
        'population', 'IMD_Decile', 'Income_Decile', 'Employment_Decile', 'Health_Decile',
        'latest_median_house_price', 'school_1_data_year', 'school_2_data_year', 'school_3_data_year',
        'primary_school_1_data_year', 'primary_school_2_data_year', 'primary_school_3_data_year',
        'gp_1_data_year', 'gp_2_data_year', 'gp_3_data_year'
    ]
    for col in int_cols:
        if col in master_df.columns:
            if master_df[col].dtype == 'object':
                master_df[col] = pd.to_numeric(master_df[col], errors='coerce').fillna(0)
            master_df[col] = master_df[col].astype(int)

    master_df.to_parquet(OUTPUT_FILE, index=False)

    print(f"Success! Master annual indicator file built with {len(master_df)} LSOA/year records.")
    print(f"File saved to {OUTPUT_FILE}")

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    sys.exit(1)

print("Script finished.")