import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys
import numpy as np
from sklearn.impute import IterativeImputer

print("Starting master annual indicator table build (IMPUTATION ENGINE)...")

#Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
#Base Geographies
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
#Sparse Annual Data
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

#Define Years
YEARS = list(range(2018, 2026))


def load_and_merge_file(master_df, file_path, on_cols, file_desc, cols_to_drop=None):
    """Helper to load and merge parquet files."""
    if file_path.exists():
        print(f"  -> Loading {file_desc}...")
        df_to_merge = pd.read_parquet(file_path)
        if cols_to_drop:
            cols_to_drop_existing = [col for col in cols_to_drop if col in df_to_merge.columns]
            if cols_to_drop_existing:
                df_to_merge = df_to_merge.drop(columns=cols_to_drop_existing)

        master_df = master_df.merge(df_to_merge, on=on_cols, how='left')
    else:
        print(f"ERROR: {file_path.name} not found. Stopping build.")
        sys.exit(1)
    return master_df


def main():
    try:
        #Master Grid
        print("Step 1: Creating master grid...")
        if not LSOA_BOUNDARIES_FILE.exists():
            print(f"ERROR: Base LSOA boundary file not found.")
            sys.exit(1)
        lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE)
        lsoa_codes = lsoa_gdf['area_code'].unique()

        master_index = pd.MultiIndex.from_product([lsoa_codes, YEARS], names=['area_code', 'year'])
        master_df = pd.DataFrame(index=master_index).reset_index()
        master_df = master_df.sort_values(by=['area_code', 'year'])

        #Merge Data
        print("Step 2: Merging sparse data...")
        master_df = load_and_merge_file(master_df, ANNUAL_POP_FILE, ['area_code', 'year'], "population")
        master_df = load_and_merge_file(master_df, ANNUAL_CRIME_FILE, ['area_code', 'year'], "crime",
                                        cols_to_drop=['population', 'raw_crime_count', 'months_of_data',
                                                      'is_full_year'])
        master_df = load_and_merge_file(master_df, ANNUAL_AIR_QUALITY_FILE, ['area_code', 'year'], "air quality")
        master_df = load_and_merge_file(master_df, ANNUAL_PRIMARY_SCORES_FILE, ['area_code', 'year'], "primary scores")
        master_df = load_and_merge_file(master_df, ANNUAL_SECONDARY_SCORES_FILE, ['area_code', 'year'],
                                        "secondary scores")
        master_df = load_and_merge_file(master_df, ANNUAL_HEALTHCARE_SCORES_FILE, ['area_code', 'year'],
                                        "healthcare scores")
        master_df = load_and_merge_file(master_df, ANNUAL_CHILDCARE_SCORES_FILE, ['area_code', 'year'],
                                        "childcare scores")

        #Merge Static (Propagated)
        master_df = load_and_merge_file(master_df, STATIC_GREENSPACE_FILE, ['area_code'], "greenspace")
        master_df = load_and_merge_file(master_df, STATIC_IMD_FILE, ['area_code'], "IMD (includes IDACI)")
        master_df = load_and_merge_file(master_df, STATIC_LATEST_HOUSE_PRICE_FILE, ['area_code'], "house prices")

        #Merge UI Details
        master_df = load_and_merge_file(master_df, STATIC_SECONDARY_EDU_FILE, ['area_code'], "sec details")
        master_df = load_and_merge_file(master_df, STATIC_PRIMARY_EDU_FILE, ['area_code'], "pri details")
        master_df = load_and_merge_file(master_df, STATIC_HEALTHCARE_FILE, ['area_code'], "health details")
        master_df = load_and_merge_file(master_df, STATIC_CHILDCARE_FILE, ['area_code'], "child details")

        #Prepare
        if 'annualized_crime_count' in master_df.columns:
            master_df = master_df.rename(columns={'annualized_crime_count': 'crime_count'})

        #Imputation
        print("Step 4: Running Advanced Imputation...")

        #Propagate Static Data
        static_cols = [
            'greenspace_percentage',
            'IMD_Decile', 'Income_Decile', 'Employment_Decile', 'Health_Decile', 'IDACI_Decile',
            'latest_median_house_price', 'avg_distance_to_gp_km'
        ]
        #Only use cols that exist
        static_cols = [c for c in static_cols if c in master_df.columns]

        master_df[static_cols] = master_df.groupby('area_code')[static_cols].ffill().bfill()

        #Fallback for Static Data
        for col in ['IDACI_Decile', 'IMD_Decile']:
            if col in master_df.columns:
                missing_count = master_df[col].isna().sum()
                if missing_count > 0:
                    print(f"  -> Filling {missing_count} missing {col} with Median (5)")
                    master_df[col] = master_df[col].fillna(5)

        #Simple Imputation for Population
        master_df['population'] = master_df.groupby('area_code')['population'].ffill()
        master_df['population'] = master_df.groupby('year')['population'].transform(lambda x: x.fillna(x.median()))

        #Multivariate Iterative Imputation (MICE)
        impute_cols = [
            'crime_count',
            'no2_mean_concentration', 'pm25_mean_concentration',
            'avg_primary_scaled_score', 'avg_ks2_pass_rate',
            'avg_progress_8', 'avg_attainment_8',
            'avg_gp_satisfaction',
            'avg_childcare_quality_score', 'avg_distance_to_childcare_km', 'total_childcare_places_nearby'
        ]

        impute_cols = [c for c in impute_cols if c in master_df.columns]
        predictors = static_cols + impute_cols

        print(f"  -> Training Iterative Imputer on {len(predictors)} variables...")
        imputer = IterativeImputer(max_iter=10, random_state=0)

        data_for_imputation = master_df[predictors].copy()
        imputed_data = imputer.fit_transform(data_for_imputation)
        master_df[predictors] = imputed_data

        #Cleanup
        print("Step 5: Final cleanup...")

        non_numeric_cols = master_df.select_dtypes(exclude=np.number).columns.drop(['area_code'])
        master_df[non_numeric_cols] = master_df[non_numeric_cols].fillna('N/A')

        pos_cols = ['crime_count', 'population', 'total_childcare_places_nearby']
        for c in pos_cols:
            if c in master_df.columns:
                master_df[c] = master_df[c].clip(lower=0)

        #Ensure Integers
        int_cols = [
            'population', 'latest_median_house_price',
            'IMD_Decile', 'Income_Decile', 'Employment_Decile', 'Health_Decile', 'IDACI_Decile'
        ]
        for col in int_cols:
            if col in master_df.columns:
                master_df[col] = master_df[col].fillna(0).astype(int)

        master_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"Success! Master file saved to {OUTPUT_FILE}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()