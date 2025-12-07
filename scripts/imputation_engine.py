import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

print("Starting Master Data Build")
# Paths and Constants
N_IMPUTATIONS = 5
PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
ANNUAL_POP_FILE = PROCESSED_DATA_DIR / "lsoa_annual_population.parquet"
ANNUAL_CRIME_FILE = PROCESSED_DATA_DIR / "lsoa_annual_crime.parquet"
ANNUAL_AIR_QUALITY_FILE = PROCESSED_DATA_DIR / "lsoa_annual_air_quality.parquet"
ANNUAL_HEALTHCARE_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_healthcare_scores.parquet"
ANNUAL_CHILDCARE_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_childcare_scores.parquet"
STATIC_GREENSPACE_FILE = PROCESSED_DATA_DIR / "lsoa_greenspace.parquet"
STATIC_IMD_FILE = PROCESSED_DATA_DIR / "lsoa_imd.parquet"
STATIC_LATEST_HOUSE_PRICE_FILE = PROCESSED_DATA_DIR / "lsoa_latest_house_prices_imputed.parquet"
ANNUAL_PRIMARY_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_primary_weighted.parquet"
ANNUAL_SECONDARY_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_secondary_weighted.parquet"
OUTPUT_FILE_ALL_RUNS = PROCESSED_DATA_DIR / "lsoa_annual_indicators_all_runs.parquet"
YEARS = [2024]

# Helpers
def load_and_merge_file(master_df, file_path, on_cols, file_desc, cols_to_drop=None):
    if file_path.exists():
        print(f"  -> Loading {file_desc}...")
        df_to_merge = pd.read_parquet(file_path)
        if 'year' in df_to_merge.columns:
            df_to_merge = df_to_merge[df_to_merge['year'].isin(YEARS)]
        df_to_merge = df_to_merge.drop_duplicates(subset=on_cols)
        if cols_to_drop:
            cols_to_drop_existing = [col for col in cols_to_drop if col in df_to_merge.columns]
            if cols_to_drop_existing:
                df_to_merge = df_to_merge.drop(columns=cols_to_drop_existing)
        master_df = master_df.merge(df_to_merge, on=on_cols, how='left')
    else:
        print(f"ERROR: {file_path.name} not found. Stopping.")
        sys.exit(1)
    return master_df

def generate_impute_list(master_df):
    cols = []
    cols.extend(['crime_count', 'no2_mean_concentration', 'pm25_mean_concentration',
                 'IDACI_Rate', 'IMD_Score', 'Income_Rate', 'Employment_Rate', 'Health_Score',
                 'greenspace_percentage'])
    prefixes = ['primary_', 'secondary_', 'gp_', 'childcare_']
    metrics = ['pass_rate', 'read_score', 'math_score', 'progress_8', 'attainment_8',
               'satisfaction', 'quality_score', 'places', 'distance_km', 'distance', 'weighted']
    for col in master_df.columns:
        if any(col.startswith(p) for p in prefixes):
            if any(m in col for m in metrics):
                cols.append(col)
    valid_cols = sorted(list(set([c for c in cols if c in master_df.columns])))
    return [c for c in valid_cols if pd.api.types.is_numeric_dtype(master_df[c])]

def calculate_intermediate_aggregates(df):
    df = df.copy()
    gp_sat_cols = [c for c in df.columns if 'gp_' in c and 'satisfaction' in c and 'avg' not in c]
    gp_dist_cols = [c for c in df.columns if 'gp_' in c and 'distance' in c and 'avg' not in c]
    if gp_sat_cols: df['avg_gp_satisfaction'] = df[gp_sat_cols].mean(axis=1)
    if gp_dist_cols: df['avg_distance_to_gp_km'] = df[gp_dist_cols].mean(axis=1)
    cc_qual_cols = [c for c in df.columns if 'childcare_' in c and 'quality_score' in c and 'avg' not in c]
    cc_dist_cols = [c for c in df.columns if 'childcare_' in c and 'distance' in c and 'avg' not in c]
    cc_place_cols = [c for c in df.columns if 'childcare_' in c and 'places' in c and 'total' not in c]
    if cc_qual_cols: df['avg_childcare_quality_score'] = df[cc_qual_cols].mean(axis=1)
    if cc_dist_cols: df['avg_distance_to_childcare_km'] = df[cc_dist_cols].mean(axis=1)
    if cc_place_cols: df['total_childcare_places_nearby'] = df[cc_place_cols].sum(axis=1)
    return df

# Main
def main():
    try:
        # Load and Merge Data
        print("Step 1: Creating master grid...")
        lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE)
        lsoa_codes = lsoa_gdf['area_code'].unique()
        master_index = pd.MultiIndex.from_product([lsoa_codes, YEARS], names=['area_code', 'year'])
        master_df = pd.DataFrame(index=master_index).reset_index()
        print("Step 2: Merging Datasets...")
        master_df = load_and_merge_file(master_df, ANNUAL_POP_FILE, ['area_code', 'year'], "population")
        master_df = load_and_merge_file(master_df, ANNUAL_CRIME_FILE, ['area_code', 'year'], "crime",
                                        cols_to_drop=['population', 'raw_crime_count', 'months_of_data',
                                                      'is_full_year'])
        master_df = load_and_merge_file(master_df, ANNUAL_AIR_QUALITY_FILE, ['area_code', 'year'], "air quality")
        master_df = load_and_merge_file(master_df, ANNUAL_PRIMARY_SCORES_FILE, ['area_code', 'year'],
                                        "primary weighted")
        master_df = load_and_merge_file(master_df, ANNUAL_SECONDARY_SCORES_FILE, ['area_code', 'year'],
                                        "secondary weighted")
        master_df = load_and_merge_file(master_df, ANNUAL_HEALTHCARE_SCORES_FILE, ['area_code', 'year'],
                                        "healthcare scores")
        master_df = load_and_merge_file(master_df, ANNUAL_CHILDCARE_SCORES_FILE, ['area_code', 'year'],
                                        "childcare scores")
        master_df = load_and_merge_file(master_df, STATIC_GREENSPACE_FILE, ['area_code'], "greenspace")
        master_df = load_and_merge_file(master_df, STATIC_IMD_FILE, ['area_code'], "IMD")
        master_df = load_and_merge_file(master_df, STATIC_LATEST_HOUSE_PRICE_FILE, ['area_code'], "house prices")
        if 'annualized_crime_count' in master_df.columns:
            master_df = master_df.rename(columns={'annualized_crime_count': 'crime_count'})

        # Propagate Static Data
        print("Step 3: Propagating static data...")
        propagate_cols = [
            'greenspace_percentage', 'IMD_Score', 'Income_Rate', 'Employment_Rate',
            'Health_Score', 'IDACI_Rate', 'latest_median_house_price', 'avg_distance_to_gp_km'
        ]
        propagate_cols = [c for c in propagate_cols if c in master_df.columns]
        master_df[propagate_cols] = master_df.groupby('area_code')[propagate_cols].ffill().bfill()
        if 'population' in master_df.columns:
            master_df['population'] = master_df.groupby('area_code')['population'].ffill()
            master_df['population'] = master_df.groupby('year')['population'].transform(lambda x: x.fillna(x.median()))

        # Imputation
        print("\nStep 4: Running INDIVIDUAL Year MICE Imputation...")
        impute_cols = generate_impute_list(master_df)
        print(f"  -> Variables to impute: {len(impute_cols)}")
        final_completed_rows = []
        for year in YEARS:
            print(f"\n  >>> Processing Year: {year}")
            year_df = master_df[master_df['year'] == year].copy()
            if year_df.empty:
                print(f"      Warning: No data for {year}, skipping.")
                continue

            # Prepare Data
            data_to_impute = year_df[impute_cols].copy()
            valid_cols_for_year = data_to_impute.columns[data_to_impute.notna().any()].tolist()
            if len(valid_cols_for_year) < len(impute_cols):
                dropped = set(impute_cols) - set(valid_cols_for_year)
                print(f"      Note: {len(dropped)} columns are 100% empty in {year} and will be skipped.")
                data_to_impute = data_to_impute[valid_cols_for_year]

            # Imputation Loop (5 Runs for this specific year)
            for i in range(N_IMPUTATIONS):
                print(f"      Run {i + 1}/{N_IMPUTATIONS}...", end='\r')
                imputer = IterativeImputer(max_iter=10, sample_posterior=True, random_state=i, min_value=0)
                imputed_matrix = imputer.fit_transform(data_to_impute)
                current_run_df = year_df.copy()
                current_run_df[valid_cols_for_year] = imputed_matrix

                # Clamping/Rounding
                precise_bounds = {
                    'Employment_Rate': (0.0, 1.0), 'IDACI_Rate': (0.0, 1.0), 'Income_Rate': (0.0, 1.0),
                    'IMD_Score': (1.58, 100.0), 'Health_Score': (-4.0, 4.0), 'greenspace_percentage': (0.0, 100.0),
                    'latest_median_house_price': (106000.0, None), 'population': (715.00, None),
                    'crime_count': (0.0, None), 'no2_mean_concentration': (1.78, None),
                    'pm25_mean_concentration': (2.41, None),
                    'primary_math_score_weighted': (60.0, 130.0), 'primary_read_score_weighted': (60.0, 130.0),
                    'primary_pass_rate_weighted': (0.0, 1.0),
                    'secondary_attainment_8_weighted': (0.0, 90.0), 'secondary_progress_8_weighted': (-5.0, 5.0),
                }
                for col, (min_val, max_val) in precise_bounds.items():
                    if col in current_run_df.columns:
                        current_run_df[col] = current_run_df[col].clip(lower=min_val, upper=max_val)

                int_cols = ['population', 'latest_median_house_price', 'crime_count']
                for col in int_cols:
                    if col in current_run_df.columns:
                        current_run_df[col] = current_run_df[col].round(0)

                # Aggregates
                current_run_df = calculate_intermediate_aggregates(current_run_df)
                current_run_df['imputation_run'] = i
                final_completed_rows.append(current_run_df)
            print("")

        # Save Result
        print("\nStep 5: Aggregating imputation runs and saving final dataset...")
        all_runs_df = pd.concat(final_completed_rows)
        # Calculate the mean across the imputation runs for each LSOA/year
        exclude_cols = ['area_code', 'year', 'imputation_run']
        agg_cols = [c for c in all_runs_df.columns
                    if pd.api.types.is_numeric_dtype(all_runs_df[c]) and c not in exclude_cols]
        final_df = all_runs_df.groupby(['area_code', 'year'])[agg_cols].mean().reset_index()
        # Re-apply necessary rounding to the final mean values
        int_cols = ['population', 'latest_median_house_price', 'crime_count']
        for col in int_cols:
            if col in final_df.columns:
                final_df[col] = final_df[col].round(0)
        final_df.to_parquet(OUTPUT_FILE_ALL_RUNS, index=False)
        print(f"Success! Saved {len(final_df)} rows.")
        print(f"File: {OUTPUT_FILE_ALL_RUNS}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()