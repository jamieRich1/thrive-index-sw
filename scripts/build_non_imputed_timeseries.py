import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys

print("Starting Master Data Build (Non-Imputed)")
# Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"
LSOA_BOUNDARIES_FILE = PROCESSED_DATA_DIR / "boundaries_lsoa.geoparquet"
ANNUAL_POP_FILE = PROCESSED_DATA_DIR / "lsoa_annual_population.parquet"
ANNUAL_CRIME_FILE = PROCESSED_DATA_DIR / "lsoa_annual_crime.parquet"
ANNUAL_AIR_QUALITY_FILE = PROCESSED_DATA_DIR / "lsoa_annual_air_quality.parquet"
ANNUAL_HEALTHCARE_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_healthcare_scores.parquet"

# CHANGED: We now ignore the scoring childcare file and use the Deep Dive one
# ANNUAL_CHILDCARE_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_childcare_scores.parquet"

STATIC_GREENSPACE_FILE = PROCESSED_DATA_DIR / "lsoa_greenspace.parquet"
STATIC_IMD_FILE = PROCESSED_DATA_DIR / "lsoa_imd.parquet"
STATIC_LATEST_HOUSE_PRICE_FILE = PROCESSED_DATA_DIR / "lsoa_latest_house_prices_imputed.parquet"
ANNUAL_PRIMARY_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_primary_weighted.parquet"
ANNUAL_SECONDARY_SCORES_FILE = PROCESSED_DATA_DIR / "lsoa_annual_secondary_weighted.parquet"

# Deep Dive Specific Files
ANNUAL_PRIMARY_DEEPDIVE_FILE = PROCESSED_DATA_DIR / "lsoa_annual_primary_deepdive.parquet"
ANNUAL_SECONDARY_DEEPDIVE_FILE = PROCESSED_DATA_DIR / "lsoa_annual_secondary_deepdive.parquet"
ANNUAL_CHILDCARE_DEEPDIVE_FILE = PROCESSED_DATA_DIR / "lsoa_annual_childcare_deepdive.parquet"  # New

OUTPUT_FILE_NON_IMPUTED = PROCESSED_DATA_DIR / "lsoa_annual_indicators_non_imputed.parquet"
YEARS = list(range(2018, 2026))


# Helpers
def load_and_merge_file(master_df, file_path, on_cols, file_desc, cols_to_drop=None):
    if file_path.exists():
        print(f"  -> Loading {file_desc}...")
        try:
            df_to_merge = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Warning: Could not read {file_path.name}: {e}")
            return master_df

        # Handle duplicates if any exist in source
        if 'year' in on_cols:
            df_to_merge = df_to_merge.drop_duplicates(subset=on_cols)
        else:
            df_to_merge = df_to_merge.drop_duplicates(subset=['area_code'])

        if cols_to_drop:
            cols_to_drop_existing = [col for col in cols_to_drop if col in df_to_merge.columns]
            if cols_to_drop_existing:
                df_to_merge = df_to_merge.drop(columns=cols_to_drop_existing)

        # Use an outer merge to keep all LSOA/Year combinations, ensuring NaN for missing data
        master_df = master_df.merge(df_to_merge, on=on_cols, how='left')
    else:
        print(f"Warning: {file_path.name} not found. Skipping merge for this file.")
        # We don't exit here, to allow partial builds if a file is missing
    return master_df


def calculate_intermediate_aggregates(df):
    """Calculates aggregate columns from existing raw columns."""
    df = df.copy()

    def get_related_cols(prefix, metric):
        return [c for c in df.columns if
                f'{prefix}_' in c and f'_{metric}' in c and 'avg' not in c and 'total' not in c]

    # GP Scores Aggregates
    gp_sat_cols = get_related_cols('gp', 'satisfaction')
    gp_dist_cols = get_related_cols('gp', 'distance')
    if gp_sat_cols: df['avg_gp_satisfaction'] = df[gp_sat_cols].mean(axis=1)
    if gp_dist_cols: df['avg_distance_to_gp_km'] = df[gp_dist_cols].mean(axis=1)
    # Childcare Scores Aggregates
    cc_qual_cols = get_related_cols('childcare', 'quality_score')
    cc_dist_cols = get_related_cols('childcare', 'distance')
    cc_place_cols = get_related_cols('childcare', 'places')
    if cc_qual_cols: df['avg_childcare_quality_score'] = df[cc_qual_cols].mean(axis=1)
    if cc_dist_cols: df['avg_distance_to_childcare_km'] = df[cc_dist_cols].mean(axis=1)
    # Use sum for total places
    if cc_place_cols: df['total_childcare_places_nearby'] = df[cc_place_cols].sum(axis=1)
    return df


# Main
def main():
    try:
        # Load and Merge Data
        print("Step 1: Creating master grid...")
        lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE)
        lsoa_codes = lsoa_gdf['area_code'].unique()
        # Create the full LSOA/Year index for the time series
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
        master_df = load_and_merge_file(master_df, ANNUAL_PRIMARY_DEEPDIVE_FILE, ['area_code', 'year'],
                                        "primary deep dive data")
        master_df = load_and_merge_file(master_df, ANNUAL_SECONDARY_DEEPDIVE_FILE, ['area_code', 'year'],
                                        "secondary deep dive data")
        master_df = load_and_merge_file(master_df, ANNUAL_CHILDCARE_DEEPDIVE_FILE, ['area_code', 'year'],
                                        "childcare deep dive data")

        # Static files merge on 'area_code' only
        master_df = load_and_merge_file(master_df, STATIC_GREENSPACE_FILE, ['area_code'], "greenspace")
        master_df = load_and_merge_file(master_df, STATIC_IMD_FILE, ['area_code'], "IMD")
        master_df = load_and_merge_file(master_df, STATIC_LATEST_HOUSE_PRICE_FILE, ['area_code'], "house prices")
        if 'annualized_crime_count' in master_df.columns:
            master_df = master_df.rename(columns={'annualized_crime_count': 'crime_count'})

        # Propagate Static Data
        print("Step 3: Propagating static data...")
        propagate_cols = [
            'greenspace_percentage', 'IMD_Score', 'Income_Rate', 'Employment_Rate',
            'Health_Score', 'IDACI_Rate', 'latest_median_house_price'
        ]
        # 'avg_distance_to_gp_km' will now be calculated in the aggregate step
        propagate_cols = [c for c in propagate_cols if c in master_df.columns]

        # Use ffill/bfill to propagate static data across years for the same LSOA
        master_df[propagate_cols] = master_df.groupby('area_code')[propagate_cols].ffill().bfill()

        # Population handling (using ffill/median for missing values)
        if 'population' in master_df.columns:
            master_df['population'] = master_df.groupby('area_code')['population'].ffill()
            master_df['population'] = master_df.groupby('year')['population'].transform(lambda x: x.fillna(x.median()))

        # Calculate Intermediate Aggregates
        print("Step 4: Calculating intermediate aggregates...")
        master_df = calculate_intermediate_aggregates(master_df)
        # Final Save Result
        print("\nStep 5: Saving final non-imputed dataset...")
        # Apply rounding to non-imputed data for consistency
        int_cols = ['population', 'latest_median_house_price', 'crime_count']
        for col in int_cols:
            if col in master_df.columns:
                master_df[col] = master_df[col].round(0)
        if 'imputation_run' in master_df.columns:
            master_df = master_df.drop(columns=['imputation_run'])

        master_df.to_parquet(OUTPUT_FILE_NON_IMPUTED, index=False)
        print(f"Success! Saved {len(master_df)} rows.")
        print(f"File: {OUTPUT_FILE_NON_IMPUTED}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()