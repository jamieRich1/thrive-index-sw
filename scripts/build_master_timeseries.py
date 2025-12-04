import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

print("Starting master annual indicator table build")

# Paths and Constants
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
STATIC_IMD_FILE = PROCESSED_DATA_DIR / "lsoa_imd.parquet"
STATIC_LATEST_HOUSE_PRICE_FILE = PROCESSED_DATA_DIR / "lsoa_latest_house_prices_imputed.parquet"
OUTPUT_FILE = PROCESSED_DATA_DIR / "lsoa_annual_indicators.parquet"
YEARS = list(range(2018, 2026))

# Helpers
def load_and_merge_file(master_df, file_path, on_cols, file_desc, cols_to_drop=None):
    if file_path.exists():
        print(f"  -> Loading {file_desc}...")
        df_to_merge = pd.read_parquet(file_path)
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

def calculate_who_score_vectorized(pollutant, concentration):
    """Re-implements scoring logic for Air Quality."""
    if pollutant == 'no2':
        xp, yp = [0, 10, 20, 40], [100, 90, 50, 10]
    elif pollutant == 'pm25':
        xp, yp = [0, 5, 10, 25], [100, 90, 50, 10]
    else:
        return 0
    return np.interp(concentration, xp, yp)

def generate_impute_list(master_df):
    """
    Dynamically finds all granular columns (school_1, gp_2, etc.) to run MICE on.
    """
    cols = []
    cols.extend(['crime_count', 'no2_mean_concentration', 'pm25_mean_concentration',
                 'IDACI_Rate', 'IMD_Score', 'Income_Rate', 'Employment_Rate', 'Health_Score',
                 'greenspace_percentage'])
    prefixes = ['school_', 'primary_school_', 'gp_', 'childcare_']
    metrics = ['progress_8', 'attainment_8', 'pass_rate', 'read_score', 'math_score',
               'satisfaction', 'quality_score', 'places']
    for col in master_df.columns:
        if any(col.startswith(p) for p in prefixes) and any(col.endswith(m) for m in metrics):
            cols.append(col)
    valid_cols = sorted(list(set([c for c in cols if c in master_df.columns])))
    return [c for c in valid_cols if pd.api.types.is_numeric_dtype(master_df[c])]

# Main
def main():
    try:
        # Master Grid
        print("Step 1: Creating master grid...")
        lsoa_gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE)
        lsoa_codes = lsoa_gdf['area_code'].unique()
        master_index = pd.MultiIndex.from_product([lsoa_codes, YEARS], names=['area_code', 'year'])
        master_df = pd.DataFrame(index=master_index).reset_index()

        # Merge Sparse Data
        print("Step 2: Merging sparse annual data...")
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

        # Merge Static Data
        print("Step 3: Merging static data...")
        master_df = load_and_merge_file(master_df, STATIC_GREENSPACE_FILE, ['area_code'], "greenspace")
        master_df = load_and_merge_file(master_df, STATIC_IMD_FILE, ['area_code'], "IMD")
        master_df = load_and_merge_file(master_df, STATIC_LATEST_HOUSE_PRICE_FILE, ['area_code'], "house prices")
        if 'annualized_crime_count' in master_df.columns:
            master_df = master_df.rename(columns={'annualized_crime_count': 'crime_count'})

        # Imputation
        print("Step 4: MICE Imputation...")
        # Propagate Static Data
        propagate_cols = [
            'greenspace_percentage',
            'IMD_Score', 'Income_Rate', 'Employment_Rate', 'Health_Score', 'IDACI_Rate',
            'latest_median_house_price', 'avg_distance_to_gp_km'
        ]
        propagate_cols = [c for c in propagate_cols if c in master_df.columns]
        master_df[propagate_cols] = master_df.groupby('area_code')[propagate_cols].ffill().bfill()
        # Population Fill
        if 'population' in master_df.columns:
            master_df['population'] = master_df.groupby('area_code')['population'].ffill()
            master_df['population'] = master_df.groupby('year')['population'].transform(lambda x: x.fillna(x.median()))
        # MICE on Granular Data
        impute_cols = generate_impute_list(master_df)
        print(f"  -> Identified {len(impute_cols)} variables for imputation.")
        # Create Imputer
        imputer = IterativeImputer(max_iter=10, random_state=0)
        # Run MICE (Only on numeric columns found)
        data_to_impute = master_df[impute_cols].copy()
        # Safety check: Ensure no non-numeric columns slipped in
        data_to_impute = data_to_impute.select_dtypes(include=[np.number])
        imputed_data = imputer.fit_transform(data_to_impute)
        master_df[data_to_impute.columns] = imputed_data

        # Post-Processing & Aggregation
        print("Step 5: Post-processing and Aggregation...")
        # Secondary Education
        p8_cols = [c for c in master_df.columns if 'progress_8' in c and 'avg' not in c]
        att8_cols = [c for c in master_df.columns if 'attainment_8' in c and 'avg' not in c]
        if p8_cols: master_df['avg_progress_8'] = master_df[p8_cols].mean(axis=1)
        if att8_cols: master_df['avg_attainment_8'] = master_df[att8_cols].mean(axis=1)
        # Primary Education
        pass_cols = [c for c in master_df.columns if 'pass_rate' in c and 'avg' not in c]
        read_cols = [c for c in master_df.columns if 'read_score' in c and 'avg' not in c]
        math_cols = [c for c in master_df.columns if 'math_score' in c and 'avg' not in c]
        if pass_cols: master_df['avg_ks2_pass_rate'] = master_df[pass_cols].mean(axis=1)
        # Calculate Scaled Score (Average of Read/Math columns across all 3 schools)
        scaled_cols = read_cols + math_cols
        if scaled_cols:
            master_df['avg_primary_scaled_score'] = master_df[scaled_cols].mean(axis=1)
        # Healthcare
        gp_sat_cols = [c for c in master_df.columns if 'gp_' in c and 'satisfaction' in c]
        gp_dist_cols = [c for c in master_df.columns if 'gp_' in c and 'distance' in c]
        if gp_sat_cols: master_df['avg_gp_satisfaction'] = master_df[gp_sat_cols].mean(axis=1)
        if gp_dist_cols: master_df['avg_distance_to_gp_km'] = master_df[gp_dist_cols].mean(axis=1)
        # Childcare
        cc_qual_cols = [c for c in master_df.columns if 'childcare_' in c and 'quality_score' in c]
        cc_dist_cols = [c for c in master_df.columns if 'childcare_' in c and 'distance' in c]
        cc_place_cols = [c for c in master_df.columns if 'childcare_' in c and 'places' in c]
        if cc_qual_cols: master_df['avg_childcare_quality_score'] = master_df[cc_qual_cols].mean(axis=1)
        if cc_dist_cols: master_df['avg_distance_to_childcare_km'] = master_df[cc_dist_cols].mean(axis=1)
        if cc_place_cols: master_df['total_childcare_places_nearby'] = master_df[cc_place_cols].sum(axis=1)
        # Air Quality
        master_df['no2_score'] = master_df['no2_mean_concentration'].apply(
            lambda x: calculate_who_score_vectorized('no2', x))
        master_df['pm25_score'] = master_df['pm25_mean_concentration'].apply(
            lambda x: calculate_who_score_vectorized('pm25', x))
        master_df['air_quality_score'] = master_df[['no2_score', 'pm25_score']].mean(axis=1)

        # Bounds & Types
        non_numeric_cols = master_df.select_dtypes(exclude=np.number).columns.drop(['area_code'])
        master_df[non_numeric_cols] = master_df[non_numeric_cols].fillna('N/A')
        # Clean Negatives
        pos_cols = ['crime_count', 'population', 'total_childcare_places_nearby', 'no2_mean_concentration',
                    'pm25_mean_concentration']
        for c in pos_cols:
            if c in master_df.columns:
                master_df[c] = master_df[c].clip(lower=0)
        # Clean Integers
        int_cols = ['population', 'latest_median_house_price', 'total_childcare_places_nearby']
        for col in int_cols:
            if col in master_df.columns:
                master_df[col] = master_df[col].fillna(0).astype(int)

        # Deduplication
        initial_len = len(master_df)
        master_df = master_df.drop_duplicates(subset=['area_code', 'year'])
        if len(master_df) < initial_len:
            print(f"  -> Removed {initial_len - len(master_df)} duplicate rows.")
        master_df.to_parquet(OUTPUT_FILE, index=False)
        print(f"Success! Master file saved to {OUTPUT_FILE}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()