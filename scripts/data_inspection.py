import pandas as pd
import geopandas as gpd
from pathlib import Path
import sys
import numpy as np

print("Starting Master Data Inspection (Selected Years Only)")

#Constants and Paths
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
REPORT_OUTPUT_FILE = PROCESSED_DATA_DIR / "data_quality_report.csv"
YEARS = [2024]


#Helpers
def load_and_merge_file(master_df, file_path, on_cols, file_desc, cols_to_drop=None):
    """Loads a parquet file, filters it by YEAR if applicable, and merges it into the master DataFrame."""
    if file_path.exists():
        print(f"  -> Loading {file_desc}...")
        try:
            df_to_merge = pd.read_parquet(file_path)
        except Exception as e:
            print(f"ERROR: Failed to read {file_path.name}. Error: {e}")
            sys.exit(1)
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

def get_mode(x):
    """Calculates the mode of a Series, returning the first mode if multiple exist, or NaN if empty."""
    m = x.mode()
    if not m.empty: return m.iloc[0]
    return np.nan

def generate_quality_report(df):
    """Generates a DataFrame containing quality statistics for all columns."""
    print("  -> Calculating statistics...")
    summary_data = []
    for col in df.columns:
        series = df[col]
        dtype = series.dtype
        total_rows = len(series)
        missing_count = series.isna().sum()
        missing_pct = (missing_count / total_rows) * 100
        mean_val = np.nan;
        min_val = np.nan;
        max_val = np.nan;
        std_val = np.nan
        mode_val = get_mode(series)
        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            mean_val = desc.get('mean')
            min_val = desc.get('min')
            max_val = desc.get('max')
            std_val = desc.get('std')
        summary_data.append({
            'Column Name': col,
            'Data Type': str(dtype),
            'Total Rows': total_rows,
            'Missing Values': missing_count,
            'Missingness (%)': round(missing_pct, 2),
            'Mean': round(mean_val, 2) if pd.notnull(mean_val) else None,
            'Min': round(min_val, 2) if pd.notnull(min_val) else None,
            'Max': round(max_val, 2) if pd.notnull(max_val) else None,
            'Std Dev': round(std_val, 2) if pd.notnull(std_val) else None,
            'Mode': mode_val
        })
    return pd.DataFrame(summary_data)

#Main
def main():
    try:
        #Load and Merge Data
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

        # Other annual scores
        master_df = load_and_merge_file(master_df, ANNUAL_HEALTHCARE_SCORES_FILE, ['area_code', 'year'],
                                        "healthcare scores")
        master_df = load_and_merge_file(master_df, ANNUAL_CHILDCARE_SCORES_FILE, ['area_code', 'year'],
                                        "childcare scores")
        # Static datasets (merged on area_code only)
        master_df = load_and_merge_file(master_df, STATIC_GREENSPACE_FILE, ['area_code'], "greenspace")
        master_df = load_and_merge_file(master_df, STATIC_IMD_FILE, ['area_code'], "IMD")
        master_df = load_and_merge_file(master_df, STATIC_LATEST_HOUSE_PRICE_FILE, ['area_code'], "house prices")
        # Rename for clarity if column exists
        if 'annualized_crime_count' in master_df.columns:
            master_df = master_df.rename(columns={'annualized_crime_count': 'crime_count'})

        #Propagate Static Data
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

        #Analysis
        print("\n" + "=" * 50)
        print("DATASET ANALYSIS REPORT (2018, 2019, 2023, 2024)")
        print("=" * 50)
        total_cells = master_df.size
        total_missing = master_df.isna().sum().sum()
        total_missing_pct = (total_missing / total_cells) * 100
        print(f"Shape: {master_df.shape[0]} rows x {master_df.shape[1]} columns")
        print(f"Overall Dataset Missingness: {total_missing_pct:.2f}%")
        print("-" * 50)
        report_df = generate_quality_report(master_df)
        report_df = report_df.sort_values(by='Missingness (%)', ascending=False)
        print("\nFull Data Quality Report (Sorted by Missingness):")
        print(report_df[['Column Name', 'Missingness (%)', 'Mean', 'Min', 'Max']].to_string(index=False))
        report_df.to_csv(REPORT_OUTPUT_FILE, index=False)
        print(f"\nFULL REPORT SAVED TO: {REPORT_OUTPUT_FILE}")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()