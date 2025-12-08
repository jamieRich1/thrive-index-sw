import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MASTER_FILE = PROCESSED_DIR / "lsoa_annual_indicators.parquet"
VARIABLES_TO_TRACE = [
    'crime_count',
    'no2_mean_concentration',
    'pm25_mean_concentration',
    'greenspace_percentage',
    'IDACI_Rate',
    'population',
    'latest_median_house_price',
    'IMD_Score',
    'Income_Rate',
    'Employment_Rate',
    'Health_Score',

    # Granular Service Metrics
    'primary_school_1_pass_rate', 'primary_school_2_pass_rate', 'primary_school_3_pass_rate',
    'primary_school_1_read_score', 'primary_school_2_read_score', 'primary_school_3_read_score',
    'primary_school_1_math_score', 'primary_school_2_math_score', 'primary_school_3_math_score',
    'school_1_progress_8', 'school_2_progress_8', 'school_3_progress_8',
    'school_1_attainment_8', 'school_2_attainment_8', 'school_3_attainment_8',
    'gp_1_satisfaction', 'gp_2_satisfaction', 'gp_3_satisfaction',
    'childcare_1_quality_score', 'childcare_2_quality_score', 'childcare_3_quality_score',
    'childcare_1_places', 'childcare_2_places', 'childcare_3_places'
]

# Trace Map
TRACE_MAP = {
    'crime_count': ('lsoa_annual_crime.parquet', ['annualized_crime_count'], False, False),
    'no2_mean_concentration': ('lsoa_annual_air_quality.parquet', ['no2_mean_concentration'], False, False),
    'pm25_mean_concentration': ('lsoa_annual_air_quality.parquet', ['pm25_mean_concentration'], False, False),
    'population': ('lsoa_annual_population.parquet', ['population'], False, False),
    'greenspace_percentage': ('lsoa_greenspace.parquet', ['greenspace_percentage'], True, False),
    'IDACI_Rate': ('lsoa_imd.parquet', ['IDACI_Rate'], True, False),
    'latest_median_house_price': ('lsoa_latest_house_prices_imputed.parquet', ['latest_median_house_price'], True,
                                  False),
    'IMD_Score': ('lsoa_imd.parquet', ['IMD_Score'], True, False),
    'Income_Rate': ('lsoa_imd.parquet', ['Income_Rate'], True, False),
    'Employment_Rate': ('lsoa_imd.parquet', ['Employment_Rate'], True, False),
    'Health_Score': ('lsoa_imd.parquet', ['Health_Score'], True, False),
    **{f'primary_school_{i}_pass_rate': ('lsoa_annual_primary_scores.parquet', [f'primary_school_{i}_pass_rate'], False,
                                         True) for i in range(1, 4)},
    **{f'primary_school_{i}_read_score': ('lsoa_annual_primary_scores.parquet', [f'primary_school_{i}_read_score'],
                                          False, True) for i in range(1, 4)},
    **{f'primary_school_{i}_math_score': ('lsoa_annual_primary_scores.parquet', [f'primary_school_{i}_math_score'],
                                          False, True) for i in range(1, 4)},
    **{f'school_{i}_progress_8': ('lsoa_annual_secondary_scores.parquet', [f'school_{i}_progress_8'], False, True) for i
       in range(1, 4)},
    **{f'school_{i}_attainment_8': ('lsoa_annual_secondary_scores.parquet', [f'school_{i}_attainment_8'], False, True)
       for i in range(1, 4)},
    **{f'gp_{i}_satisfaction': ('lsoa_annual_healthcare_scores.parquet', [f'gp_{i}_satisfaction'], False, True) for i in
       range(1, 4)},
    **{f'childcare_{i}_quality_score': ('lsoa_annual_childcare_scores.parquet', [f'childcare_{i}_quality_score'], False,
                                        True) for i in range(1, 4)},
    **{f'childcare_{i}_places': ('lsoa_annual_childcare_scores.parquet', [f'childcare_{i}_places'], False, True) for i
       in range(1, 4)},
}

# Helpers
def load_data():
    """Loads master file and all necessary intermediate files."""
    if not MASTER_FILE.exists():
        sys.exit("❌ Master file not found. Run rebuild_data.sh first.")
    master_df = pd.read_parquet(MASTER_FILE)
    intermediate_dfs = {}
    for final_col in VARIABLES_TO_TRACE:
        trace_info = TRACE_MAP.get(final_col, (None, None, False, False))
        file_name, _, _, _ = trace_info
        if file_name and file_name not in intermediate_dfs:
            try:
                intermediate_dfs[file_name] = pd.read_parquet(PROCESSED_DIR / file_name)
            except Exception as e:
                print(f"Warning: Could not load intermediate file {file_name}: {e}")
                intermediate_dfs[file_name] = None
    return master_df, intermediate_dfs

def get_sample_lsoas(df, n=5):
    """Picks n random LSOAs."""
    lsoas = df['area_code'].unique()
    rng = np.random.default_rng()
    return list(rng.choice(lsoas, size=min(n, len(lsoas)), replace=False))

def compare_values(raw_val, final_val, is_aggregated=False):
    """
    Compares values and determines match/imputation status.
    is_aggregated is NOT used for the main logic here, but it's passed for consistency.
    """
    try:
        raw_val = float(raw_val) if pd.notna(raw_val) else np.nan
    except:
        raw_val = np.nan
    try:
        final_val = float(final_val) if pd.notna(final_val) else np.nan
    except:
        final_val = np.nan
    if pd.isna(raw_val) and pd.notna(final_val):
        return "IMPUTED", final_val
    elif pd.isna(raw_val) and pd.isna(final_val):
        return "MATCH (Empty)", np.nan
    elif pd.notna(raw_val) and pd.notna(final_val) and np.isclose(raw_val, final_val, atol=0.01):
        return "MATCH", final_val
    else:
        return "MISMATCH/LOSS", final_val

# Main Audit Function
def run_traceability_audit():
    master_df, intermediate_dfs = load_data()
    sample_lsoas = get_sample_lsoas(master_df)
    print("FORENSIC TRACEABILITY AUDIT: RAW/INTERMEDIATE vs. FINAL MASTER FILE")
    print(f"Sampling {len(sample_lsoas)} LSOAs across all years ({min(master_df['year'])} - {max(master_df['year'])}).")

    # Prepare Header
    header = f"\n{'FINAL_COL':<25} | {'LSOA_CODE':<15} | {'YEAR':<5} | {'RAW_VAL (Source)':<15} | {'FINAL_VAL':<15} | {'STATUS':<25} | {'RAW_DF_ROW':<15}"
    print(header)

    # Iterate through LSOAs and Years
    for lsoa in sample_lsoas:
        lsoa_data = master_df[master_df['area_code'] == lsoa].sort_values('year')
        for index, master_row in lsoa_data.iterrows():
            year = int(master_row['year'])
            # Iterate through all target variables
            for final_col in VARIABLES_TO_TRACE:
                trace_info = TRACE_MAP.get(final_col, (None, None, False, False))
                file_name, raw_cols_to_check, is_static, is_aggregated = trace_info
                if not file_name or not raw_cols_to_check: continue

                # Get Raw/Intermediate Value (Single Source Column)
                intermediate_df = intermediate_dfs.get(file_name)
                if intermediate_df is None: continue
                raw_val = np.nan
                raw_row_index = 'N/A'
                raw_col_to_use = raw_cols_to_check[0]
                if is_static:
                    raw_rows = intermediate_df[intermediate_df['area_code'] == lsoa]
                else:
                    raw_rows = intermediate_df[
                        (intermediate_df['area_code'] == lsoa) & (intermediate_df['year'] == year)]
                if not raw_rows.empty and raw_col_to_use in raw_rows.columns:
                    raw_row = raw_rows.iloc[0]
                    raw_row_index = raw_rows.index[0]
                    raw_val = pd.to_numeric(raw_row[raw_col_to_use], errors='coerce')
                final_val = master_row[final_col]
                status, reported_val = compare_values(raw_val, final_val)
                print(
                    f"{final_col:<25} | {lsoa:<15} | {year:<5} | {str(raw_val)[:15]:<15} | {str(reported_val)[:15]:<15} | {status:<25} | Row: {raw_row_index}")
    print("\nAudit Complete. Review the 'STATUS' column for any 'MISMATCH/LOSS' entries.")

if __name__ == "__main__":
    run_traceability_audit()