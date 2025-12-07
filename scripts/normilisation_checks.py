import pandas as pd
from pathlib import Path
import numpy as np
import sys
from scipy.stats import skew, kurtosis


#Helpers
def calculate_crime_rate(df):
    """Calculates crime rate per 1000 people using imputed data."""
    # Ensure crime_count and population exist and handle division by zero
    if 'crime_count' in df.columns and 'population' in df.columns:
        safe_population = df['population'].replace(0, 1).fillna(1)
        df['crime_rate_per_1000'] = (df['crime_count'] / safe_population) * 1000
    else:
        df['crime_rate_per_1000'] = np.nan
    return df

def run_normalization_check():
    """Loads final imputed data and runs statistical analysis."""
    # The final output file containing all MICE runs
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    data_file = PROJECT_DIR / "data" / "processed" / "lsoa_annual_indicators_all_runs.parquet"
    if not data_file.exists():
        sys.exit(
            "ERROR: Could not find data/processed/lsoa_annual_indicators_all_runs.parquet. Please run 'rebuild_data.sh' first.")
    df_all_runs = pd.read_parquet(data_file)
    df_imputed_mean = df_all_runs.groupby(['area_code', 'year']).mean(numeric_only=True).reset_index()
    # 1. Calculate the derived crime rate column
    df_imputed_mean = calculate_crime_rate(df_imputed_mean)
    # 2. Select variables relevant for normalization (based on project structure)
    target_columns = [
        'greenspace_percentage', 'no2_mean_concentration', 'pm25_mean_concentration',
        'crime_rate_per_1000', 'latest_median_house_price',
        'Income_Rate', 'Employment_Rate', 'Health_Score',
        'avg_gp_satisfaction', 'avg_childcare_quality_score',
        'secondary_progress_8_weighted', 'primary_read_score_weighted',
        'primary_math_score_weighted'
    ]
    # Filter for the latest year for the most relevant snapshot
    latest_year = df_imputed_mean['year'].max()
    df_latest = df_imputed_mean[df_imputed_mean['year'] == latest_year].copy().dropna(
        subset=[col for col in target_columns if col in df_imputed_mean.columns],
        how='all'
    )
    if df_latest.empty:
        sys.exit(f"ERROR: No data found for the latest year ({latest_year}) in the imputed file.")

    # Run Statistical Checks
    results = []
    print(f"\nAnalyzing Statistical Distribution for Year {latest_year} ({len(df_latest)} LSOAs)...")
    for col in target_columns:
        if col not in df_latest.columns: continue
        series = df_latest[col].dropna()
        if series.empty:
            results.append({'Indicator': col.replace('_', ' ').title(), 'Result': "No data after dropna"})
            continue
        results.append({
            'Indicator': col.replace('_', ' ').title(),
            'Mean': series.mean(),
            'Std. Dev.': series.std(),
            'Min': series.min(),
            'Max': series.max(),
            'Skewness': skew(series),
            'Kurtosis': kurtosis(series),
        })
    report_df = pd.DataFrame(results).set_index('Indicator')
    print("\n--- Imputed Data Statistical Check for Normalization Decision ---")
    with pd.option_context(
            'display.max_columns', None,
            'display.width', 1000,
            'display.float_format', '{:.2f}'.format
    ):
        print(report_df)
    print("\nCheck Complete. Use this table to inform your normalization decision.")

if __name__ == "__main__":
    run_normalization_check()