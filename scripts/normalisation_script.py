import pandas as pd
from pathlib import Path
import numpy as np
import sys

print("Starting Normalization Engine (Winsorized Min-Max 0-100)")

#Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "lsoa_annual_indicators_all_runs.parquet"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "lsoa_annual_normalized_scores.parquet"
SCORING_VARIABLES = {
    # Safety & Environment
    'air_quality_no2': ['no2_mean_concentration', True],
    'air_quality_pm25': ['pm25_mean_concentration', True],
    'crime': ['crime_rate_per_1000', True],
    # Socio-Economic / Health
    'income': ['Income_Rate', True],
    'employment': ['Employment_Rate', True],
    # Education
    'primary_read': ['primary_read_score_weighted', False],
    'primary_math': ['primary_math_score_weighted', False],
    'secondary_progress': ['secondary_progress_8_weighted', False],
    'secondary_attainment': ['secondary_attainment_8_weighted', False],
    # Services
    'childcare_quality': ['avg_childcare_quality_score', False],
}

#Normalisation Functions
def calculate_crime_rate(df):
    """Derives the crime rate per 1000 people from imputed data."""
    if 'crime_count' in df.columns and 'population' in df.columns:
        safe_population = df['population'].replace(0, 1).fillna(1)
        df['crime_rate_per_1000'] = (df['crime_count'] / safe_population) * 1000
    else:
        df['crime_rate_per_1000'] = np.nan
    return df

def winsorized_min_max(series: pd.Series, invert: bool, p_low: float = 2.5, p_high: float = 97.5) -> pd.Series:
    """
    Applies Winsorisation (clipping outliers) followed by Min-Max scaling to 0-100.
    Retains all data points but assigns the 0/100 bounds to extremes.
    """
    if series.empty or series.nunique() <= 1:
        return pd.Series(50.0, index=series.index)
    # 1. Winsorisation: Calculate bounds from data
    low_clip = series.quantile(p_low / 100)
    high_clip = series.quantile(p_high / 100)
    # 2. Clip the series: Values outside the bounds are set to the bounds
    clipped_series = series.clip(lower=low_clip, upper=high_clip)
    min_val = clipped_series.min()
    max_val = clipped_series.max()
    # Handle case where clipping results in constant data (min_val == max_val)
    if max_val == min_val:
        return pd.Series(50.0, index=series.index)
    # 3. Min-Max Scaling to 0-100
    if invert:
        # Low raw value = High score (100)
        score = ((max_val - clipped_series) / (max_val - min_val)) * 100
    else:
        # High raw value = High score (100)
        score = ((clipped_series - min_val) / (max_val - min_val)) * 100
    return score.round(2)

#Main
def main():
    if not DATA_FILE.exists():
        sys.exit(f"ERROR: Imputed data file not found at {DATA_FILE}. Please run 'rebuild_data.sh' first.")
    df_all_runs = pd.read_parquet(DATA_FILE)
    # Calculate the mean across the imputation runs to get the best estimate
    df_imputed_mean = df_all_runs.groupby(['area_code', 'year']).mean(numeric_only=True).reset_index()
    latest_year = 2024
    if latest_year not in df_imputed_mean['year'].unique():
        latest_year = df_imputed_mean['year'].max()
        print(f"Warning: 2024 data not found. Using data for max year: {latest_year}")
    df_2024 = df_imputed_mean[df_imputed_mean['year'] == latest_year].copy()
    # 2. Derive Crime Rate
    df_2024 = calculate_crime_rate(df_2024)
    # 3. & 4. Winsorize and Normalize to 0-100
    for key, (col_name, invert) in SCORING_VARIABLES.items():
        if col_name in df_2024.columns:
            print(f"-> Normalizing '{col_name}' (Invert={invert})...")
            # Apply Winsorised Min-Max Scaling
            df_2024[f'{key}_score'] = winsorized_min_max(df_2024[col_name], invert=invert)
            # Add the raw metric to the output for traceability
            if col_name not in df_2024.columns:
                df_2024[col_name] = df_imputed_mean[df_imputed_mean['year'] == latest_year][col_name]
    # Final cleanup and save
    output_cols = [
                      'area_code', 'year'
                  ] + [f'{key}_score' for key in SCORING_VARIABLES.keys() if f'{key}_score' in df_2024.columns]
    # Also include the core underlying metrics for the final database for convenience
    output_cols.extend([col for col, _ in SCORING_VARIABLES.values() if col in df_2024.columns])
    output_cols = list(set(output_cols))
    df_final = df_2024[output_cols]
    df_final.to_parquet(OUTPUT_FILE, index=False)
    print("\n----------------------------------------------------")
    print(f"SUCCESS! Final normalized scores saved to: {OUTPUT_FILE}")
    print(f"Processed {len(df_final)} LSOAs.")
    print("----------------------------------------------------")

if __name__ == "__main__":
    main()