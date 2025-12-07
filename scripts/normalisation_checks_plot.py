import pandas as pd
from pathlib import Path
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns


#Helpers
def calculate_crime_rate(df):
    """Derives the crime rate per 1000 people from imputed data."""
    if 'crime_count' in df.columns and 'population' in df.columns:
        safe_population = df['population'].replace(0, 1).fillna(1)
        df['crime_rate_per_1000'] = (df['crime_count'] / safe_population) * 1000
    else:
        df['crime_rate_per_1000'] = np.nan
    return df

def generate_winsorization_plot(df, variables_to_plot, output_path, percentile_low=2.5, percentile_high=97.5):
    """Generates a multi-panel box plot in a 2x2 grid, showing distribution and proposed clipping boundaries."""
    n_rows = 2
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12), sharex=False)
    axes = axes.flatten()
    fig.suptitle(f"Figure 14: Justification for Winsorisation ({percentile_low}% to {percentile_high}%) - 4 Worst Outliers",
                 fontsize=16,
                 fontweight='bold')
    display_name_map = {
        'crime_rate_per_1000': 'Crime Rate Per 1000',
        'greenspace_percentage': 'Greenspace Percentage',
        'secondary_progress_8_weighted': 'Secondary Progress 8 Weighted',
        'avg_childcare_quality_score': 'Avg Childcare Quality Score',
    }
    for i, col in enumerate(variables_to_plot):
        if col not in df.columns:
            axes[i].set_title(f"Data Missing for: {col}")
            continue
        series = df[col].dropna()
        if series.empty: continue
        low_clip = series.quantile(percentile_low / 100)
        high_clip = series.quantile(percentile_high / 100)
        # Determine Title for Plot
        title = display_name_map.get(col, col.replace('_', ' ').title())
        # Create box plot showing standard IQR and outliers
        sns.boxplot(y=series, ax=axes[i], orient='v',
                    flierprops={"marker": "o", "markersize": 5, "markerfacecolor": "red"})
        # Add visual lines for the proposed clipping boundary
        axes[i].axhline(low_clip, color='darkgreen', linestyle='--', linewidth=1.5, label=f'{percentile_low}% Clip')
        axes[i].axhline(high_clip, color='darkgreen', linestyle='--', linewidth=1.5, label=f'{percentile_high}% Clip')
        axes[i].set_title(f"{title} Distribution")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"\nSUCCESS: Saved plot to {output_path}")

#Main
def main():
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    data_file = PROJECT_DIR / "data" / "processed" / "lsoa_annual_indicators_all_runs.parquet"
    output_dir = PROJECT_DIR / "data" / "analysis_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "winsorization_justification_4worst_plot.png"
    if not data_file.exists():
        sys.exit("ERROR: Could not find data/processed/lsoa_annual_indicators_all_runs.parquet. Cannot generate plot.")
    df_all_runs = pd.read_parquet(data_file)
    df_imputed_mean = df_all_runs.groupby(['area_code', 'year']).mean(numeric_only=True).reset_index()
    df_imputed_mean = calculate_crime_rate(df_imputed_mean)
    latest_year = df_imputed_mean['year'].max()
    df_latest = df_imputed_mean[df_imputed_mean['year'] == latest_year].copy()
    # The 4 worst outlier variables (excluding House Price)
    variables = [
        'crime_rate_per_1000',
        'greenspace_percentage',
        'secondary_progress_8_weighted',
        'avg_childcare_quality_score'
    ]
    generate_winsorization_plot(df_latest, variables, output_path, percentile_low=2.5, percentile_high=97.5)

if __name__ == "__main__":
    main()