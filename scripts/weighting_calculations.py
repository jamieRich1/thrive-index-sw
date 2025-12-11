import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns

#Set display options for high-precision output to the console
np.set_printoptions(precision=5, suppress=True, linewidth=100)
pd.set_option('display.float_format', '{:.5f}'.format)
pd.set_option('display.width', 1000)

#Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
NORMALIZED_DATA_FILE = PROJECT_DIR / "data" / "processed" / "lsoa_annual_normalized_scores.parquet"
OUTPUT_BASE_DIR = PROJECT_DIR / "data" / "analysis_results_2024_refined_WEIGHTING"
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
TARGET_YEAR = 2024
N_COMPONENTS = 5
SOCIO_ECONOMIC_VARS = ['income_score', 'employment_score', 'crime_score']
ENVIRONMENTAL_SAFETY_VARS = ['air_quality_no2_score', 'air_quality_pm25_score']
SECONDARY_EDUCATION_VARS = ['secondary_progress_score', 'secondary_attainment_score']
PRIMARY_EDUCATION_VARS = ['primary_read_score', 'primary_math_score']
CHILDCARE_QUALITY_VARS = ['childcare_quality_score']
ALL_VARS = (
        SOCIO_ECONOMIC_VARS + PRIMARY_EDUCATION_VARS + SECONDARY_EDUCATION_VARS +
        ENVIRONMENTAL_SAFETY_VARS + CHILDCARE_QUALITY_VARS
)
PILLAR_NAMES = {
    1: 'RC1: Socio-Economic Deprivation',
    2: 'RC2: Environmental Safety',
    3: 'RC3: Secondary Education',
    4: 'RC4: Primary Education',
    5: 'RC5: Childcare Quality',
}

#Helpers
def get_standardized_data(df, variable_list):
    """
    Re-standardizes normalized scores (0-100) to Z-scores for PCA/FA accuracy.
    Inversion is NOT needed as data is already correctly oriented (high=good).
    """
    valid_vars = [v for v in variable_list if v in df.columns]
    df_subset = df[valid_vars].dropna()
    if df_subset.empty: return pd.DataFrame()
    # Use StandardScaler to convert normalized scores (0-100 scale) into z-scores
    scaler = StandardScaler()
    data_std = scaler.fit_transform(df_subset)
    return pd.DataFrame(data_std, columns=valid_vars, index=df_subset.index)

#Varimax Rotation Core Function
def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    """Performs Varimax (orthogonal) rotation on the loading matrix Phi."""
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = np.linalg.svd(
            np.dot(Phi.T, np.asarray(Lambda) ** 3 - (gamma / p) * np.dot(Lambda, np.diag(np.sum(Lambda ** 2, axis=0)))))
        R = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol: break
    return np.dot(Phi, R)


# OECD Weighting Calculation Function
def calculate_composite_weights_and_scores(pca, std_df, N_COMPONENTS):
    """Performs the two-level OECD weighting process."""
    raw_loadings = pca.components_[:N_COMPONENTS].T * np.sqrt(pca.explained_variance_[:N_COMPONENTS])
    rotated_loadings = varimax(raw_loadings)
    loading_df = pd.DataFrame(
        rotated_loadings,
        index=std_df.columns,
        columns=[f'RC{x + 1}' for x in range(N_COMPONENTS)]
    )
    # 1.1 Indicator Grouping (Assignment)
    abs_loadings = loading_df.abs()
    indicator_grouping = abs_loadings.idxmax(axis=1).rename('Assigned_RC')
    # 1.2 Indicator Weight Calculation (Level 1 Weights)
    squared_loadings_df = loading_df ** 2
    grouped_df = pd.DataFrame(index=std_df.columns,
                              columns=['Loading', 'L_Squared', 'Assigned_RC', 'Sum_L2_RC', 'Weight'])
    for rc_name in [f'RC{i + 1}' for i in range(N_COMPONENTS)]:
        assigned_indicators = indicator_grouping[indicator_grouping == rc_name].index
        if assigned_indicators.empty: continue
        sum_l2_rc = squared_loadings_df.loc[assigned_indicators, rc_name].sum()
        for indicator in assigned_indicators:
            l2 = squared_loadings_df.loc[indicator, rc_name]
            weight = l2 / sum_l2_rc
            grouped_df.loc[indicator, 'Loading'] = loading_df.loc[indicator, rc_name]
            grouped_df.loc[indicator, 'L_Squared'] = l2
            grouped_df.loc[indicator, 'Assigned_RC'] = rc_name
            grouped_df.loc[indicator, 'Sum_L2_RC'] = sum_l2_rc
            grouped_df.loc[indicator, 'Weight'] = weight
    # 2.1 Factor Weight Calculation (Level 2 Weights)
    rotated_variance = np.sum(rotated_loadings ** 2, axis=0)
    total_retained_variance = np.sum(rotated_variance)
    factor_weights = pd.DataFrame({
        'Rotated_Expl_Var': rotated_variance,
        'Factor_Weight': rotated_variance / total_retained_variance
    }, index=[f'RC{x + 1}' for x in range(N_COMPONENTS)])
    factor_weights['Pillar_Name'] = factor_weights.index.map(lambda x: PILLAR_NAMES.get(int(x[2:]), x))
    return {
        'indicator_weights_df': grouped_df,
        'factor_weights_df': factor_weights,
        'rotated_loadings_df': loading_df,
        'total_retained_variance': total_retained_variance
    }

#Heatmap Check
def plot_heatmap(loadings_df, year, output_dir):
    """Generates and saves a heatmap of the rotated factor loadings."""
    plt.figure(figsize=(9, 7))
    rc_labels = [PILLAR_NAMES.get(int(col[2:]), col) for col in loadings_df.columns]
    loading_order = loadings_df.abs().idxmax(axis=1).sort_values().index
    sorted_loadings = loadings_df.loc[loading_order]
    sns.heatmap(
        sorted_loadings,
        annot=True,
        fmt=".3f",
        cmap="vlag",
        cbar_kws={'label': 'Factor Loading'},
        linewidths=.5,
        linecolor='black',
        vmin=-1,
        vmax=1,
        yticklabels=True,
        xticklabels=rc_labels
    )
    plt.title(f'Figure 10: Rotated Factor Loadings Heatmap (Varimax) - {year}', fontsize=14)
    plt.xlabel('Rotated Component (RC) / Composite Pillar', fontsize=12)
    plt.ylabel('Indicator Variable (Grouped by Highest Loading)', fontsize=12)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # Save the plot
    heatmap_path = output_dir / f"rotated_loadings_heatmap_{year}.png"
    plt.savefig(heatmap_path)
    plt.close()
    print(f"\n--- Heatmap generated and saved to: {heatmap_path} ---")


#Main
def main():
    if not NORMALIZED_DATA_FILE.exists():
        print(f"Error: Normalized data file not found at {NORMALIZED_DATA_FILE}")
        print("Please ensure you have run the normalization script and saved the output to that location.")
        sys.exit(1)
    # Load the normalized scores (0-100 scale, correct inversion)
    df_all = pd.read_parquet(NORMALIZED_DATA_FILE)
    df_year = df_all[df_all['year'] == TARGET_YEAR].copy()
    if 'area_code' in df_year.columns: df_year = df_year.set_index('area_code')
    # Run PCA/FA on the z-scores of the normalized (0-100) data.
    std_all = get_standardized_data(df_year, ALL_VARS)
    if std_all.empty:
        print("Error: Standardized data is empty after dropping NaNs.")
        sys.exit(1)
    # 1. PCA and Component Retention
    pca = PCA()
    pca.fit(std_all)
    # 2. Varimax Rotation and Weight Calculation
    weights_results = calculate_composite_weights_and_scores(pca, std_all, N_COMPONENTS)
    total_retained_variance = weights_results['total_retained_variance']
    rotated_loadings_df = weights_results['rotated_loadings_df']
    plot_heatmap(rotated_loadings_df, TARGET_YEAR, OUTPUT_BASE_DIR)
    # 3. TERMINAL REPORT GENERATION
    print("\n" + "=" * 70)
    print(f"FACTOR ANALYSIS AND OECD WEIGHTING SUMMARY (ON NORMALIZED SCORES) ({TARGET_YEAR})")
    print("=" * 70)
    # A. PCA and Rotation Summary (Initial Metrics)
    print("\n1. PCA AND ROTATION SUMMARY")
    # Initial PCA Metrics
    print("\n1.1 Initial PCA Component Metrics:")
    print("Component | Eigenvalue | Variance Explained | Cumulative Variance")
    print("-----------------------------------------------------------------")
    for i, val in enumerate(pca.explained_variance_):
        exp_var = pca.explained_variance_ratio_[i]
        cum_var = np.cumsum(pca.explained_variance_ratio_)[i]
        status = ' (RETAINED)' if i < N_COMPONENTS else ''
        print(f"    PC{i + 1}: {val:.5f}   | {exp_var:.5f} | {cum_var:.5f} {status}")
    # Rotated Loadings Matrix
    print("\n\n1.2 Rotated Loadings Matrix (A_rotated) - Component Loadings:")
    print(rotated_loadings_df.to_string())
    print("This table shows the correlation (loading) of each indicator with its respective factor.")
    print("-" * 70)
    # B. OECD WEIGHTING PROCESS
    print("\n\n2. OECD COMPOSITE INDICATOR WEIGHTING PROCESS")
    # 2.1 Level 1: Indicator Weights
    print("\n2.1 LEVEL 1: INDICATOR WEIGHTS (wi,j) - Within Factor")
    # 2.1.1 Indicator Grouping Table
    print("\n2.1.1 Indicator Grouping (Assignment):")
    grouping_table = weights_results['indicator_weights_df'].reset_index()
    grouping_table = grouping_table[['index', 'Loading', 'Assigned_RC']].rename(columns={
        'index': 'Indicator Variable', 'Loading': 'Highest Loading (L)', 'Assigned_RC': 'Assigned RC'
    })
    grouping_table['Pillar Name'] = grouping_table['Assigned RC'].map(lambda x: PILLAR_NAMES.get(int(x[2:]), ''))
    # Output grouping table
    print(grouping_table.to_string(index=False))
    # 2.1.2 Indicator Weight Calculation Table
    print("\n\n2.1.2 Indicator Weight Calculation (Normalized Squared Loadings):")
    indicator_calc_table = weights_results['indicator_weights_df'].reset_index()
    indicator_calc_table = indicator_calc_table[[
        'index', 'Assigned_RC', 'Loading', 'L_Squared', 'Sum_L2_RC', 'Weight'
    ]].rename(columns={
        'index': 'Indicator', 'Loading': 'Loading (L)', 'L_Squared': 'L^2 (Squared Loading)',
        'Sum_L2_RC': 'Sum L^2 of Assigned RC', 'Weight': 'Indicator Weight (wi,j)'
    })
    # Output Indicator Weight table
    print(indicator_calc_table.to_string(index=False))
    # Verification Table
    sum_check = indicator_calc_table.groupby('Assigned_RC')['Indicator Weight (wi,j)'].sum().to_frame(
        name='Sum of Weights')
    print("\nVerification: Sum of Weights per RC (Should be ~1.0):")
    print(sum_check.to_string())
    # 2.2 Level 2: Factor Weights
    print("\n\n2.2 LEVEL 2: FACTOR WEIGHTS (wfactor,j) - Between Factors")
    # 2.2.1 Factor Weight Calculation Table
    print("\n2.2.1 Factor Weight Calculation (Proportion of Explained Variance):")
    # Create the factor table using the index as the RC name
    factor_table = weights_results['factor_weights_df'].copy()
    factor_table['Total Retained Var (B)'] = total_retained_variance
    factor_table = factor_table.rename(columns={
        'Rotated_Expl_Var': 'Rotated Variance Explained (A)',
        'Factor_Weight': 'Factor Weight (A/B)',
        'Pillar_Name': 'Factor Description'
    })
    factor_table = factor_table[
        ['Factor Description', 'Rotated Variance Explained (A)', 'Total Retained Var (B)', 'Factor Weight (A/B)']
    ]
    # Add Total row
    total_row = pd.DataFrame({
        'Factor Description': ['TOTAL'],
        'Rotated Variance Explained (A)': [factor_table['Rotated Variance Explained (A)'].sum()],
        'Total Retained Var (B)': [factor_table['Total Retained Var (B)'].iloc[0]],
        'Factor Weight (A/B)': [factor_table['Factor Weight (A/B)'].sum()]
    }, index=['TOTAL'])
    factor_table = pd.concat([factor_table, total_row])
    # Output Factor Weight table
    print(factor_table.to_string())
    print("Analysis Complete. Results printed to terminal and heatmap saved.")

if __name__ == "__main__":
    main()