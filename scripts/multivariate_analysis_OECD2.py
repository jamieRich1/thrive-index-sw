import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pingouin as pg
import geopandas as gpd
from matplotlib.colors import ListedColormap
import numpy as np

#Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "lsoa_annual_indicators_all_runs.parquet"
OUTPUT_BASE_DIR = PROJECT_DIR / "data" / "analysis_results_2024_refined"
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
YEARS_TO_ANALYZE = [2024]
TARGET_YEAR = YEARS_TO_ANALYZE[0]

# PC1: Socio-Economic Deprivation
SOCIO_ECONOMIC_VARS = [
    'Income_Rate',
    'Employment_Rate',
    'crime_rate_per_1000',
]
# PC2: Environmental Safety
ENVIRONMENTAL_SAFETY_VARS = [
    'no2_mean_concentration',
    'pm25_mean_concentration'
]
# PC3: Secondary Education
SECONDARY_EDUCATION_VARS = [
    'secondary_progress_8_weighted',
    'secondary_attainment_8_weighted',
]
# PC4: Primary Education
PRIMARY_EDUCATION_VARS = [
    'primary_read_score_weighted',
    'primary_math_score_weighted',
]
# PC5: Childcare Quality
CHILDCARE_QUALITY_VARS = [
    'avg_childcare_quality_score',
]
# All variables
ALL_VARS = (
        SOCIO_ECONOMIC_VARS + PRIMARY_EDUCATION_VARS + SECONDARY_EDUCATION_VARS +
        ENVIRONMENTAL_SAFETY_VARS + CHILDCARE_QUALITY_VARS
)
# Variables requiring inversion (High value = Bad outcome)
NEGATIVE_VARS = [
    'crime_rate_per_1000',
    'no2_mean_concentration',
    'pm25_mean_concentration',
    'Income_Rate',
    'Employment_Rate',
]

#Helpers
def calculate_cronbach_robust(df, variable_list):
    """Calculates Cronbach's Alpha for a list of variables."""
    valid_vars = [v for v in variable_list if v in df.columns]
    if not valid_vars: return 0.0, 0.0
    df_raw = df[valid_vars].copy().dropna()
    if df_raw.empty: return 0.0, 0.0
    for col in NEGATIVE_VARS:
        if col in df_raw.columns:
            df_raw[col] = df_raw[col] * -1
    try:
        alpha_raw = pg.cronbach_alpha(data=df_raw)[0]
    except:
        alpha_raw = 0.0
    try:
        df_std = (df_raw - df_raw.mean()) / df_raw.std()
        alpha_std = pg.cronbach_alpha(data=df_std)[0]
    except:
        alpha_std = 0.0
    return alpha_raw, alpha_std

def get_standardized_data(df, variable_list):
    """Standardizes and inverts variables for PCA/Clustering."""
    valid_vars = [v for v in variable_list if v in df.columns]
    df_subset = df[valid_vars].dropna()
    if df_subset.empty: return pd.DataFrame()
    for col in NEGATIVE_VARS:
        if col in df_subset.columns:
            df_subset[col] = -df_subset[col]
    scaler = StandardScaler()
    data_std = scaler.fit_transform(df_subset)
    return pd.DataFrame(data_std, columns=valid_vars, index=df_subset.index)

def calculate_crime_rate(df):
    """Derives the risk rate from imputed count and population."""
    if 'crime_count' in df.columns and 'population' in df.columns:
        # Crime rate = (Count / Population) * 1000
        safe_population = df['population'].replace(0, 1).fillna(1)
        df['crime_rate_per_1000'] = (df['crime_count'] / safe_population) * 1000
    else:
        df['crime_rate_per_1000'] = np.nan
    return df

def plot_scree_cumulative_single(result, output_path, year):
    """Plots the Scree Plot and Cumulative Variance Plot side-by-side (REUSED)."""
    pca = result['pca']
    variances = pca.explained_variance_ratio_
    cum_variances = variances.cumsum()
    x_indices = range(len(variances))
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left Plot: Scree Plot (Explained Variance)
    ax1 = axes[0]
    ax1.plot(x_indices, variances, marker='o', linestyle='-', color='blue')
    ax1.set_title(f"Scree Plot - Explained Variance by PC ({year})", fontsize=14)
    ax1.set_xlabel("Principal Component Index", fontsize=12)
    ax1.set_ylabel("Variance Explained", fontsize=12)
    ax1.set_xticks(x_indices, [f"PC{j + 1}" for j in x_indices])
    ax1.grid(True, linestyle='--')

    # Right Plot: Cumulative Variance Plot
    ax2 = axes[1]
    ax2.plot(x_indices, cum_variances, marker='s', linestyle='-', color='green')
    ax2.axhline(0.80, color='red', linestyle='--', label='80% Threshold')
    ax2.set_title(f"Cumulative Variance Explained by PC ({year})", fontsize=14)
    ax2.set_xlabel("Principal Component Index", fontsize=12)
    ax2.set_ylabel("Cumulative Variance Explained", fontsize=12)
    ax2.set_xticks(x_indices, [f"PC{j + 1}" for j in x_indices])
    ax2.set_ylim(0, 1.05)
    ax2.legend()
    ax2.grid(True, linestyle='--')

    fig.suptitle(f"Figure 5: PCA Component Selection Plots ({year})", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()

def plot_biplot_single(result, output_path, year):
    """Plots a single Structural Biplot with 5 FACTOR PILLARS."""
    pca = result['pca']
    std_df = result['std_df']
    loadings = pca.components_.T
    labels = std_df.columns
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle("Figure 11: Structural Biplot: 5 Factor-Driven Pillars (Crime in PC1 Test)", fontsize=16,
                 fontweight='bold')
    ax.set_title(
        f"{year} (PC1: {pca.explained_variance_ratio_[0]:.1%} | PC2: {pca.explained_variance_ratio_[1]:.1%})",
        fontsize=12)
    colors = {
        'Socio-Economic': '#e41a1c',  # Red
        'Environmental Safety': '#377eb8',  # Blue
        'Secondary Education': '#4daf4a',  # Green
        'Primary Education': '#ff7f00',  # Orange
        'Childcare Quality': '#984ea3',  # Purple
    }
    for j, label in enumerate(labels):
        x, y = loadings[j, 0], loadings[j, 1]
        color, marker, pillar_label = 'gray', '?', 'Uncategorized'
        if label in SOCIO_ECONOMIC_VARS:
            color = colors['Socio-Economic']
            marker = 'o' if label != 'crime_count' else 'x'
            pillar_label = 'Socio-Economic'
        elif label in ENVIRONMENTAL_SAFETY_VARS:
            color = colors['Environmental Safety']
            marker = 's'
            pillar_label = 'Environmental Safety'
        elif label in SECONDARY_EDUCATION_VARS:
            color = colors['Secondary Education']
            marker = 'D'
            pillar_label = 'Secondary Education'
        elif label in PRIMARY_EDUCATION_VARS:
            color = colors['Primary Education']
            marker = '^'
            pillar_label = 'Primary Education'
        elif label in CHILDCARE_QUALITY_VARS:
            color = colors['Childcare Quality']
            marker = '*'
            pillar_label = 'Childcare Quality'
        ax.arrow(0, 0, x, y, color=color, alpha=0.6, head_width=0.03, lw=1.5)
        ax.text(x * 1.15, y * 1.15, label.replace('_', ' ').title().replace('Weighted', ''),
                color='black', ha='center', va='center', fontsize=9)
        ax.scatter([x], [y], color=color, marker=marker, s=50)
    ax.set_xlim(-1.0, 1.0);
    ax.set_ylim(-1.0, 1.0)
    ax.axhline(0, color='black', lw=1);
    ax.axvline(0, color='black', lw=1)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.grid(True, linestyle='--')
    legend_elements = [
        Line2D([0], [0], color=colors['Socio-Economic'], marker='o', lw=0,
               label='Socio-Economic (PC1, includes Crime)'),
        Line2D([0], [0], color=colors['Environmental Safety'], marker='s', lw=0, label='Environmental Safety (PC2)'),
        Line2D([0], [0], color=colors['Secondary Education'], marker='D', lw=0, label='Secondary Education (PC3)'),
        Line2D([0], [0], color=colors['Primary Education'], marker='^', lw=0, label='Primary Education (PC4)'),
        Line2D([0], [0], color=colors['Childcare Quality'], marker='*', lw=0, label='Childcare Quality (PC5)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_heatmap_single(result, output_path, year):
    """Plots a single Loadings Heatmap for the target year."""
    pca = result['pca']
    std_df = result['std_df']
    loadings = pca.components_[:5].T
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        loadings, annot=True, cmap='coolwarm', center=0, fmt='.2f',
        xticklabels=[f'PC{x + 1}' for x in range(loadings.shape[1])],
        yticklabels=[c.replace('_', ' ').title().replace('Weighted', '') for c in std_df.columns],
        cbar_kws={'label': 'Loading Score'}
    )
    plt.title(f"Figure 6: Variable Loadings on First {loadings.shape[1]} Components ({year})", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_means_single(result, output_path, year):
    """Plots a single Cluster Means (Profile) Plot for the target year (REUSED)."""
    std_df = result['std_df'].copy()
    clusters = result['clusters']
    std_df['Cluster'] = clusters
    cluster_means = std_df.groupby('Cluster').mean()
    n_clusters = len(cluster_means)
    colors = sns.color_palette("bright", n_clusters)
    plt.figure(figsize=(12, 7))
    for cluster_id in range(n_clusters):
        plt.plot(cluster_means.columns, cluster_means.iloc[cluster_id],
                 marker='o', linewidth=2, label=f'Cluster {cluster_id}', color=colors[cluster_id])
    plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Avg (Z=0)')
    plt.title(f"Figure 7: Cluster Profiles: Standardized Means ({year})", fontsize=16)
    plt.xlabel("Indicator Variable", fontsize=12)
    plt.ylabel("Z-Score (Standard Deviations from Mean)", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(title="Cluster ID")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_scatter_single(result, output_path, year):
    """Plots a single Cluster Scatter Plot in PC space for the target year."""
    pca_data = result['pca_data']
    clusters = result['clusters']
    exp_var = result['pca'].explained_variance_ratio_
    plot_df = pd.DataFrame(pca_data[:, :2], columns=['PC1', 'PC2'])
    plot_df['Cluster'] = clusters
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        x='PC1', y='PC2', hue='Cluster', data=plot_df,
        palette='bright', s=40,
        ax=ax
    )
    plt.title(f"Cluster Segmentation in PC Space ({year})", fontsize=16)
    plt.xlabel(f"PC1 ({exp_var[0]:.1%})", fontsize=12)
    plt.ylabel(f"PC2 ({exp_var[1]:.1%})", fontsize=12)
    plt.legend(title='Cluster ID')
    plt.grid(True)
    ax.axhline(0, color='black', lw=0.5);
    ax.axvline(0, color='black', lw=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_map_single(result, output_path, project_dir, year):
    """Plots a single Choropleth Map for the target year (REUSED)."""
    LSOA_FILE = project_dir / "data" / "processed" / "boundaries_lsoa.geoparquet"
    if not LSOA_FILE.exists():
        print(f"Warning: LSOA Boundary file not found at {LSOA_FILE}. Skipping map generation.")
        return
    lsoa_gdf = gpd.read_parquet(LSOA_FILE)
    cluster_series = result['cluster_series']
    map_gdf = lsoa_gdf.merge(cluster_series.rename('Cluster'), left_on='area_code', right_index=True, how='inner')
    n_clusters = map_gdf['Cluster'].nunique()
    cmap = ListedColormap(sns.color_palette("bright", n_clusters))
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    map_gdf.plot(
        column='Cluster', ax=ax, cmap=cmap, legend=True,
        edgecolor='black', linewidth=0.1, categorical=True,
        legend_kwds={'title': 'Cluster ID', 'loc': 'lower left', 'fontsize': 10}
    )
    ax.set_title(f"Figure 8: Geographic Cluster Distribution ({year})", fontsize=16, fontweight='bold')
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

#Main
def main():
    if not DATA_FILE.exists():
        print(f"Error: Data file not found at {DATA_FILE}")
        return
    print("Loading Master Data...")
    df_all = pd.read_parquet(DATA_FILE)
    if 'imputation_run' in df_all.columns:
        df_all = df_all[df_all['imputation_run'] == 0]
    results = {}
    year = TARGET_YEAR
    print(f"\n--- PHASE 1: Computation for {year} ---")
    df_year = df_all[df_all['year'] == year].copy()
    if df_year.empty:
        print(f"Error: No data found for {year}.")
        return
    if 'area_code' in df_year.columns: df_year = df_year.set_index('area_code')
    df_year = calculate_crime_rate(df_year)
    std_all = get_standardized_data(df_year, ALL_VARS)
    if std_all.empty:
        print("Error: Standardized data is empty after dropping NaNs.")
        return

    # PCA
    pca = PCA()
    pca.fit(std_all)

    # Kaiser Criterion
    eigenvalues = pca.explained_variance_
    kaiser_components = np.sum(eigenvalues > 1.0)
    variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)

    # Reliability
    a_socio_raw, a_socio_std = calculate_cronbach_robust(df_year, SOCIO_ECONOMIC_VARS)
    a_env_safe_raw, a_env_safe_std = calculate_cronbach_robust(df_year, ENVIRONMENTAL_SAFETY_VARS)
    a_sec_ed_raw, a_sec_ed_std = calculate_cronbach_robust(df_year, SECONDARY_EDUCATION_VARS)
    a_pri_ed_raw, a_pri_ed_std = calculate_cronbach_robust(df_year, PRIMARY_EDUCATION_VARS)
    a_child_raw, a_child_std = calculate_cronbach_robust(df_year, CHILDCARE_QUALITY_VARS)
    a_tot_raw, a_tot_std = calculate_cronbach_robust(df_year, ALL_VARS)

    # Clustering
    N_COMPONENTS = 5
    N_CLUSTERS = 5
    pca_data = pca.transform(std_all)
    N_COMPONENTS = min(N_COMPONENTS, pca_data.shape[1])
    pca_input = pca_data[:, :N_COMPONENTS]
    print(f"Clustering with N_COMPONENTS={N_COMPONENTS} and N_CLUSTERS={N_CLUSTERS} (for >80% variance coverage)...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pca_input)
    results[year] = {
        'std_df': std_all,
        'pca': pca,
        'pca_data': pca_data,
        'clusters': clusters,
        'cluster_series': pd.Series(clusters, index=df_year.index),
        'alphas': {
            'socio_economic': (a_socio_raw, a_socio_std),
            'environmental_safety': (a_env_safe_raw, a_env_safe_std),
            'secondary_education': (a_sec_ed_raw, a_sec_ed_std),
            'primary_education': (a_pri_ed_raw, a_pri_ed_std),
            'childcare_quality': (a_child_raw, a_child_std),
            'total': (a_tot_raw, a_tot_std)
        },
        'kaiser_components': kaiser_components,
        'eigenvalues': eigenvalues.tolist(),
        'variance_ratio': variance_ratio.tolist(),
        'cumulative_variance': cumulative_variance.tolist()
    }
    print("\n--- PHASE 2 & 3: Generation of Charts and Report ---")
    result = results[year]
    plot_scree_cumulative_single(result, OUTPUT_BASE_DIR / "pca_scree_cumulative_2024_refined.png", year)
    plot_biplot_single(result, OUTPUT_BASE_DIR / "pca_biplot_2024_refined.png", year)
    plot_heatmap_single(result, OUTPUT_BASE_DIR / "pca_loadings_2024_refined.png", year)
    plot_means_single(result, OUTPUT_BASE_DIR / "cluster_means_2024_refined.png", year)
    plot_scatter_single(result, OUTPUT_BASE_DIR / "cluster_scatter_2024_refined.png", year)
    plot_map_single(result, OUTPUT_BASE_DIR / "cluster_map_2024_refined.png", PROJECT_DIR, year)

    # Report Generation
    with open(OUTPUT_BASE_DIR / "reliability_summary_2024_refined.txt", "w") as f:
        f.write(f"ANALYSIS SUMMARY FOR {year} (CRIME IN PC1 TEST)\n")
        f.write("====================================================\n\n")
        alphas = result['alphas']
        f.write(f"YEAR: {year}\n")
        f.write("\n--- INTERNAL RELIABILITY (CRONBACH'S ALPHA) ON 5 FACTOR PILLARS ---\n")
        f.write(
            f"  Pillar 1 (Socio-Economic + Crime): Raw={alphas['socio_economic'][0]:.3f} | Std={alphas['socio_economic'][1]:.3f}\n")
        f.write(
            f"  Pillar 2 (Env. Safety):            Raw={alphas['environmental_safety'][0]:.3f} | Std={alphas['environmental_safety'][1]:.3f}\n")
        f.write(
            f"  Pillar 3 (Secondary Ed.):          Raw={alphas['secondary_education'][0]:.3f} | Std={alphas['secondary_education'][1]:.3f}\n")
        f.write(
            f"  Pillar 4 (Primary Ed.):            Raw={alphas['primary_education'][0]:.3f} | Std={alphas['primary_education'][1]:.3f}\n")
        f.write(
            f"  Pillar 5 (Childcare Quality):      Raw={alphas['childcare_quality'][0]:.3f} | Std={alphas['childcare_quality'][1]:.3f}\n")
        f.write(f"  TOTAL (All Vars):                  Raw={alphas['total'][0]:.3f} | Std={alphas['total'][1]:.3f}\n")
        f.write("\n--- PCA COMPONENT SELECTION (CUMULATIVE VARIANCE RULE) ---\n")
        f.write(f"  Components with Eigenvalue > 1.0 (Kaiser): {result['kaiser_components']}\n")
        f.write(f"  Components retained for >80% Variance: {N_COMPONENTS} (Used {N_CLUSTERS} clusters)\n")
        f.write("  Eigenvalue | Variance Explained | Cumulative Variance\n")
        f.write("  --------------------------------------------------\n")
        eigenvalues = result['eigenvalues']
        variance_ratio = result['variance_ratio']
        cumulative_variance = result['cumulative_variance']
        for i, val in enumerate(eigenvalues):
            exp_var = variance_ratio[i]
            cum_var = cumulative_variance[i]
            status = ' (Retained by Kaiser)' if val > 1.0 else ''
            if i < N_COMPONENTS:
                status += ' (RETAINED)'
            f.write(
                f"    PC{i + 1}: {val:.3f}   | {exp_var:.3f} ({exp_var * 100:.1f}%) | {cum_var:.3f} ({cum_var * 100:.1f}%) {status}\n")
        f.write("-" * 30 + "\n")
    print(f"\nAnalysis Complete. All charts saved to: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()