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
OUTPUT_BASE_DIR = PROJECT_DIR / "data" / "analysis_results_2024_baseline"
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
plt.style.use('seaborn-v0_8-whitegrid')
YEARS_TO_ANALYZE = [2024]
TARGET_YEAR = YEARS_TO_ANALYZE[0]

# Safety
SAFETY_VARS = [
    'crime_rate_per_1000',
    'no2_mean_concentration',
    'pm25_mean_concentration'
]
# Opportunity
OPPORTUNITY_VARS = [
    'primary_read_score_weighted',
    'primary_math_score_weighted',
    'secondary_progress_8_weighted',
    'secondary_attainment_8_weighted',
    'avg_gp_satisfaction',
    'avg_childcare_quality_score',
    'greenspace_percentage',
]
ECONOMIC_VARS = [
    'Income_Rate',
    'Employment_Rate',
]
ALL_VARS = SAFETY_VARS + OPPORTUNITY_VARS + ECONOMIC_VARS
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
    """Derives the crime rate per 1000 people."""
    if 'crime_count' in df.columns and 'population' in df.columns:
        safe_population = df['population'].replace(0, 1).fillna(1)
        df['crime_rate_per_1000'] = (df['crime_count'] / safe_population) * 1000
    else:
        df['crime_rate_per_1000'] = df['crime_count']
    return df

def plot_scree_cumulative_single(result, output_path, year):
    """Plots the Scree Plot and Cumulative Variance Plot side-by-side."""
    pca = result['pca']
    variances = pca.explained_variance_ratio_
    cum_variances = variances.cumsum()
    x_indices = range(len(variances))
    fig, axes = plt.subplots(1, 2, figsize=(18, 6)) # Create a figure with 1 row and 2 columns

    #Left Plot: Scree Plot (Explained Variance)
    ax1 = axes[0]
    ax1.plot(x_indices, variances, marker='o', linestyle='-', color='blue')
    ax1.set_title(f"Scree Plot - Explained Variance by PC ({year})", fontsize=14)
    ax1.set_xlabel("Principal Component Index", fontsize=12)
    ax1.set_ylabel("Variance Explained", fontsize=12)
    ax1.set_xticks(x_indices, [f"PC{j + 1}" for j in x_indices])
    ax1.grid(True, linestyle='--')

    #Right Plot: Cumulative Variance Plot
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

    fig.suptitle(f"Figure 3: PCA Component Selection Plots ({year})", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    plt.close()

def plot_biplot_single(result, output_path, year):
    """Plots a single Structural Biplot for the target year."""
    pca = result['pca']
    std_df = result['std_df']
    loadings = pca.components_.T
    labels = std_df.columns
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.suptitle("Figure 6: Structural Biplot: Safety vs Opportunity Vectors", fontsize=16, fontweight='bold')
    ax.set_title(
        f"{year} (PC1: {pca.explained_variance_ratio_[0]:.1%} | PC2: {pca.explained_variance_ratio_[1]:.1%})",
        fontsize=12)
    # Plot Vectors
    for j, label in enumerate(labels):
        x, y = loadings[j, 0], loadings[j, 1]
        if label in SAFETY_VARS:
            color = 'red'
            marker = 's'
        elif label in OPPORTUNITY_VARS:
            color = 'blue'
            marker = 'o'
        else:
            color = 'green'
            marker = 'x'
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
    # Legend
    legend_elements = [
        Line2D([0], [0], color='red', marker='s', lw=0, label='Safety'),
        Line2D([0], [0], color='blue', marker='o', lw=0, label='Opportunity'),
        Line2D([0], [0], color='green', marker='s', lw=0, label='Economic'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
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
        yticklabels=[c.replace('_', ' ').title() for c in std_df.columns],
        cbar_kws={'label': 'Loading Score'}
    )
    plt.title(f"Figure 4: Variable Loadings on First {loadings.shape[1]} Components ({year})", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_means_single(result, output_path, year):
    """Plots a single Cluster Means (Profile) Plot for the target year."""
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
    plt.title(f"Figure 8: Cluster Profiles: Standardized Means ({year})", fontsize=16)
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
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='PC1', y='PC2', hue='Cluster', data=plot_df,
        palette='bright', s=40
    )
    plt.title(f"Cluster Segmentation in PC Space ({year})", fontsize=16)
    plt.xlabel(f"PC1 ({exp_var[0]:.1%})", fontsize=12)
    plt.ylabel(f"PC2 ({exp_var[1]:.1%})", fontsize=12)
    plt.legend(title='Cluster ID')
    plt.grid(True)
    plt.axhline(0, color='black', lw=0.5);
    plt.axvline(0, color='black', lw=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_map_single(result, output_path, project_dir, year):
    """Plots a single Choropleth Map for the target year."""
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
    ax.set_title(f"Figure 9: Geographic Cluster Distribution ({year})", fontsize=16, fontweight='bold')
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

    pca = PCA()
    pca.fit(std_all)

    #Kaiser Criterion
    eigenvalues = pca.explained_variance_
    kaiser_components = np.sum(eigenvalues > 1.0)

    #Calculate Variance Explained and Cumulative Variance Explained
    variance_ratio = pca.explained_variance_ratio_  # Individual variance explained
    cumulative_variance = np.cumsum(variance_ratio)  # Cumulative variance explained

    # Reliability
    a_safe_raw, a_safe_std = calculate_cronbach_robust(df_year, SAFETY_VARS)
    a_opp_raw, a_opp_std = calculate_cronbach_robust(df_year, OPPORTUNITY_VARS)
    a_ec_raw, a_ec_std = calculate_cronbach_robust(df_year, ECONOMIC_VARS)
    a_tot_raw, a_tot_std = calculate_cronbach_robust(df_year, ALL_VARS)

    # Clustering (Tandem: PCA -> KMeans)
    # Using 4 Components and 4 Clusters as default
    pca_data = pca.transform(std_all)
    pca_input = pca_data[:, :4]
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pca_input)

    results[year] = {
        'std_df': std_all,
        'pca': pca,
        'pca_data': pca_data,
        'clusters': clusters,
        'cluster_series': pd.Series(clusters, index=df_year.index),
        'alphas': {
            'safety': (a_safe_raw, a_safe_std),
            'opportunity': (a_opp_raw, a_opp_std),
            'economic': (a_ec_raw, a_ec_std),
            'total': (a_tot_raw, a_tot_std)
        },
        'kaiser_components': kaiser_components,
        'eigenvalues': eigenvalues.tolist(),
        'variance_ratio': variance_ratio.tolist(),
        'cumulative_variance': cumulative_variance.tolist()
    }

    #Plotting Phase
    print("\n--- PHASE 2: Generation of Single Charts ---")
    result = results[year]
    print("Generating Scree and Cumulative Variance Plot (Combined)...")
    plot_scree_cumulative_single(result, OUTPUT_BASE_DIR / "pca_scree_cumulative_2024.png", year)
    print("Generating Biplot...")
    plot_biplot_single(result, OUTPUT_BASE_DIR / "pca_biplot_2024.png", year)
    print("Generating Loadings Heatmap...")
    plot_heatmap_single(result, OUTPUT_BASE_DIR / "pca_loadings_2024.png", year)
    print("Generating Cluster Means (Profile) Plot...")
    plot_means_single(result, OUTPUT_BASE_DIR / "cluster_means_2024.png", year)
    print("Generating Cluster Scatter Plot...")
    plot_scatter_single(result, OUTPUT_BASE_DIR / "cluster_scatter_2024.png", year)
    print("Generating Geographic Map...")
    plot_map_single(result, OUTPUT_BASE_DIR / "cluster_map_2024.png", PROJECT_DIR, year)

    #Report Generation
    print("\n--- PHASE 3: Reliability and PCA Report ---")
    with open(OUTPUT_BASE_DIR / "reliability_summary_2024.txt", "w") as f:
        f.write(f"ANALYSIS SUMMARY FOR {year}\n")
        f.write("=====================================\n\n")
        alphas = result['alphas']
        f.write(f"YEAR: {year}\n")
        f.write("\n--- INTERNAL RELIABILITY (CRONBACH'S ALPHA) ---\n")
        f.write(f"  Safety:      Raw={alphas['safety'][0]:.3f} | Std={alphas['safety'][1]:.3f}\n")
        f.write(f"  Opportunity: Raw={alphas['opportunity'][0]:.3f} | Std={alphas['opportunity'][1]:.3f}\n")
        f.write(f"  Economic: Raw={alphas['economic'][0]:.3f} | Std={alphas['economic'][1]:.3f}\n")
        f.write(f"  TOTAL:       Raw={alphas['total'][0]:.3f} | Std={alphas['total'][1]:.3f}\n")
        f.write("\n--- PCA COMPONENT SELECTION (KAISER CRITERION) ---\n")
        f.write(f"  Components with Eigenvalue > 1.0: {result['kaiser_components']}\n")
        f.write("  Eigenvalue | Variance Explained | Cumulative Variance\n")
        f.write("  --------------------------------------------------\n")

        eigenvalues = result['eigenvalues']
        variance_ratio = result['variance_ratio']
        cumulative_variance = result['cumulative_variance']

        for i, val in enumerate(eigenvalues):
            exp_var = variance_ratio[i]
            cum_var = cumulative_variance[i]
            status = ' (Retain)' if val > 1.0 else ''
            f.write(
                f"    PC{i + 1}: {val:.3f}   | {exp_var:.3f} ({exp_var * 100:.1f}%) | {cum_var:.3f} ({cum_var * 100:.1f}%) {status}\n")
        f.write("-" * 30 + "\n")
    print(f"\nAnalysis Complete. All single-year charts saved to: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()