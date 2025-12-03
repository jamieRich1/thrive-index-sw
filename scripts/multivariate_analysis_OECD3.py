import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pingouin as pg

# --- CONSTANTS & PATHS ---
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "lsoa_annual_indicators.parquet"
LSOA_BOUNDARIES_FILE = PROJECT_DIR / "data" / "processed" / "boundaries_lsoa.geoparquet"
OUTPUT_DIR = PROJECT_DIR / "data" / "analysis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Plot Style
plt.style.use('seaborn-v0_8-whitegrid')

# --- VARIABLE DEFINITIONS (FINAL MODEL) ---

# Pillar 1: Safety
SAFETY_VARS = [
    'crime_count',
    'no2_mean_concentration',
    'pm25_mean_concentration'
]

# Pillar 2: Opportunity (Final Selection)
# Excluded: avg_gp_satisfaction, avg_ks2_pass_rate (based on reliability testing)
OPPORTUNITY_FINAL = [
    'avg_childcare_quality_score',
    'avg_progress_8',
    'avg_primary_scaled_score',
    'IDACI_Rate'
]

# Combined Analysis List
ALL_VARS = SAFETY_VARS + OPPORTUNITY_FINAL

# Variables requiring inversion (High Value = Negative Outcome)
# These will be inverted so that High Z-Score always equals Good Performance
NEGATIVE_VARS = [
    'crime_count',
    'no2_mean_concentration',
    'pm25_mean_concentration',
    'IDACI_Rate'
]


# --- HELPER FUNCTIONS ---

def calculate_alpha(df):
    """Calculates Standardized Cronbach's Alpha using pingouin."""
    if df.empty or df.shape[1] < 2:
        return 0
    try:
        alpha = pg.cronbach_alpha(data=df)[0]
        return alpha
    except Exception:
        return 0


def get_standardized_data(df, variable_list):
    """Standardizes data (Z-score) and inverts negative indicators."""
    valid_vars = [v for v in variable_list if v in df.columns]
    df_subset = df[valid_vars].dropna()
    if df_subset.empty:
        return pd.DataFrame()

    # Invert negative indicators
    for col in NEGATIVE_VARS:
        if col in df_subset.columns:
            df_subset[col] = -df_subset[col]

    scaler = StandardScaler()
    data_std = scaler.fit_transform(df_subset)
    return pd.DataFrame(data_std, columns=valid_vars, index=df_subset.index)


def plot_variance_analysis(pca, output_dir):
    """Generates Scree Plot and Cumulative Variance Plot."""
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)
    n_components = len(exp_var)
    x_range = range(1, n_components + 1)

    # 1. Scree Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, exp_var, 'bo-', linewidth=2, markersize=8)
    plt.title("Figure 8: Scree Plot: Explained Variance by Component")
    plt.xlabel("Principal Component")
    plt.ylabel("Proportion of Variance Explained")
    plt.xticks(x_range)
    plt.grid(True)
    plt.tight_layout()
    output_scree = output_dir / "pca_scree_plot_final.png"
    plt.savefig(output_scree)
    print(f"-> Chart saved: {output_scree.name}")

    # 2. Cumulative Variance Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, cum_var, 'ro-', linewidth=2, markersize=8)
    plt.fill_between(x_range, cum_var, alpha=0.1, color='red')
    plt.axhline(y=0.65, color='gray', linestyle='--', label='65% Threshold')

    # Annotate first two components
    if n_components >= 2:
        plt.annotate(f'2 PCs: {cum_var[1]:.1%}',
                     xy=(2, cum_var[1]),
                     xytext=(2.5, cum_var[1] - 0.1),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    plt.title("Figure 9: Cumulative Variance Explained")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Variance")
    plt.xticks(x_range)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    output_cum = output_dir / "pca_cumulative_variance_final.png"
    plt.savefig(output_cum)
    print(f"-> Chart saved: {output_cum.name}")


def plot_structural_biplot(pca, df_std, title, output_path):
    """Plots the structural relationship of the variables."""
    loadings = pca.components_.T
    labels = df_std.columns
    plt.figure(figsize=(12, 10))

    for i, label in enumerate(labels):
        x = loadings[i, 0]
        y = loadings[i, 1]

        if label in SAFETY_VARS:
            color = '#e74c3c'  # Professional Red
            marker = 's'
            lbl = 'Safety Indicators'
        else:
            color = '#3498db'  # Professional Blue
            marker = 'o'
            lbl = 'Opportunity Indicators'

        plt.arrow(0, 0, x, y, color=color, alpha=0.5, head_width=0.03, linewidth=1.5)
        plt.text(x * 1.15, y * 1.15, label, color='black', ha='center', va='center', fontsize=9)
        plt.scatter([x], [y], color=color, marker=marker, s=60)

    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} Variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} Variance)")
    plt.grid(True, linestyle='--')
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.title(title)

    legend_elements = [
        Line2D([0], [0], color='#e74c3c', marker='s', lw=0, label='Safety'),
        Line2D([0], [0], color='#3498db', marker='o', lw=0, label='Opportunity')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"-> Chart saved: {output_path.name}")


def plot_cluster_map(cluster_df, k, output_path):
    """Plots the clusters on a map."""
    if not LSOA_BOUNDARIES_FILE.exists():
        print(f"Warning: Boundary file not found at {LSOA_BOUNDARIES_FILE}. Cannot generate map.")
        return

    print(f"Generating Geographic Cluster Map...")
    gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE)

    # Determine merge column
    merge_col = 'area_code' if 'area_code' in gdf.columns else gdf.columns[0]
    gdf_merged = gdf.merge(cluster_df, left_on=merge_col, right_on='area_code', how='inner')

    plt.figure(figsize=(15, 15))
    gdf.plot(ax=plt.gca(), color='#f0f0f0', edgecolor='none')

    gdf_merged.plot(
        column='Cluster',
        ax=plt.gca(),
        categorical=True,
        legend=True,
        cmap='viridis',
        edgecolor='none',
        legend_kwds={'title': f'Cluster Group (k={k})', 'loc': 'upper right'}
    )

    plt.title(f"Figure 11: Geographic Distribution of Clusters (Final Model)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"   -> Map saved: {output_path.name}")


def plot_loadings_heatmap(pca, feature_names, output_path):
    """Visualizes variable loadings as a heatmap."""
    loadings = pca.components_[:4].T
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        loadings,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt='.2f',
        xticklabels=['PC1', 'PC2', 'PC3', 'PC4'],
        yticklabels=feature_names
    )
    plt.title("Figure: Variable Loadings on Principal Components")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"    -> Heatmap saved: {output_path.name}")


def perform_cluster_analysis(df_year, k=3):
    """Runs K-Means clustering on the final dataset."""
    print(f"\n   [Running K-Means Clustering | k={k}]")
    std_data = get_standardized_data(df_year, ALL_VARS)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(std_data)

    profile_df = std_data.copy()
    profile_df['Cluster'] = clusters
    cluster_means = profile_df.groupby('Cluster').mean()

    # Plot Means (Parallel Coordinates)
    plt.figure(figsize=(14, 8))
    colors = sns.color_palette("bright", k)

    for i in range(k):
        plt.plot(cluster_means.columns, cluster_means.values[i], marker='o', linewidth=2,
                 label=f'Cluster {i}', color=colors[i])

    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Standardized Score (Z-Score)")
    plt.title(f"Figure 10: Cluster Profiles (Final Model) - k={k}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_means_final.png")
    print(f"-> Chart saved: cluster_means_final.png")

    # Interpretation Table
    print("\nCluster Interpretation: Pillar Averages")
    print(f"{'Cluster':<8} | {'Safety':<10} | {'Opportunity':<15} | {'Count':<8}")
    print("-" * 55)

    valid_safe = [v for v in SAFETY_VARS if v in std_data.columns]
    valid_opp = [v for v in OPPORTUNITY_FINAL if v in std_data.columns]
    counts = pd.Series(clusters).value_counts().sort_index()

    for i in range(k):
        s_score = cluster_means.loc[i, valid_safe].mean()
        o_score = cluster_means.loc[i, valid_opp].mean()
        print(f"{i:<8} | {s_score:+.3f}      | {o_score:+.3f}          | {counts[i]:<8}")
    print("-" * 55)

    # Map Generation
    cluster_results = pd.DataFrame({
        'area_code': df_year.loc[std_data.index, 'area_code'],
        'Cluster': clusters
    })
    plot_cluster_map(cluster_results, k, OUTPUT_DIR / "cluster_map_final.png")


def print_loadings_matrix(pca, variable_names):
    """Prints Factor Loadings Matrix."""
    n_components = pca.components_.shape[0]
    cols = [f'PC{i + 1}' for i in range(n_components)]

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=cols,
        index=variable_names
    )

    print("\n" + "=" * 50)
    print("FACTOR LOADINGS MATRIX (FINAL MODEL)")
    print("=" * 50)
    print("Interpretation: Values > 0.4 indicate strong relationship.")
    print("-" * 50)
    print(loadings.iloc[:, :3].round(3))
    print("-" * 50)
    return loadings


# --- MAIN EXECUTION ---

def main():
    print("\n" + "=" * 80)
    print("COMPOSITE INDEX ANALYSIS: FINAL MODEL SPECIFICATION")
    print("=" * 80 + "\n")

    if not DATA_FILE.exists():
        print("Error: Data file not found.")
        return

    df = pd.read_parquet(DATA_FILE)
    analysis_year = 2025
    df_year = df[df['year'] == analysis_year].copy()
    print(f"Analysis Reference Year: {analysis_year}")

    # 1. Reliability Analysis
    print("\n[1] RELIABILITY ANALYSIS (Cronbach's Alpha)")
    print("-" * 40)

    # Safety Pillar
    std_safe = get_standardized_data(df_year, SAFETY_VARS)
    alpha_safe = calculate_alpha(std_safe)
    print(f"Safety Pillar Reliability:      {alpha_safe:.3f}")

    # Opportunity Pillar
    std_opp = get_standardized_data(df_year, OPPORTUNITY_FINAL)
    alpha_opp = calculate_alpha(std_opp)
    print(f"Opportunity Pillar Reliability: {alpha_opp:.3f}")

    # Global Reliability
    std_all = get_standardized_data(df_year, ALL_VARS)
    alpha_total = calculate_alpha(std_all)
    print(f"Global Composite Reliability:   {alpha_total:.3f}")

    # 2. PCA Analysis
    print("\n[2] PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("-" * 40)
    pca = PCA()
    pca.fit(std_all)

    print(f"Variance Explained by PC1: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"Variance Explained by PC2: {pca.explained_variance_ratio_[1]:.1%}")
    print(f"Variance Explained by PC3: {pca.explained_variance_ratio_[2]:.1%}")
    if len(pca.explained_variance_ratio_) > 3:
        print(f"Variance Explained by PC4: {pca.explained_variance_ratio_[3]:.1%}")

    print_loadings_matrix(pca, ALL_VARS)
    plot_loadings_heatmap(pca, std_all.columns, OUTPUT_DIR / "pca_loadings_heatmap_final.png")

    # Generate Variance Plots
    plot_variance_analysis(pca, OUTPUT_DIR)

    # Structural Plot
    plot_structural_biplot(
        pca,
        std_all,
        f"Figure 8: Structural Analysis: Final Model ({analysis_year})",
        OUTPUT_DIR / "structural_pca_final.png"
    )

    # 3. Cluster Analysis
    print("\n[3] CLUSTER ANALYSIS (TYPOLOGIES)")
    print("-" * 40)
    perform_cluster_analysis(df_year, k=3)


if __name__ == "__main__":
    main()