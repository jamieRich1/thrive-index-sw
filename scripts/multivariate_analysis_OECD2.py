import pandas as pd
import geopandas as gpd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pingouin as pg

#Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "processed" / "lsoa_annual_indicators.parquet"
LSOA_BOUNDARIES_FILE = PROJECT_DIR / "data" / "processed" / "boundaries_lsoa.geoparquet"
OUTPUT_DIR = PROJECT_DIR / "data" / "analysis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#Plot Style
plt.style.use('seaborn-v0_8-whitegrid')

# Pillars
#Safety
SAFETY_VARS = [
    'crime_count',
    'no2_mean_concentration',
    'pm25_mean_concentration'
]

#Opportunity
OPPORTUNITY_REFINED = [
    'avg_gp_satisfaction',
    'avg_childcare_quality_score',
    'avg_progress_8',
    'avg_primary_scaled_score',
    'avg_ks2_pass_rate'
]

#Combined list
ALL_VARS = SAFETY_VARS + OPPORTUNITY_REFINED

#Variables requiring inversion
NEGATIVE_VARS = ['crime_count', 'no2_mean_concentration', 'pm25_mean_concentration']

#Helpers
def calculate_alpha(df):
    """Calculates Standardized Cronbach's Alpha using pingouin."""
    if df.empty or df.shape[1] < 2:
        return 0
    alpha = pg.cronbach_alpha(data=df)[0]
    return alpha


def get_standardized_data(df, variable_list):
    """Standardizes data (Z-score) and inverts negative indicators."""
    valid_vars = [v for v in variable_list if v in df.columns]
    df_subset = df[valid_vars].dropna()
    if df_subset.empty: return pd.DataFrame()
    for col in NEGATIVE_VARS:
        if col in df_subset.columns:
            df_subset[col] = -df_subset[col]
    scaler = StandardScaler()
    data_std = scaler.fit_transform(df_subset)
    return pd.DataFrame(data_std, columns=valid_vars, index=df_subset.index)


def plot_structural_biplot(pca, df_std, title, output_path):
    """Plots the structural relationship of the Refined variables."""
    loadings = pca.components_.T
    labels = df_std.columns
    plt.figure(figsize=(12, 10))
    for i, label in enumerate(labels):
        x = loadings[i, 0]
        y = loadings[i, 1]
        if label in SAFETY_VARS:
            color = 'red'
            marker = 's'
            lbl = 'Safety'
        else:
            color = 'blue'
            marker = 'o'
            lbl = 'Opportunity (Refined)'
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
        Line2D([0], [0], color='red', marker='s', lw=0, label='Safety'),
        Line2D([0], [0], color='blue', marker='o', lw=0, label='Opportunity (Refined)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"-> Chart saved: {output_path.name}")

def plot_cluster_map(cluster_df, k, output_path):
    """
    Plots the clusters on a map of the South West.
    cluster_df should have 'area_code' and 'Cluster' columns.
    """
    if not LSOA_BOUNDARIES_FILE.exists():
        print(f"Warning: Boundary file not found at {LSOA_BOUNDARIES_FILE}. Cannot generate map.")
        return

    print(f"Generating Cluster Map")
    gdf = gpd.read_parquet(LSOA_BOUNDARIES_FILE)

    #Merge Clusters
    gdf_merged = gdf.merge(cluster_df, on='area_code', how='inner')
    plt.figure(figsize=(15, 15))
    #Plot LSOAs
    gdf.plot(ax=plt.gca(), color='#f0f0f0', edgecolor='none')
    #Plot clusters
    gdf_merged.plot(
        column='Cluster',
        ax=plt.gca(),
        categorical=True,
        legend=True,
        cmap='viridis',
        edgecolor='none',
        legend_kwds={'title': f'Cluster (k={k})', 'loc': 'upper right'}
    )

    plt.title(f"Geographic Distribution of Clusters (k={k})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"   -> Map saved: {output_path.name}")

def perform_cluster_analysis(df_year, k=3):
    """
    Runs K-Means on the refined dataset and plots results.
    """
    print(f"\n   [Running K-Means Clustering with k={k}]")
    std_data = get_standardized_data(df_year, ALL_VARS)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(std_data)
    profile_df = std_data.copy()
    profile_df['Cluster'] = clusters
    cluster_means = profile_df.groupby('Cluster').mean()
    #Plot Means
    plt.figure(figsize=(14, 8))
    colors = sns.color_palette("bright", k)
    for i in range(k):
        plt.plot(cluster_means.columns, cluster_means.values[i], marker='o', linewidth=2,
                 label=f'Cluster {i}', color=colors[i])

    plt.axhline(0, color='black', linestyle='--', linewidth=1)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Standardized Score")
    plt.title(f"Cluster Profiles (Refined Variables) - k={k}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cluster_means_refined.png")
    print(f"-> Chart saved: cluster_means_refined.png")

    #Table
    print("\nCluster Interpretation: Refined Pillars")
    print(f"{'Cluster':<8} | {'Safety':<10} | {'Opp (Refined)':<15} | {'Count':<8}")
    print("-" * 55)

    valid_safe = [v for v in SAFETY_VARS if v in std_data.columns]
    valid_opp = [v for v in OPPORTUNITY_REFINED if v in std_data.columns]
    counts = pd.Series(clusters).value_counts().sort_index()
    for i in range(k):
        s_score = cluster_means.loc[i, valid_safe].mean()
        o_score = cluster_means.loc[i, valid_opp].mean()
        print(f"{i:<8} | {s_score:+.3f}     | {o_score:+.3f}          | {counts[i]:<8}")
    print("-" * 55)

    #Map Generation
    cluster_results = pd.DataFrame({
        'area_code': df_year.loc[std_data.index, 'area_code'],
        'Cluster': clusters
    })
    plot_cluster_map(cluster_results, k, OUTPUT_DIR / "cluster_map_refined.png")

#Main
def main():
    print("\n" + "=" * 80)
    print("REFINED ANALYSIS")
    print("=" * 80 + "\n")

    if not DATA_FILE.exists():
        print("Error: Data file not found.")
        return

    df = pd.read_parquet(DATA_FILE)
    analysis_year = 2025
    df_year = df[df['year'] == analysis_year].copy()

    #Cronbach Alpha Section
    print("CRONBACH ALPHA")

    #Safety
    std_safe = get_standardized_data(df_year, SAFETY_VARS)
    alpha_safe = calculate_alpha(std_safe)
    print(f"\nSafety Pillar")
    print(f"- Alpha: {alpha_safe:.3f}")

    #Opportunity
    std_opp = get_standardized_data(df_year, OPPORTUNITY_REFINED)
    alpha_opp = calculate_alpha(std_opp)
    print(f"\nOpportunity Pillar (Refined)")
    print(f"- Variables: {', '.join(OPPORTUNITY_REFINED)}")
    print(f"- Alpha: {alpha_opp:.3f}")

    #PCA Section
    print("\nPCA ANALYSIS")
    std_all = get_standardized_data(df_year, ALL_VARS)
    pca = PCA()
    pca.fit(std_all)
    print(f"- PC1 Variance: {pca.explained_variance_ratio_[0]:.1%}")
    print(f"- PC2 Variance: {pca.explained_variance_ratio_[1]:.1%}")

    plot_structural_biplot(
        pca,
        std_all,
        f"Structural Analysis: Refined Pillars ({analysis_year})",
        OUTPUT_DIR / "structural_pca_refined.png"
    )

    #Overall Reliability
    print("\n" + "-" * 40)
    print("OVERALL COMPOSITE RELIABILITY")
    alpha_total = calculate_alpha(std_all)
    print(f"- Global Cronbach's Alpha: {alpha_total:.3f}")

    #Cluster Analysis
    print("\n" + "-" * 40)
    print("CLUSTER ANALYSIS")
    perform_cluster_analysis(df_year, k=3)

if __name__ == "__main__":
    main()