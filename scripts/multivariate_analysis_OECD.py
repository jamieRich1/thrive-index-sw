import pandas as pd
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
OUTPUT_DIR = PROJECT_DIR / "data" / "analysis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

#Plot Style
plt.style.use('seaborn-v0_8-whitegrid')

#Conceptual Pillars
#Safety
SAFETY_VARS = [
    'crime_count',
    'no2_mean_concentration',
    'pm25_mean_concentration'
]
#Opportunity
OPPORTUNITY_FULL = [
    'avg_primary_scaled_score',
    'avg_ks2_pass_rate',
    'avg_progress_8',
    'avg_attainment_8',
    'avg_gp_satisfaction',
    'avg_childcare_quality_score',
    'greenspace_percentage'
]
#Combined List
ALL_VARS = SAFETY_VARS + OPPORTUNITY_FULL
#Variables requiring inversion
NEGATIVE_VARS = ['crime_count', 'no2_mean_concentration', 'pm25_mean_concentration']

#Helpers
def calculate_alpha(df):
    """
    Calculates Standardized Cronbach's Alpha using pingouin.
    Methodology: Run this on specific sub-pillars to test internal consistency.
    """
    if df.empty: return 0
    try:
        alpha = pg.cronbach_alpha(data=df)[0]
        return alpha
    except Exception:
        return 0

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
    """
    Plots the structural relationship of ALL variables.
    Color codes based on theoretical pillars to visualize alignment.
    """
    loadings = pca.components_.T
    labels = df_std.columns
    plt.figure(figsize=(12, 10))
    #Plot Vectors
    for i, label in enumerate(labels):
        x = loadings[i, 0]
        y = loadings[i, 1]

        #Colour Code Pillars
        if label in SAFETY_VARS:
            color = 'red'
            marker = 's'
        else:
            color = 'blue'
            marker = 'o'

        plt.arrow(0, 0, x, y, color=color, alpha=0.5, head_width=0.03, linewidth=1.5)
        plt.text(x * 1.15, y * 1.15, label, color='black', ha='center', va='center', fontsize=9)
        plt.scatter([x], [y], color=color, marker=marker, s=50)

    #Context
    plt.xlim(-1.0, 1.0)
    plt.ylim(-1.0, 1.0)
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} Variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} Variance)")
    plt.grid(True, linestyle='--')
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    plt.title(title)

    #Legend
    legend_elements = [
        Line2D([0], [0], color='red', marker='s', lw=0, label='Theoretical: Safety'),
        Line2D([0], [0], color='blue', marker='o', lw=0, label='Theoretical: Opportunity')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"   -> Chart saved: {output_path.name}")

def plot_cluster_means(cluster_centers, variable_names, title, output_path):
    """
    Generates a Means Plot (Parallel Coordinates) to visualize cluster profiles.
    Reference: Figure 13 in OECD Handbook.
    """
    n_clusters = len(cluster_centers)
    plt.figure(figsize=(14, 8))
    colors = sns.color_palette("bright", n_clusters)

    #Plot Line for each cluster
    for i in range(n_clusters):
        plt.plot(variable_names, cluster_centers[i], marker='o', linewidth=2,
                 label=f'Cluster {i}', color=colors[i])

    plt.axhline(0, color='black', linestyle='--', linewidth=1, label='Average (Z=0)')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Standardized Score (Z-Score)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"   -> Chart saved: {output_path.name}")

def perform_cluster_analysis(df_year, k=4):
    """
    Runs K-Means clustering to identify distinct profiles (e.g., High Opp/Low Safety).
    k=4 is chosen to potentially capture the 4 quadrants of performance.
    """
    print(f"\nRunning K-Means Clustering with k={k}")

    #Standardise
    std_data = get_standardized_data(df_year, ALL_VARS)
    #K-Means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(std_data)
    profile_df = std_data.copy()
    profile_df['Cluster'] = clusters
    #Calculate centroids
    cluster_means = profile_df.groupby('Cluster').mean()
    #Generate Plot
    plot_cluster_means(
        cluster_means.values,
        cluster_means.columns,
        f"Cluster Profiles (Means Plot) - k={k}",
        OUTPUT_DIR / "cluster_means_plot.png"
    )
    #Interpration Info
    valid_safety = [v for v in SAFETY_VARS if v in std_data.columns]
    valid_opp = [v for v in OPPORTUNITY_FULL if v in std_data.columns]

    print("\nCluster Interpretation: Pillar Averages (Z-Scores)")
    print("(Positive = Good Performance, Negative = Poor Performance)")
    print(f"{'Cluster':<10} | {'Safety Score':<15} | {'Opportunity Score':<20} | {'Count':<10}")
    print("-" * 70)

    counts = pd.Series(clusters).value_counts().sort_index()
    for i in range(k):
        # Average Z-score for Safety variables in this cluster
        safe_score = cluster_means.loc[i, valid_safety].mean()
        # Average Z-score for Opportunity variables in this cluster
        opp_score = cluster_means.loc[i, valid_opp].mean()
        count = counts[i]
        print(f"{i:<10} | {safe_score:+.3f}          | {opp_score:+.3f}               | {count:<10}")
    print("-" * 70)

#Main
def main():
    print("\n" + "=" * 80)
    print("METHODOLOGICAL WORKFLOW: PCA (Structure), ALPHA (Consistency) & CLUSTERING")
    print("=" * 80 + "\n")

    if not DATA_FILE.exists():
        print("Error: Data file not found.")
        return

    df = pd.read_parquet(DATA_FILE)
    analysis_year = 2025
    df_year = df[df['year'] == analysis_year].copy()
    print(f"Analysis Reference Year: {analysis_year}\n")

    #PCA Section
    print("PCA ANALYSIS")

    #Standardize all variables together
    std_all = get_standardized_data(df_year, ALL_VARS)

    #PCA
    pca = PCA()
    pca.fit(std_all)
    exp_var = pca.explained_variance_ratio_
    print("\nPCA Variance Explained")
    for i, var in enumerate(exp_var[:5]):
        print(f"- PC{i + 1}: {var:.1%}")
    print(f"- Cumulative (First 2): {(exp_var[0] + exp_var[1]):.1%}")

    plot_structural_biplot(
        pca,
        std_all,
        f"Structural Analysis: Safety vs Opportunity ({analysis_year})",
        OUTPUT_DIR / "structural_pca_biplot.png"
    )

    #Cronbach Alpha Section
    print("\n" + "-" * 40)
    print("CRONBACH ALPHA")
    #Safety
    std_safety = get_standardized_data(df_year, SAFETY_VARS)
    alpha_safe = calculate_alpha(std_safety)
    print(f"\nSafety Pillar Reliability")
    print(f"- Cronbach's Alpha: {alpha_safe:.3f}")
    #Opportunity
    std_opp = get_standardized_data(df_year, OPPORTUNITY_FULL)
    alpha_opp = calculate_alpha(std_opp)
    print(f"\n[Opportunity Pillar] Reliability")
    print(f"- Cronbach's Alpha: {alpha_opp:.3f}")
    #Sensitivity Check
    print("\nOpportunity Sensitivity Analysis")
    #Without Greenspace
    vars_no_green = [v for v in OPPORTUNITY_FULL if v != 'greenspace_percentage']
    alpha_no_green = calculate_alpha(get_standardized_data(df_year, vars_no_green))
    #Without Attainment 8
    vars_no_att = [v for v in OPPORTUNITY_FULL if v != 'avg_attainment_8']
    alpha_no_att = calculate_alpha(get_standardized_data(df_year, vars_no_att))
    print(f"- Baseline:          {alpha_opp:.3f}")
    print(f"- Without Greenspace: {alpha_no_green:.3f} (Change: {alpha_no_green - alpha_opp:+.3f})")
    print(f"- Without Attainment: {alpha_no_att:.3f} (Change: {alpha_no_att - alpha_opp:+.3f})")
    #Overall Reliability
    print("\n" + "-" * 40)
    print("OVERALL COMPOSITE RELIABILITY")
    alpha_total = calculate_alpha(std_all)
    print(f"Global Cronbach's Alpha: {alpha_total:.3f}")
    #Cluster Section (K-Means)
    print("\n" + "-" * 40)
    print("CLUSTER ANALYSIS")
    perform_cluster_analysis(df_year, k=4)

if __name__ == "__main__":
    main()