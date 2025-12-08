import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
NORMALIZED_DATA_FILE = PROJECT_DIR / "data" / "processed" / "lsoa_annual_normalized_scores.parquet"
BASELINE_FILE = PROJECT_DIR / "data" / "processed" / "lsoa_final_composite_scores_2024.parquet"
OUTPUT_DIR = PROJECT_DIR / "data" / "analysis_results_robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_YEAR = 2024
NUM_SIMULATIONS = 100
UNCERTAINTY_RANGE = 0.20
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

# Weighting Mirror Thrive Score
def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    """Performs Varimax (orthogonal) rotation. Copied from weighting script."""
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

def calculate_exact_oecd_weights(df, cols):
    """
    Re-runs the exact PCA/Factor Analysis logic to derive the baseline weights dynamically.
    Returns: A numpy array of 'Global Weights' (Factor Weight * Indicator Weight) summing to 1.
    """
    print("  -> Re-calculating PCA weights from source data...")
    # Standardize
    scaler = StandardScaler()
    data_std = scaler.fit_transform(df[cols])
    # PCA
    pca = PCA()
    pca.fit(data_std)
    # Loadings & Rotation
    raw_loadings = pca.components_[:N_COMPONENTS].T * np.sqrt(pca.explained_variance_[:N_COMPONENTS])
    rotated_loadings = varimax(raw_loadings)
    loading_df = pd.DataFrame(rotated_loadings, index=cols, columns=[f'RC{x + 1}' for x in range(N_COMPONENTS)])
    # Weight Calculation
    squared_loadings = loading_df ** 2
    # Identify Factor Assignment (highest loading)
    grouping = loading_df.abs().idxmax(axis=1)
    # Calculate Factor Weights (Variance Explained)
    rotated_variance = np.sum(rotated_loadings ** 2, axis=0)
    total_variance = np.sum(rotated_variance)
    factor_weights_map = rotated_variance / total_variance
    # Calculate Indicator Weights (Level 1)
    global_weights = pd.Series(0.0, index=cols)
    for rc_idx in range(N_COMPONENTS):
        rc_name = f'RC{rc_idx + 1}'
        indicators_in_rc = grouping[grouping == rc_name].index
        if indicators_in_rc.empty: continue
        # Sum of squared loadings for this specific factor (denominator)
        sum_l2 = squared_loadings.loc[indicators_in_rc, rc_name].sum()
        # Factor Weight (Level 2)
        w_factor = factor_weights_map[rc_idx]
        for indicator in indicators_in_rc:
            l2 = squared_loadings.loc[indicator, rc_name]
            # Level 1 Weight (Indicator Importance within Pillar)
            w_indicator = l2 / sum_l2
            global_weights[indicator] = w_factor * w_indicator
    return global_weights.values

# Data Loading
def load_data():
    if not NORMALIZED_DATA_FILE.exists() or not BASELINE_FILE.exists():
        print("Error: Input files not found.")
        exit()
    # Load Normalized Indicators
    df_norm = pd.read_parquet(NORMALIZED_DATA_FILE)
    df_norm = df_norm[df_norm['year'] == TARGET_YEAR].set_index('area_code')
    # Load Baseline Score
    df_base = pd.read_parquet(BASELINE_FILE)
    df_base = df_base[['area_code', 'Final_CI_Score']].rename(columns={'Final_CI_Score': 'Baseline_Score'})
    df_base = df_base.set_index('area_code')
    # Join
    df = df_norm.join(df_base, how='inner')
    return df

# Sensativity Analysis
def run_robustness_scenarios(df):
    print(f"\nPART 1: ROBUSTNESS SCENARIOS ({TARGET_YEAR})")
    # Scenario A: Equal Weights
    df['Scenario_A_Equal'] = df[ALL_VARS].mean(axis=1)
    # Scenario B: Geometric Aggregation (add 1 to avoid zero issues, then sub 1)
    df['Scenario_B_Geo'] = (df[ALL_VARS] + 1).product(axis=1) ** (1 / len(ALL_VARS)) - 1
    # Rank Comparisons
    df['Rank_Baseline'] = df['Baseline_Score'].rank(ascending=False)
    df['Rank_Equal'] = df['Scenario_A_Equal'].rank(ascending=False)
    df['Rank_Geo'] = df['Scenario_B_Geo'].rank(ascending=False)
    # Shifts
    df['Shift_Baseline_vs_Equal'] = (df['Rank_Baseline'] - df['Rank_Equal']).abs()
    df['Shift_Baseline_vs_Geo'] = (df['Rank_Baseline'] - df['Rank_Geo']).abs()
    # Stats
    comparisons = [
        ('Figure 16: Baseline vs Equal Weights', 'Baseline_Score', 'Scenario_A_Equal', 'Rank_Baseline', 'Rank_Equal',
         'Shift_Baseline_vs_Equal'),
        ('Figure 17: Baseline vs Geometric Agg', 'Baseline_Score', 'Scenario_B_Geo', 'Rank_Baseline', 'Rank_Geo',
         'Shift_Baseline_vs_Geo')
    ]
    print(f"{'Comparison':<30} | {'Pearson R':<10} | {'Spearman Rho':<12} | {'Avg Rank Shift':<15}")
    print("-" * 75)
    for label, s1, s2, r1, r2, shift in comparisons:
        p_corr, _ = pearsonr(df[s1], df[s2])
        s_corr, _ = spearmanr(df[r1], df[r2])
        avg_shift = df[shift].mean()
        print(f"{label:<30} | {p_corr:.4f}     | {s_corr:.4f}       | {avg_shift:.1f} places")
        # Scatter Plot
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=df[r1], y=df[r2], alpha=0.3, s=10)
        max_rank = max(df[r1].max(), df[r2].max())
        plt.plot([0, max_rank], [0, max_rank], 'r--', lw=1, label="Perfect Match")
        plt.title(f"{label}\nAvg Rank Shift: {avg_shift:.1f}")
        plt.xlabel(f"{r1}")
        plt.ylabel(f"{r2}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        filename = label.replace(" ", "_").lower() + ".png"
        plt.savefig(OUTPUT_DIR / filename)
        plt.close()

# Uncertainty Analysis
def run_monte_carlo(df):
    print(f"\nPART 2: MONTE CARLO UNCERTAINTY ANALYSIS (N={NUM_SIMULATIONS})")
    # Get the weights
    base_weights_array = calculate_exact_oecd_weights(df, ALL_VARS)
    print("\nBaseline Weights (Dynamically Calculated):")
    for name, w in zip(ALL_VARS, base_weights_array):
        print(f"  {name:<30}: {w:.4f}")
    rank_storage = pd.DataFrame(index=df.index)
    # Run Simulations
    for i in range(NUM_SIMULATIONS):
        # Generate random noise
        random_noise = np.random.uniform(
            1 - UNCERTAINTY_RANGE,
            1 + UNCERTAINTY_RANGE,
            size=len(base_weights_array)
        )
        # Perturb the weights
        sim_weights = base_weights_array * random_noise
        sim_weights /= sim_weights.sum()
        # Score and Rank
        scores = (df[ALL_VARS] * sim_weights).sum(axis=1)
        rank_storage[f'sim_{i}'] = scores.rank(ascending=False)
    return rank_storage

def plot_caterpillar(ranks):
    print("Generating Caterpillar Plot...")
    stats = pd.DataFrame()
    stats['median_rank'] = ranks.median(axis=1)
    stats['p05_rank'] = ranks.quantile(0.05, axis=1)
    stats['p95_rank'] = ranks.quantile(0.95, axis=1)
    stats = stats.sort_values('median_rank')
    # Sample every 10th point for cleaner plotting
    sample = stats.iloc[::10, :]
    x = range(len(sample))
    plt.figure(figsize=(12, 8))
    # Error bars
    lower = sample['median_rank'] - sample['p05_rank']
    upper = sample['p95_rank'] - sample['median_rank']
    plt.errorbar(
        x, sample['median_rank'], yerr=[lower, upper],
        fmt='none', ecolor='gray', alpha=0.3, elinewidth=1, zorder=1
    )
    plt.scatter(x, sample['median_rank'], s=2, c='black', zorder=2, label='Median Rank')
    plt.title(
        f'Figure 18: Uncertainty Analysis (Caterpillar Plot)\n{NUM_SIMULATIONS} Simulations varying Dynamic PCA Weights by +/- {int(UNCERTAINTY_RANGE * 100)}%')
    plt.xlabel('LSOAs (Sorted by Median Rank)')
    plt.ylabel('Rank Range (5th - 95th Percentile)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "dynamic_uncertainty_caterpillar.png")
    print(f"Plots saved to {OUTPUT_DIR}")

# Main
if __name__ == "__main__":
    df = load_data()
    # Run Part 1 (Scenarios)
    run_robustness_scenarios(df)
    # Run Part 2 (Monte Carlo with Dynamic Weights)
    mc_ranks = run_monte_carlo(df)
    plot_caterpillar(mc_ranks)