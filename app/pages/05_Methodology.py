import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

# Page Config
st.set_page_config(
    page_title="Methodology",
    layout="wide"
)

#Paths and Constants
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
NORMALIZED_DATA_FILE = DATA_DIR / "lsoa_annual_normalized_scores.parquet"
BASELINE_FILE = DATA_DIR / "lsoa_final_composite_scores_2024.parquet"
TARGET_YEAR = 2024
N_COMPONENTS = 5
NUM_SIMULATIONS = 100
UNCERTAINTY_RANGE = 0.20
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
    1: 'RC1: Socio-Economic',
    2: 'RC2: Env. Safety',
    3: 'RC3: Secondary Ed.',
    4: 'RC4: Primary Ed.',
    5: 'RC5: Childcare'
}

# Cached Data Loading
@st.cache_data
def load_analysis_data():
    """Loads normalized data and baseline scores."""
    if not NORMALIZED_DATA_FILE.exists() or not BASELINE_FILE.exists():
        st.error(f"Data files not found. Expected at: {NORMALIZED_DATA_FILE}")
        return None
    # Load Normalized
    df_norm = pd.read_parquet(NORMALIZED_DATA_FILE)
    df_norm = df_norm[df_norm['year'] == TARGET_YEAR].set_index('area_code')
    # Load Baseline
    df_base = pd.read_parquet(BASELINE_FILE)
    df_base = df_base[['area_code', 'Final_CI_Score']].rename(columns={'Final_CI_Score': 'Baseline_Score'})
    df_base = df_base.set_index('area_code')
    # Merge
    df = df_norm.join(df_base, how='inner')
    return df

# Helpers
def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
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

@st.cache_data
def run_pca_analysis(df):
    """Runs PCA and generates detailed weighting tables."""
    # 1. Standardize
    scaler = StandardScaler()
    data_std = scaler.fit_transform(df[ALL_VARS])
    # 2. PCA
    pca = PCA()
    pca.fit(data_std)
    # 3. Loadings & Rotation
    raw_loadings = pca.components_[:N_COMPONENTS].T * np.sqrt(pca.explained_variance_[:N_COMPONENTS])
    rotated_loadings = varimax(raw_loadings)
    # Create DataFrame for Heatmap
    loadings_df = pd.DataFrame(
        rotated_loadings,
        index=ALL_VARS,
        columns=[PILLAR_NAMES[i + 1] for i in range(N_COMPONENTS)]
    )
    # 4. Calculate Dynamic Weights (OECD Method)
    squared_loadings = pd.DataFrame(rotated_loadings ** 2, index=ALL_VARS,
                                    columns=[f'RC{i + 1}' for i in range(N_COMPONENTS)])
    grouping = squared_loadings.idxmax(axis=1)  # Assign to factor
    # Pillar Weights
    rot_var = np.sum(rotated_loadings ** 2, axis=0)
    total_var = np.sum(rot_var)
    fact_weights = rot_var / total_var
    pillar_data = []
    for i in range(N_COMPONENTS):
        rc_key = f'RC{i + 1}'
        pillar_data.append({
            "Pillar": PILLAR_NAMES[i + 1],
            "Variance Explained": rot_var[i],
            "Pillar Weight": fact_weights[i]
        })
    pillar_weights_df = pd.DataFrame(pillar_data)
    # Global Weights
    detailed_data = []
    global_weights = {}
    for indicator in ALL_VARS:
        rc = grouping[indicator]
        rc_idx = int(rc.replace('RC', '')) - 1
        # Level 1 Weight Calculation
        sum_l2 = squared_loadings.loc[grouping == rc, rc].sum()
        l2 = squared_loadings.loc[indicator, rc]
        l1_weight = l2 / sum_l2
        g_weight = l1_weight * fact_weights[rc_idx]
        global_weights[indicator] = g_weight
        detailed_data.append({
            "Pillar": PILLAR_NAMES[rc_idx + 1],
            "Indicator": indicator.replace('_', ' ').title(),
            "Loading (L²)": l2,
            "Within-Pillar Weight": l1_weight,
            "Global Weight": g_weight
        })
    indicator_weights_df = pd.DataFrame(detailed_data)
    return pca, loadings_df, global_weights, pillar_weights_df, indicator_weights_df

@st.cache_data
def run_monte_carlo_simulation(df, base_weights_dict):
    """Runs the Monte Carlo uncertainty analysis."""
    weights_array = np.array([base_weights_dict[c] for c in ALL_VARS])
    rank_storage = pd.DataFrame(index=df.index)
    # Simulation Loop
    np.random.seed(42)
    for i in range(NUM_SIMULATIONS):
        noise = np.random.uniform(1 - UNCERTAINTY_RANGE, 1 + UNCERTAINTY_RANGE, size=len(weights_array))
        sim_weights = weights_array * noise
        sim_weights /= sim_weights.sum()
        scores = (df[ALL_VARS] * sim_weights).sum(axis=1)
        rank_storage[f'sim_{i}'] = scores.rank(ascending=False)
    return rank_storage

def plot_scree(pca):
    fig, ax = plt.subplots(figsize=(8, 4))
    var = pca.explained_variance_ratio_
    cum_var = np.cumsum(var)
    ax.bar(range(1, len(var) + 1), var, alpha=0.5, align='center', label='Individual')
    ax.step(range(1, len(var) + 1), cum_var, where='mid', label='Cumulative')
    ax.axhline(y=0.8, color='r', linestyle='--', linewidth=1, label='80% Threshold')
    ax.set_ylabel('Variance Ratio')
    ax.set_xlabel('Components')
    ax.set_title('Scree Plot (Variance Explained)')
    ax.legend(loc='best')
    return fig

def plot_loadings_heatmap(loadings_df):
    order = loadings_df.abs().idxmax(axis=1).sort_values().index
    sorted_df = loadings_df.loc[order]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(sorted_df, annot=True, cmap='vlag', center=0, fmt='.2f', ax=ax)
    ax.set_title("Rotated Factor Loadings")
    ax.set_ylabel("Indicator")
    ax.set_xlabel("Latent Pillar")
    return fig

def plot_scatter_robustness(df, col_x, col_y, label_x, label_y):
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.scatterplot(x=df[col_x], y=df[col_y], alpha=0.2, s=15, ax=ax)
    max_val = max(df[col_x].max(), df[col_y].max())
    ax.plot([0, max_val], [0, max_val], 'r--', lw=1)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_title(f"{label_y} vs {label_x}")
    return fig

def plot_caterpillar(ranks):
    stats = pd.DataFrame({
        'median': ranks.median(axis=1),
        'p05': ranks.quantile(0.05, axis=1),
        'p95': ranks.quantile(0.95, axis=1)
    }).sort_values('median')
    sample = stats.iloc[::10, :]
    x = range(len(sample))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, sample['median'],
                yerr=[sample['median'] - sample['p05'], sample['p95'] - sample['median']],
                fmt='none', ecolor='gray', alpha=0.3, zorder=1)
    ax.scatter(x, sample['median'], s=3, c='black', zorder=2)
    ax.invert_yaxis()
    ax.set_xlabel("LSOAs (Sorted by Rank)")
    ax.set_ylabel("Rank Range")
    ax.set_title("Uncertainty Analysis 'Caterpillar Plot'")
    return fig

# Main
st.title("Methodology")
st.markdown(
    "This page provides a transparent view into the statistical construction of the **Thrive Index**, following the **OECD Handbook (2008)**.")
# Load
df = load_analysis_data()
if df is not None:
    # Run
    pca, loadings_df, dynamic_weights, pillar_weights_df, detailed_weights_df = run_pca_analysis(df)
    tab_construct, tab_robust = st.tabs(["**Construction (PCA)**", "**Robustness Testing**"])
    # Tab 1 - Construction
    with tab_construct:
        st.header("1. Multivariate Analysis (PCA)")
        st.markdown("""
        Using **Principal Component Analysis (PCA)** to understand the underlying structure of the data. 
        This ensures that the 5 Pillars are statistically distinct and that weights are assigned based on 
        variance explained, not arbitrary choices.
        """)
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Factor Structure")
            st.markdown(
                "The **Heatmap** below shows how each indicator 'loads' onto specific pillars. "
                "High values (red/blue) indicate a strong relationship."
            )
            st.pyplot(plot_loadings_heatmap(loadings_df))
        with col2:
            st.subheader("Variance Explained")
            st.markdown(
                "The **Scree Plot** shows how much information is captured. "
                "The 5-pillar model captures over **80%** of the total variance in the dataset."
            )
            st.pyplot(plot_scree(pca))
        st.markdown("---")
        st.header("2. Derived Statistical Weights")
        st.markdown("""
        Weights are calculated in a two-stage process (OECD Handbook, p. 89).
        1. **Pillar Weight:** Derived from how much variance the Pillar explains (Level 2).
        2. **Indicator Weight:** Derived from how strongly the indicator correlates with the Pillar (Level 1).
        """)
        # Table 1: Pillar Weights
        st.subheader("Level 2: Pillar Weights")
        st.dataframe(
            pillar_weights_df.style.format({
                "Variance Explained": "{:.2f}",
                "Pillar Weight": "{:.2%}"
            }),
            width='stretch',
            hide_index=True
        )
        # Table 2: Detailed Weights Breakdown (Grouped by Pillar)
        st.subheader("Level 1 & Final Global Weights")
        st.caption("Grouped by Pillar. Global Weight = Pillar Weight × Within-Pillar Weight.")
        # Sort by Pillar to group them nicely
        detailed_weights_df = detailed_weights_df.sort_values("Pillar")
        st.dataframe(
            detailed_weights_df.style.format({
                "Loading (L²)": "{:.3f}",
                "Within-Pillar Weight": "{:.2%}",
                "Global Weight": "{:.2%}"
            }),
            width='stretch',
            hide_index=True
        )

    # Tab 2 - Sensativity Analysis
    with tab_robust:
        st.header("3. Robustness & Sensitivity")
        st.markdown("""
        Performing **Sensitivity Analysis** to ensure the rankings are reliable. 
        The charts below are generated *live* from the current dataset.
        """)
        # Equal Weighting
        st.subheader("A. Is the index biased by weights?")
        df['Equal_Score'] = df[ALL_VARS].mean(axis=1)
        df['Rank_Base'] = df['Baseline_Score'].rank(ascending=False)
        df['Rank_Equal'] = df['Equal_Score'].rank(ascending=False)
        corr_weight, _ = spearmanr(df['Rank_Base'], df['Rank_Equal'])
        shift_weight = (df['Rank_Base'] - df['Rank_Equal']).abs().mean()
        col_a1, col_a2 = st.columns([1, 2])
        with col_a1:
            st.metric("Spearman Correlation", f"{corr_weight:.4f}")
            st.metric("Avg Rank Shift", f"{shift_weight:.1f} places")
            st.caption(f"Comparing PCA Weights vs Equal Weights (N={len(df)})")
            st.markdown(
                "**Conclusion:** The index is highly robust to weighting. The diagonal line indicates almost perfect agreement.")
        with col_a2:
            st.pyplot(plot_scatter_robustness(df, 'Rank_Base', 'Rank_Equal', 'Baseline Rank', 'Equal Weights Rank'))
        st.markdown("---")

        # Geometric Aggregation
        st.subheader("B. Does 'Compensability' matter?")
        df['Geom_Score'] = (df[ALL_VARS] + 1).product(axis=1) ** (1 / len(ALL_VARS)) - 1
        df['Rank_Geom'] = df['Geom_Score'].rank(ascending=False)
        corr_agg, _ = spearmanr(df['Rank_Base'], df['Rank_Geom'])
        shift_agg = (df['Rank_Base'] - df['Rank_Geom']).abs().mean()
        col_b1, col_b2 = st.columns([1, 2])
        with col_b1:
            st.metric("Spearman Correlation", f"{corr_agg:.4f}")
            st.metric("Avg Rank Shift", f"{shift_agg:.1f} places")
            st.caption("Comparing Linear vs Geometric Aggregation")
            st.markdown(
                "**Conclusion:** Moderate Sensitivity to Aggregation Change. The large difference on the Y Axis shows how neighbourhoods with inconsistent scores (Low Socio-economic but High Education Scores for example) are punished within geometric aggregation. Whereas neighbourhoods with consistent scores are rewarded. \nThese lack of compensability opportunity does not align with the theoretical framework of the Thrive Score, so the decision to use Linear Aggregation where a low score in one pillar can be offset by a good score in another seems more balanced and fair.")
        with col_b2:
            st.pyplot(plot_scatter_robustness(df, 'Rank_Base', 'Rank_Geom', 'Baseline Rank', 'Geometric Rank'))
        st.markdown("---")
        # Uncertainty Analysis
        st.subheader("C. Monte Carlo Uncertainty Analysis")
        st.markdown(
            f"**{NUM_SIMULATIONS} simulations** were ran, randomly varying the weights by ±{int(UNCERTAINTY_RANGE * 100)}%. "
            "The chart below shows the range of rankings for each LSOA."
        )
        mc_ranks = run_monte_carlo_simulation(df, dynamic_weights)
        st.pyplot(plot_caterpillar(mc_ranks))
        st.markdown(
            "**Conclusion:** Short vertical lines at the left (top ranks) and right (bottom ranks) indicate **high reliability** for the best and worst performing neighborhoods."
            "\nOverall the rank range indicated does not vary too much as moving down the LSOAs median rank order with the rank range staying consistent with the overall trend."
            "\nThe most affected areas seem to be the LSOAs in the middle of the median ranking order.")