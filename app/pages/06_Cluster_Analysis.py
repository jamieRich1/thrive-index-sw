import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geopandas as gpd

#Page Config
st.set_page_config(
    page_title="Cluster Analysis (Typologies)",
    layout="wide"
)
# Paths and Constants
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
NORMALIZED_DATA_FILE = DATA_DIR / "lsoa_annual_normalized_scores.parquet"
BOUNDARIES_FILE = DATA_DIR / "boundaries_lsoa.geoparquet"
LAD_OUTLINE_FILE = DATA_DIR / "lad_sw_outline.geojson"
TARGET_YEAR = 2024
# Variables for Clustering
VARS_CONFIG = {
    'Socio-Economic': ['income_score', 'employment_score', 'crime_score'],
    'Env. Safety': ['air_quality_no2_score', 'air_quality_pm25_score'],
    'Secondary Ed.': ['secondary_progress_score', 'secondary_attainment_score'],
    'Primary Ed.': ['primary_read_score', 'primary_math_score'],
    'Childcare': ['childcare_quality_score']
}
ALL_VARS = [var for sublist in VARS_CONFIG.values() for var in sublist]

# Data Loading
@st.cache_data
def load_cluster_data():
    if not NORMALIZED_DATA_FILE.exists():
        st.error(f"Data not found at {NORMALIZED_DATA_FILE}")
        return None, None, None
    df = pd.read_parquet(NORMALIZED_DATA_FILE)
    df = df[df['year'] == TARGET_YEAR].set_index('area_code')
    if not BOUNDARIES_FILE.exists():
        st.error(f"Boundaries not found at {BOUNDARIES_FILE}")
        return df, None, None
    gdf = gpd.read_parquet(BOUNDARIES_FILE)
    lad_gdf = None
    if LAD_OUTLINE_FILE.exists():
        lad_gdf = gpd.read_file(LAD_OUTLINE_FILE)
    return df, gdf, lad_gdf


# Helpers
@st.cache_data
def run_tandem_analysis(df, n_clusters, n_components=5):
    """
    Performs Tandem Analysis:
    1. PCA to reduce dimensionality.
    2. K-Means clustering on the Principal Components.
    """
    # Standardize
    scaler = StandardScaler()
    data_std = scaler.fit_transform(df[ALL_VARS])
    # PCA
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(data_std)
    # Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(pca_components)
    df_out = df.copy()
    df_out['Cluster'] = clusters.astype(str)  # String for categorical plotting
    # Calculate Cluster Profiles (Mean Scores of Indicators)
    profiles = df_out.groupby('Cluster')[ALL_VARS].mean()
    return df_out, profiles, pca_components, pca

def generate_relative_descriptions(profiles):
    """Generates descriptions based on RELATIVE performance (Column-wise)."""
    # Aggregate Indicators to Pillars for simpler text
    pillar_scores = pd.DataFrame(index=profiles.index)
    for pillar, vars_list in VARS_CONFIG.items():
        pillar_scores[pillar] = profiles[vars_list].mean(axis=1)
    descriptions = {}
    # Iterate rows (clusters)
    for cluster in pillar_scores.index:
        strengths = []
        weaknesses = []
        # Check each pillar score against the range of that pillar across ALL clusters
        for pillar in pillar_scores.columns:
            val = pillar_scores.loc[cluster, pillar]
            col_min = pillar_scores[pillar].min()
            col_max = pillar_scores[pillar].max()
            if col_max == col_min:
                norm_val = 0.5
            else:
                norm_val = (val - col_min) / (col_max - col_min)
            if norm_val >= 0.66:
                strengths.append(pillar)
            elif norm_val <= 0.33:
                weaknesses.append(pillar)
        desc_text = []
        if strengths: desc_text.append(f"<b>Strengths:</b> {', '.join(strengths)}")
        if weaknesses: desc_text.append(f"<b>Challenges:</b> {', '.join(weaknesses)}")
        if not desc_text: desc_text.append("<b>Profile:</b> Average/Mixed performance.")
        descriptions[cluster] = "<br>".join(desc_text)
    return descriptions

def plot_radar_profiles(profiles):
    """Creates a Radar Chart comparing cluster profiles."""
    categories = list(VARS_CONFIG.keys())
    fig = go.Figure()
    for cluster in profiles.index:
        values = [profiles.loc[cluster, vars_list].mean() for vars_list in VARS_CONFIG.values()]
        values += [values[0]]  # Close loop
        cats = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=cats,
            fill='toself',
            name=f'Cluster {cluster}'
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Cluster Profiles (Average Pillar Scores)",
        height=500
    )
    return fig

def plot_pca_scatter_interactive(pca_data, clusters, pca_model, index_labels):
    """
    Creates an interactive Plotly Scatter plot for PCA.
    Uses .values to ensure alignment and prevent Index Mismatches.
    """
    exp_var = pca_model.explained_variance_ratio_
    plot_df = pd.DataFrame(pca_data[:, :2], columns=['PC1', 'PC2'])
    plot_df['Cluster'] = clusters.values
    plot_df['Area Code'] = index_labels.values
    fig = px.scatter(
        plot_df,
        x='PC1',
        y='PC2',
        color='Cluster',
        hover_data=['Area Code'],
        title=f"Cluster Segmentation (PC1 vs PC2)",
        opacity=0.7,
        template="plotly_white",
        height=600
    )
    fig.update_layout(
        xaxis_title=f"PC1 ({exp_var[0]:.1%} variance)",
        yaxis_title=f"PC2 ({exp_var[1]:.1%} variance)",
        legend_title="Cluster"
    )
    return fig

def plot_static_map_with_context(geo_df, clustered_df, lad_gdf=None):
    """
    Creates a static Matplotlib map with city labels for context.
    This avoids the 500MB size limit crash by rendering an image instead of interactive data.
    """
    # Merge Cluster IDs into Geometry
    map_data = geo_df.merge(clustered_df[['Cluster']], left_on='area_code', right_index=True)
    # Project to British National Grid (EPSG:27700)
    if map_data.crs.to_string() != "EPSG:27700":
        map_data = map_data.to_crs(epsg=27700)
    if lad_gdf is not None and lad_gdf.crs.to_string() != "EPSG:27700":
        lad_gdf = lad_gdf.to_crs(epsg=27700)
    # Setup Plot
    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    # Plot Clusters (LSOAs)
    map_data.plot(
        column='Cluster',
        ax=ax,
        legend=True,
        categorical=True,
        cmap='viridis',
        linewidth=0.0,
        alpha=0.85
    )
    # Overlay LAD Boundaries
    if lad_gdf is not None:
        lad_gdf.plot(
            ax=ax,
            facecolor='none',
            edgecolor='black',
            linewidth=0.6,
            alpha=0.6
        )
    # Add Major Cities for Context
    cities_data = {
        'City': ['Bristol', 'Plymouth', 'Exeter', 'Bournemouth', 'Truro', 'Taunton', 'Gloucester', 'Swindon'],
        'Lon': [-2.5879, -4.1427, -3.5339, -1.8808, -5.0510, -3.1032, -2.2386, -1.7797],
        'Lat': [51.4545, 50.3755, 50.7184, 50.7192, 50.2632, 51.0157, 51.8642, 51.5558]
    }
    cities_df = pd.DataFrame(cities_data)
    cities_gdf = gpd.GeoDataFrame(
        cities_df,
        geometry=gpd.points_from_xy(cities_df.Lon, cities_df.Lat),
        crs="EPSG:4326"
    ).to_crs(epsg=27700)
    # Plot City Markers
    cities_gdf.plot(ax=ax, color='white', edgecolor='black', markersize=60, zorder=10)
    # Add City Names
    for x, y, label in zip(cities_gdf.geometry.x, cities_gdf.geometry.y, cities_gdf.City):
        ax.text(
            x, y + 2000, label,
            fontsize=11,
            ha='center',
            weight='bold',
            color='black',
            path_effects=[pe.withStroke(linewidth=3, foreground="white")]
        )
    ax.set_axis_off()
    ax.set_title(f"Geographic Distribution of Clusters ({TARGET_YEAR})", fontsize=18)
    plt.tight_layout()
    return fig

# Main Page
st.title("Neighborhood Typologies (Cluster Analysis)")
st.markdown("""
This tool uses **Tandem Analysis** (PCA followed by K-Means Clustering) to group neighborhoods into distinct "types".
Unlike rankings, clustering reveals **patterns** (e.g., "High Education but Low Safety").
""")

# Load Data
data_df, geo_df, lad_df = load_cluster_data()
if data_df is not None:

    # Sidebar
    st.sidebar.header("Clustering Controls")
    n_clusters = st.sidebar.select_slider(
        "Number of Clusters (Types)",
        options=[3, 4, 5, 6],
        value=4,
        help="Select 3 for broad groups (e.g. High/Med/Low) or 5-6 for nuanced profiles."
    )
    # Run Analysis
    clustered_df, profiles, pca_data, pca_model = run_tandem_analysis(data_df, n_clusters)
    # Generate Descriptions (Relative Logic)
    descriptions = generate_relative_descriptions(profiles)
    # Interpretation
    st.header(f"Identified {n_clusters} Neighborhood Types")
    # Create rows of 3 columns
    cols_per_row = 3
    cluster_ids = sorted(profiles.index)
    for i in range(0, len(cluster_ids), cols_per_row):
        batch = cluster_ids[i: i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, cluster_id in zip(cols, batch):
            count = len(clustered_df[clustered_df['Cluster'] == cluster_id])
            pct = (count / len(clustered_df)) * 100
            with col:
                st.info(f"**Cluster {cluster_id}**")
                st.metric("Size", f"{count} LSOAs ({pct:.1f}%)")
                st.markdown(descriptions[cluster_id], unsafe_allow_html=True)
    st.markdown("---")

    # Visualisation Tabs
    tab_profiles, tab_map, tab_scatter = st.tabs(["**Profiles (Radar)**", "**Geographic Map**", "**PCA Scatter**"])
    with tab_profiles:
        st.subheader("What defines each cluster?")
        st.plotly_chart(plot_radar_profiles(profiles), width='stretch')
    with tab_map:
        st.subheader("Where are these clusters located?")
        st.markdown("""
        **Cluster Distribution Map:** - **Colors** represent the cluster type.
        - **Black Lines** show Local Authority boundaries.
        - **Labels** show major cities for orientation.
        """)
        if geo_df is not None:
            # Generate the static map with context
            try:
                fig_map = plot_static_map_with_context(geo_df, clustered_df, lad_df)
                st.pyplot(fig_map)
            except Exception as e:
                st.error(f"Error creating map: {e}")
        else:
            st.warning("Boundary data not available. Map cannot be displayed.")
    with tab_scatter:
        st.subheader("Cluster Separation")
        st.markdown("Shows how distinct the clusters are in the statistical space. **Hover to see Area Codes.**")
        fig_scatter = plot_pca_scatter_interactive(
            pca_data,
            clustered_df['Cluster'],
            pca_model,
            clustered_df.index
        )
        st.plotly_chart(fig_scatter, width='stretch')
    # Raw Data Table
    st.markdown("---")
    with st.expander("View Raw Profile Data Table (Heatmap)", expanded=True):
        st.caption("Values represent the average score (0-100) for each indicator within the cluster.")
        st.caption("Green = Relative Strength, Red = Relative Weakness (Column-wise comparison).")
        st.dataframe(
            profiles.style.format("{:.1f}").background_gradient(cmap='RdYlGn', axis=0),
            width='stretch'
        )