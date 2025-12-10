# Imports
import streamlit as st

# Page Config
st.set_page_config(
    page_title="Thrive Index SW",
    layout="wide"
)

# Header
st.title("Welcome to the Thrive Index for the South West")
st.markdown("---")

# Intro
col_intro, col_img = st.columns([2, 1])
with col_intro:
    st.subheader("Measuring Child Prosperity at the Neighbourhood Level")
    st.write(
        "The **Thrive Index** is a sophisticated data tool designed to measure prosperity for children "
        "across South West England. Unlike standard reports that look at large districts, this tool drills down "
        "to the **Neighbourhood (LSOA)** level—areas of approximately 1,500 residents."
    )
    st.write(
        "By aggregating data points, from school exam results to air pollution, it calculates "
        "a statistically robust **'Thrive Score'** for every community, allowing for fair comparison across the region."
    )

# Methodology Container
with st.container(border=True):
    st.subheader("📊 How the 'Thrive Score' is Calculated")
    st.markdown(
        """
        The Thrive Score (0-100) is not a simple average. It is built using a rigorous, academic data pipeline designed to handle the complexity of real-world data:

        1.  **Imputation (MICE):** Using *Multivariate Imputation by Chained Equations* to intelligently fill gaps in historical data on only 0.2% of overall data, ensuring no neighbourhood is penalized for missing records.
        2.  **Normalization:** Data is normalized using a *Winsorized Min-Max* technique. This handles outliers (extreme values) so they don't skew the results, while ensuring all indicators are on a comparable 0-100 scale.
        3.  **Weighting (PCA):** Instead of arbitrary manual weights, *Principal Component Analysis (PCA)* was used. This statistical method determines which indicators are the strongest drivers of variance, assigning weights based on mathematical relationships rather than opinion.
        """
    )

# Pillars Container
with st.container(border=True):
    st.subheader("🧩 The 5 Pillars of the Thrive Score")
    st.markdown("The final Composite Score is derived from five statistically identified pillars:")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown("#### 1. Socio-Economic")
        st.caption("Deprivation & Safety")
        st.markdown("- Income Deprivation Rate\n- Employment Deprivation Rate\n- Community Safety (Crime Rate per 1000 Residents)")

    with c2:
        st.markdown("#### 2. Env. Safety")
        st.caption("Air Quality")
        st.markdown("- NO₂ Concentration\n- PM₂.₅ Concentration")

    with c3:
        st.markdown("#### 3. Secondary Ed.")
        st.caption("Key Stage 4")
        st.markdown("- Progress 8 Scores\n- Attainment 8 Scores")

    with c4:
        st.markdown("#### 4. Primary Ed.")
        st.caption("Key Stage 2")
        st.markdown("- Reading Scores\n- Math Scores")

    with c5:
        st.markdown("#### 5. Childcare")
        st.caption("Early Years Access")
        st.markdown("- Ofsted Quality Ratings")

    st.divider()
    st.markdown("**Additional Contextual Data:**")
    st.markdown(
        "While not included in the mathematical *score* calculation, the dashboard also provides context on "
        "**Greenspace Access**, **GP Satisfaction**, and **House Prices** to paint a complete picture of an area."
    )

# Boundaries Note
with st.container(border=True):
    st.subheader("📍 A Note on Boundaries")
    st.markdown(
        """
        * **Statistical Data** is calculated at the **LSOA (2021)** level for maximum precision.
        * **Map Boundaries** use the latest **May 2025 Wards** for intuitive navigation.

        *Because statistical boundaries do not always align perfectly with political wards, you may see minor visual overlaps. 
        For the most accurate result, use the **Postcode Search** on the dashboard.*
        """
    )

st.markdown("---")

# Navigation Grid
st.subheader("Start Exploring")
col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.page_link("pages/01_Map_Dashboard.py", label="**Map Dashboard**", icon="🗺️")
        st.caption("Drill down from Region to Ward to Neighbourhood.")

with col2:
    with st.container(border=True):
        st.page_link("pages/02_Deep_Dive.py", label="**Deep Dive Analysis**", icon="📊")
        st.caption("View historical trends and detailed service lists for any area.")

col3, col4 = st.columns(2)
with col3:
    with st.container(border=True):
        st.page_link("pages/03_Data_Exploration.py", label="**Data Explorer**", icon="📈")
        st.caption("Rank areas, compare neighbours, and find correlations.")

with col4:
    with st.container(border=True):
        st.page_link("pages/04_Sources_&_Licensing.py", label="**Sources & Licensing**", icon="⚖️")
        st.caption("Transparency on data origins and attribution.")

col5, col6 = st.columns(2)
with col5:
    with st.container(border=True):
        st.page_link("pages/05_Methodology.py", label="**Methodology**", icon="📚")
        st.caption("Explaination of Composite Construction.")

with col6:
    with st.container(border=True):
        st.page_link("pages/06_Cluster_Analysis.py", label="**Cluster Analysis**", icon="🧩")
        st.caption("K-Means Tandem Clustering.")