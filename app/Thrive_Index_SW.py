#Imports
import streamlit as st

#Page Config
st.set_page_config(
    page_title="Thrive Index SW",
    page_icon="🗺️",
    layout="wide"
)

st.title("Welcome to the Thrive Index for the South West 🗺️")
st.markdown("---")
st.subheader("Understanding Your Local Area")
st.write(
    "The Thrive Index is a powerful tool designed to provide insights into the quality of life across different neighbourhoods. "
    "By combining key indicators like greenspace access and air quality, it offers a comparative score to help residents, "
    "researchers, and policymakers understand the unique characteristics of each community."
)

#Explaination Container
with st.container(border=True):
    st.subheader("How the 'Thrive Score' is Calculated")
    st.markdown(
        """
        The **Thrive Score** is a composite metric based on data at the **LSOA (Lower Layer Super Output Area)** level—small neighbourhoods of about 1,500 people.

        1.  **Ranking:** For each indicator, every LSOA in the South West is ranked against all others. This percentile rank becomes its score (from 0 to 100). A score of 100 means an LSOA is in the top 1% for that indicator, while a score of 0 means it's in the bottom 1%.
        2.  **Weighting:** You can adjust the importance (or "weight") of each category in the **Map Dashboard** and **Deep Dive** pages. The final Thrive Score is the weighted average of all six indicator scores.
        3.  **Aggregation:** Ward scores are calculated by averaging the scores of all the LSOAs within them.
        """
    )

    st.subheader("What We Measure")
    st.markdown(
        """
        The index combines six key categories to build its score:

        * 🌳 **Greenspace:** The percentage of an area's land that is covered by parks, public gardens, playing fields, and other accessible green spaces.
        * 🌬️ **Air Quality:** Based on the annual mean concentration of NO₂ and PM₂.₅. Lower concentrations result in a better score.
        * 🛡️ **Safety:** Calculated from the annual crime rate per 1,000 people (using ONS population data). A lower crime rate results in a better score.
        * 🎓 **Education:** A combined score for the 3 nearest primary and 3 nearest secondary schools, based on DfE performance data (like KS2 scores, Progress 8, and Attainment 8).
        * 🩺 **Healthcare:** A combined score based on two factors: the average *distance* to the 3 nearest GPs and the *overall patient satisfaction* at those practices.
        * 👶 **Childcare:** A combined score for the 3 nearest providers, based on *distance*, *Ofsted quality rating*, and the *total number of registered places*.
        """
    )

#Explanatory Container for Boundaries
with st.container(border=True):
    st.subheader("A Note on Map Boundaries")
    st.markdown(
        """
        To provide the most intuitive and up-to-date navigation, this app uses the latest **2025 Ward and Local Authority (LAD) boundaries** for the map display. These are the official administrative areas you are familiar with.

        However, the detailed statistical data for indicators is based on the **2021 Lower Layer Super Output Areas (LSOAs)**, which are the building blocks for census and other government data.

        **What does this mean?**
        * Because the LSOA shapes (from 2021) do not always fit perfectly inside the newer Ward shapes (from 2025), you may notice slight visual mismatches or overlaps on the map.
        * This is a standard and expected result of using the official "best-fit" data provided by the Office for National Statistics (ONS).

        **Having trouble finding a specific neighbourhood?**
        Use the **postcode search** feature on the Map Dashboard. It will pinpoint the exact LSOA for any given postcode, bypassing any visual ambiguity.
        """
    )

st.markdown("---")

#2 by 2 grid for nav buttons
col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/01_Map_Dashboard.py", label="Go to the Map Dashboard", icon="🗺️", use_container_width=True)

with col2:
    st.page_link("pages/02_Deep_Dive.py", label="Explore via Deep Dive", icon="📊", use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    st.page_link("pages/03_Data_Exploration.py", label="Data Exploration Tool", icon="📈", use_container_width=True)

with col4:
    st.page_link("pages/04_Sources_&_Licensing.py", label="Sourcing & Licensing", icon="📄", use_container_width=True)