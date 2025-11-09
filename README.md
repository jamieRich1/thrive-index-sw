# Thrive Index SW

> A high-resolution dashboard for exploring neighbourhood-level quality of life and child prosperity across South West England.

This project is a sophisticated Streamlit web application that allows users to explore, compare, and understand neighbourhood conditions. It aggregates multiple public datasets into a single, cohesive "Thrive Score," providing detailed insights at the LSOA (Lower Layer Super Output Area), Ward, and Local Authority levels.

The application is built on a robust data processing pipeline that cleans, geocodes, and analyzes spatial data to find the nearest services (like schools, GPs, and childcare) for every neighbourhood, tracking changes over time.

---

## Key Features

* **Interactive Map Dashboard:** A multi-level Folium map that allows users to drill down from the South West region into Local Authorities, Wards, and finally to individual LSOA neighbourhoods. Features a choropleth overlay to visualize scores and a right-hand panel displaying key indicators for any selected area.
* **Deep Dive Analysis:** A detailed report page for any selected Ward or LSOA, providing:
    * Historical trend charts for the "Thrive Score" and its underlying indicators.
    * Time-series breakdowns for house prices, air quality, and crime rates.
    * Detailed tables of the 3-nearest GPs, childcare providers, and schools (Primary and Secondary), including their performance data and locations.
* **Data Exploration Suite:** A powerful tool for data analysis with four distinct modes:
    * **Rank Areas:** Rank all Wards or LSOAs from best to worst for any single indicator.
    * **Compare Areas:** Select multiple areas for a side-by-side comparison of all metrics.
    * **Explore Correlations:** A scatter plot tool to visualize the relationship between any two indicators (e.g., "Thrive Score" vs. "IMD Decile") with an OLS trendline.
    * **View Distribution:** A histogram plot that shows the distribution of a single indicator across all areas, highlighting where a selected area falls and its percentile rank.
* **Customizable Scoring:** The final "Thrive Score" is calculated using a weighted average. Users can adjust the weights of the six core categories via sliders, allowing them to define what "thriving" means to them.
* **Postcode Search:** A "find-as-you-type" search bar to instantly locate any postcode in the South West and zoom the map directly to its LSOA.

---

## 📊 Core Indicators

The "Thrive Score" is a composite metric built from six key domains, which are themselves derived from over a dozen individual data sources:

* **🌳 Greenspace:** Score based on the percentage of an LSOA's total area that is covered by accessible public greenspace (from OS Open Greenspace).
* **🌬️ Air Quality:** Score based on the annual mean concentrations of Nitrogen Dioxide (NO₂) and PM₂.₅ particulates (from UK-AIR). Lower pollution equals a higher score.
* **🛡️ Community Safety:** Score based on the annual crime rate per 1,000 people. This includes "community impact" crimes like anti-social behaviour, burglary, and public order offenses (from police.uk).
* **🎓 Education:** A combined score based on the performance of the 3-nearest Primary schools (KS2 pass rate, scaled score) and 3-nearest Secondary schools (Progress 8, Attainment 8).
* **🩺 Healthcare:** A combined score based on the average distance to the 3-nearest GP practices and the average patient satisfaction score at those practices.
* **👶 Childcare:** A combined score for the 3-nearest providers, based on average distance, average Ofsted quality rating (1-4), and the total number of registered places.

---

## 🔬 Methodology

The project is divided into two main parts: a data processing pipeline (`scripts/`) that pre-processes all data, and the Streamlit application (`app/`) that reads and visualizes it.

### 1. Data Processing Pipeline (The `scripts/` Directory)

This pipeline is responsible for the complex ETL (Extract, Transform, Load) process that turns dozens of raw files into a single, clean file that powers the app.

**Geospatial Foundation:**
* `build_boundaries_sw.py`: Loads raw UK Local Authority (LAD) and LSOA geometry. It filters these to keep only the LSOAs within the South West region.
* `build_boundaries_ward.py`: Loads the 2025 Ward boundaries and merges them with LAD information.
* `build_area_codes.py`: Creates a master lookup file (`area_codes.csv`) that maps every LSOA (2021) to its corresponding Ward (2025) and LAD (2025), handling spatial joins and "best-fit" logic.

**Indicator-Specific Processing (`process_*.py`):**
* Each `process_*.py` script is a self-contained pipeline for a single indicator (e.g., `process_healthcare_data.py`, `process_primary_school_data.py`).
* **Nearest Neighbour Analysis:** For services like GPs, schools, and childcare, the scripts use a k-d tree spatial index (via `scipy.spatial.cKDTree`). This efficiently calculates the 3-nearest services for the centroid of every LSOA (thousands of them) and retrieves their distances.
* **Time-Series Generation:** The scripts generate annual data for each indicator. They are designed to handle missing data (e.g., DfE performance data during 2020-2021) by explicitly forward-filling and back-filling (`ffill`/`bfill`) the data, ensuring a complete time-series for trend analysis.
* **Dual Output:** Most scripts generate two files:
    * `lsoa_annual_..._scores.parquet`: Time-series data (e.g., average satisfaction of 3-nearest GPs per year).
    * `lsoa_..._details.parquet`: Static data (e.g., the names and URNs of the 3-nearest GPs, based on the latest data).

**Final Master File Generation:**
* `build_master_timeseries.py`: This is the final and most important script. It loads all the individual `lsoa_annual_*.parquet` and `lsoa_*_details.parquet` files and merges them all into a single, master file: `lsoa_annual_indicators.parquet`.
* This master file contains one row for every LSOA, for every year (2018-2025), and is the single data source the Streamlit application loads.

### 2. App Scoring & Aggregation (The `app/` Directory)

The Streamlit app reads the `lsoa_annual_indicators.parquet` file and performs all scoring and visualization in memory, cached via `@st.cache_data`.

**Normalization via Percentile Ranking:**
* The core scoring methodology is defined in `app/utils.py`. To compare different metrics (e.g., crime rate vs. % greenspace), every indicator is normalized using percentile ranking (`.rank(pct=True)`).
* Each LSOA is ranked against all other LSOAs in the South West for that year. This converts every absolute value (e.g., 5.2% greenspace) into a relative score from 0 to 100.
* **Example:** A `greenspace_score` of 95 means that LSOA has more greenspace coverage than 95% of other LSOAs in the South West.

**Custom Weighted Score:**
* The final "Thrive Score" is a weighted average of the six normalized (0-100) indicator scores.
* The weights are taken directly from the user's sliders in the sidebar, allowing for a fully dynamic and personalized composite score.

**Geographic Aggregation:**
* **LSOA Level:** All data is processed and scored at the LSOA level.
* **Ward Level:** All Ward-level scores and indicators are calculated by taking the mean average of the values from all LSOAs within that Ward's boundaries.

---

## 🚀 How to Run Locally

### 1. Prerequisites
* Python 3.9+
* All libraries from `requirements.txt`

### 2. Setup
Clone the repository:

```bash
git clone <repository-url>
cd thrive-index-sw

Create Virtual Environement:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install Dependencies:

```bash
pip install -r requirements.txt
```
### 3. Generate the Data

The processed data files required by the app are not included in this repository. You must run the data processing scripts to generate them.

**Acquire Raw Data:**
You must first download all the raw data files (CSVs, GeoJSON, GPKG) referenced in the `scripts/` directory and place them in the correct subfolders within `data/raw/`. Key files and sources are described in `app/licensing.py`.
* `data/raw/geometry/`: ONS Open Geography boundary files (LADs, LSOAs, Wards).
* `data/raw/lookups/`: ONS lookup files (LSOA to Ward/LAD).
* `data/raw/postcodes/`: Code-Point Open postcode CSVs.
* `data/raw/crime/`: police.uk crime data CSVs.
* ...and so on for `air_quality`, `childcare`, `healthcare`, `housing`, etc.

**Run the Processing Scripts:**
Once the raw data is in place, you must run the scripts in order.

### 1.  Build base geographies

```bash
python scripts/build_boundaries_sw.py
python scripts/build_area_codes.py
python scripts/build_boundaries_ward.py
python scripts/build_postcodes_sw.py
```
    
### 2.  Process all individual indicators

```bash
python scripts/process_air_quality.py
python scripts/process_childcare.py
python scripts/process_crime_data.py
python scripts/process_greenspace.py
python scripts/process_healthcare_data.py
python scripts/process_house_prices.py
python scripts/process_imd.py
python scripts/process_population.py
python scripts/process_primary_school_data.py
python scripts/process_secondary_school_data.py
```
    
### 3.  Build the final master file

```bash
python scripts/build_master_timeseries.py
```

If successful, you will now have a populated `data/processed/` directory, including the critical `lsoa_annual_indicators.parquet` file.

### 4. Run the Streamlit App

```bash
streamlit run app/Thrive_Index_SW.py
```

The application will open in your web browser.

---

## 🎓 Project Context

This application was developed as the final project for a Master's degree in Data Science at UWE Bristol.

---

## ©️ Data Sources & Licensing

This project is built entirely on publicly available data, primarily under the Open Government Licence v3.0. All data sources, their specific attribution text, and links to their homepages are centrally managed in `app/licensing.py` and. dynamically displayed on the "Sources & Licensing" page of the app.

Key data providers include:

* Office for National Statistics (ONS)
* Ordnance Survey (OS)
* police.uk
* GOV.UK (DfE, Ofsted, Defra)
* NHS Digital

The code for this project is licensed under the MIT License. Derived data (composite scores) are provided for informational purposes only and are not official statistics.
