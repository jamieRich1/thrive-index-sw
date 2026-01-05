# Thrive Index SW

The Dashboard can be accessed on [StreamLit Community Cloud](https://thrive-index-sw-cbe2s5vnzxxp47nh87kbyg.streamlit.app/).

**A high-resolution data intelligence tool for measuring child prosperity at the neighbourhood level across South West England.**

The **Thrive Index SW** is a sophisticated dashboard designed to move beyond broad regional statistics. By aggregating millions of data points into a single, cohesive **"Thrive Score,"** it provides granular insights at the LSOA (*Lower Layer Super Output Area*), Ward, and Local Authority levels.

Unlike standard indices that often rely on simple averages, this project employs advanced data science techniques—including multiple imputation and principal component analysis—to ensure that every score is statistically robust, fair, and comparable.

---

## Project Aims

* **Granularity:** To provide insight at the "hyper-local" level (approx. 1,500 residents), revealing pockets of deprivation or prosperity that are often masked by Ward or District averages.
* **Robustness:** To build a composite indicator that handles missing data intelligently, ensuring no community is penalized for gaps in historical records.
* **Objectivity:** To use statistical weighting (PCA) rather than arbitrary opinion to determine which factors drive quality of life.
* **Accessibility:** To make complex government datasets (like air quality sensors and exam results) accessible to non-technical users through an intuitive, interactive interface.

---

## Key Features

* **Interactive Map Dashboard:** A multi-level geospatial tool allowing users to drill down from the South West region to individual streets. It features a choropleth overlay to visualize the 2024 Thrive Scores across the landscape.
* **Deep Dive Analysis:** A comprehensive reporting suite for any selected area, offering:
    * Historical trend analysis for House Prices and Air Quality.
    * Detailed performance tables for the nearest Schools, GPs, and Childcare providers.
    * Crime breakdown by type and temporal trends.
* **Data Exploration Suite:** A dedicated analytical toolkit enabling users to:
    * **Rank:** Identify top and bottom performing areas for any specific metric.
    * **Compare:** benchmark multiple neighbourhoods side-by-side.
    * **Correlate:** Discover statistical relationships between indicators (e.g., "Does higher greenspace coverage correlate with better GP satisfaction?").
* **Postcode Search:** An instant "find-as-you-type" search engine that bridges the gap between physical addresses and statistical LSOA boundaries.

---

## How It Was Built: The Data Pipeline

The Thrive Score is not a simple calculation. It is the product of a rigorous, multi-stage data pipeline designed to handle the complexity and "messiness" of real-world public data.

### 1. Spatial Analysis & Geocoding
It ingests raw data from disparate sources (ONS, DfE, Police, Ofsted). Much of this data is sparse or location-based rather than area-based. Using **k-d tree nearest-neighbour algorithms** to calculate precise distances from every neighbourhood centroid to its three nearest service providers (schools, GPs, childcare), converting physical locations into accessibility metrics.

### 2. Intelligent Imputation (MICE)
Real-world data often has gaps. Instead of discarding incomplete records—which would bias the index against rural or under-reported areas—using **Multivariate Imputation by Chained Equations (MICE)**. This advanced statistical technique models each variable as a function of the others, probabilistically filling gaps to ensure 100% data coverage without introducing significant distortion.

### 3. Winsorized Normalization
To create a fair comparison between unlike metrics (e.g., *Crime Rate per 1,000* vs. *GCSE Attainment 8*), all data is normalized to a 0-100 scale. Applying **Winsorization** to clip extreme outliers (typically the top/bottom 2.5%), preventing anomalies from skewing the entire dataset. Negative metrics (like pollution and crime) are inverted so that a higher score always represents a better outcome.

### 4. Statistical Weighting (PCA)
Rather than manually assigning weights based on opinion, using **Principal Component Analysis (PCA)** with Varimax rotation. This method identifies the underlying structure of the data, grouping correlated variables into latent "factors." Weights are then assigned mathematically based on how much variance each indicator explains, ensuring the final score reflects the true statistical drivers of prosperity.

---

## The 5 Pillars of the Thrive Score

The composite score is derived from five statistically identified domains:

1.  **Socio-Economic:** A measure of economic resilience and safety, derived from Income Deprivation Rate, Employment Deprivation Rates, and Community Safety (Crime per 1000 Population) statistics.
2.  **Environmental Safety:** A measure of physical health risks, based on annual mean concentrations of Nitrogen Dioxide (NO₂) and PM₂.₅ particulates.
3.  **Secondary Education:** A weighted performance score based on Progress 8 and Attainment 8 results from the nearest state secondary schools.
4.  **Primary Education:** A weighted performance score based on Key Stage 2 Reading and Maths from the nearest primary schools.
5.  **Childcare Quality:** A measure of early years provision, based on Ofsted quality ratings.

*Note: Contextual indicators like **Greenspace %**, **GP Satisfaction**, and **House Prices** are provided in the dashboard for context but are kept separate from the core score calculation.*

---

## Data Sources

This project is built entirely on publicly available data, primarily under the **Open Government Licence v3.0**. Key sources include:

* **Office for National Statistics (ONS):** Boundaries, Population, Postcodes.
* **Police UK:** Street-level crime data.
* **Department for Education (DfE):** School performance tables.
* **Ministry of Housing, Communities & Local Government:** Indices of Deprivation (IMD 2019).
* **Defra (UK-AIR):** Air quality modelling data.
* **NHS Digital:** GP practice locations.
* **Ofsted:** Childcare provider inspections.

*Full attribution and links to original datasets can be found within the application's "Sources & Licensing" page.*

## Project Context

This application was developed as the final project for a Master's degree in Data Science at UWE Bristol.
