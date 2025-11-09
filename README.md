# Thrive Index SW
**Safety & Prosperity Index for Children’s Neighbourhoods (South West England)**

Thrive Index SW is a public, open-source dashboard that maps neighbourhood conditions for children across South West England. It combines indicators such as crime, air quality, school performance, deprivation, health, and green space into a transparent composite score at small-area geography (LSOA), with optional time trends.

## ✨ Key Features
- Interactive map with LSOA polygons for Bristol, Cornwall, Devon, Dorset, Gloucestershire, Somerset, and Wiltshire.
- Postcode search (zoom to area) and click-to-select neighbourhoods.
- Adjustable weights for Safety/Prosperity domains.
- Year selector (where time-series data exists).
- Downloadable area table; clear Sources & Licensing page.

## 🧭 Geography
- Display geometry: LSOA 2021 (generalised, clipped) for web maps.
- Filtering: Local Authority Districts (Dec 2023) to create the South West study area.
- Search: ONS Postcode Directory (ONSPD) → postcode to LSOA (+ lat/lon).

## 🗂 Data Files (file-based model)
The app reads tidy Parquet/GeoParquet files (no database required):

    data/
      processed/
        boundaries_lsoa.geoparquet     # area_code, area_name, geometry (EPSG:4326)
        scores.parquet                  # area_code, year, domain_safety, domain_prosperity, composite
        indicators.parquet              # (optional) long form: area_code, year, indicator_id, value, unit, source
        postcode_lookup.parquet         # postcode (no space, uppercase), area_code, lat, lon

## 🚀 Quick Start (PyCharm or CLI)
1) Install dependencies

    pip install -r requirements.txt

2) Run the app

    streamlit run app/streamlit_app.py

Then open http://localhost:8501.

## 🧱 Project Structure

    app/
      streamlit_app.py            # main UI
      pages/01_Sources.py         # per-dataset attributions
      utils/licensing.py          # licence helper/footers
    data/
      raw/                        # original downloads (not committed)
      processed/                  # cleaned files the app reads
    requirements.txt
    README.md

## 🔬 Methods (short overview)
- Indicators are normalised (0–1), combined into domain scores (Safety, Prosperity), then aggregated to a composite.
- Sensitivity analysis tests weight robustness; comparative validation against IMD and historic frameworks (e.g., CWI) checks face validity.
- Time-series scores are computed where multi-year data exists to explore trends.

## 🔐 Ethics & Privacy
- No primary or identifiable data is collected. All data are public.
- Outputs are derived indicators and composite scores; limitations and caveats are documented in the app and report.

## 🪪 Licences & Attribution
This project re-uses public datasets. Attribution and licence statements are shown in the app’s sticky footer and on the Sources & Licensing page.

Core statements used in the app (replace {{YEAR}} with the current year; the app does this automatically):

- ONS boundaries (LSOA/LAD):  
  “Source: Office for National Statistics licensed under the Open Government Licence v3.0. Contains OS data © Crown copyright and database right {{YEAR}}.”

- ONS Postcode Directory (ONSPD):  
  “Contains OS data © Crown copyright and database right {{YEAR}}. Contains Royal Mail data © Royal Mail copyright and database right {{YEAR}}. Source: Office for National Statistics licensed under the Open Government Licence v3.0.”

Other datasets you may include (verify each dataset’s page before use):
- police.uk — typically OGL v3.0; attribute with link to dataset page.
- DEFRA UK-AIR — often OGL v3.0; check layer-specific terms.
- DfE / Ofsted — typically OGL v3.0; cite the release/table.
- IMD (DLUHC/MHCLG) — OGL v3.0; cite year/version.
- OHID/Fingertips/PHOF — OGL v3.0 in most cases.
- OS Open Greenspace / Natural England — OGL v3.0; include OS attribution where required.

## 📜 Compliance Tips
- Keep a “Sources & Licensing” page in the app with per-dataset statements.
- Keep a short footer note visible on every page (e.g., “Uses public data under OGL v3.0; see Sources for full attributions”).
- In the repo, include links to each dataset’s page and licence in a data dictionary.

## 🧰 Requirements
Listed in requirements.txt; core stack:
- streamlit, pandas, geopandas, pyogrio, pyarrow, shapely, folium, streamlit-folium, plotly

## 📄 Licence (code & derived data)
- Code: MIT (or your chosen licence).
- Derived data (composite scores): specify your licence; note that underlying source data remain under their original licences (e.g., OGL v3.0, OS/Royal Mail attributions).

## 🙌 Acknowledgements
Thanks to ONS, OS, Royal Mail, DEFRA, police.uk, DfE, Ofsted, OHID, and others for making public data accessible. This project is not an official statistic and any errors or interpretations are the author’s own.
