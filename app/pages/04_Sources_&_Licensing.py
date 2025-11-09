#Imports
import streamlit as st
from licensing import SOURCES
from datetime import date

st.set_page_config(page_title="Sources & Licensing", layout="wide")
st.title("Sources & Licensing")

st.markdown("This page lists per-dataset attributions, licences, and links.")

#Current Year
current_year = date.today().year

#Dynamically generate a section for each data source
for source_id, source in SOURCES.items():
    st.header(source.name)
    st.caption(source.attribution_template.format(year=current_year))
    st.markdown(f"- **Notes:** {source.notes}")
    st.markdown(f"- **Link:** [{source.url}]({source.url})")
    st.divider()

st.header("General Licence Notes")
st.markdown("""
- Most UK government data here is released under the **Open Government Licence v3.0 (OGL)**.
- This app also includes derived metrics and composite scores. These are not official statistics; methods are documented in the README.
""")