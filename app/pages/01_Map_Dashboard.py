# Imports
import streamlit as st
import folium
import geopandas as gpd
import pandas as pd
from streamlit_folium import st_folium
from licensing import generate_attribution_markdown

from utils import (
    get_2024_map_data, load_political_boundaries,
    load_greenspace_geometries, load_postcode_list,
    get_postcode_coords, find_containing_area, get_color, LAD_PALETTE
)

# Page Settings
st.set_page_config(
    page_title="Map Dashboard - Thrive Index SW",
    layout="wide"
)

# Streamlit Session State Init
if "selected_lad_code" not in st.session_state:
    st.session_state.selected_lad_code = None
if "selected_ward_code" not in st.session_state:
    st.session_state.selected_ward_code = None
if "selected_lsoa_code" not in st.session_state:
    st.session_state.selected_lsoa_code = None

# Data Loading
lad_gdf, ward_gdf = load_political_boundaries()
greenspace_gdf = load_greenspace_geometries()
lsoa_to_display, ward_to_display = get_2024_map_data()
lsoa_index_gdf_base = lsoa_to_display

# Set Target Year Strictly to 2024
TARGET_YEAR = 2024

# Calculate the bounding box for the entire South West region to set the initial map view
SW_BOUNDS = lad_gdf.total_bounds.tolist()
# Bounds in [lat, lon] format
SW_BOUNDS_CORRECTED = [[SW_BOUNDS[1], SW_BOUNDS[0]], [SW_BOUNDS[3], SW_BOUNDS[2]]]

# Postcode Search
if 'postcode_search' in st.session_state and st.session_state.postcode_search:
    selected_postcode = st.session_state.postcode_search
    coords = get_postcode_coords(selected_postcode)
    if coords:
        lat, lon = coords
        containing_lsoa_geom = find_containing_area(lsoa_index_gdf_base, lat, lon)
        if containing_lsoa_geom is not None:
            # Get LSOA code from geometry
            lsoa_code = containing_lsoa_geom['area_code']
            # Get Ward/LAD info from master_gdf
            lsoa_info = lsoa_to_display[lsoa_to_display['area_code'] == lsoa_code].iloc[0]
            # Set the session state to drill down
            st.session_state.selected_lad_code = lsoa_info['LAD25CD']
            st.session_state.selected_ward_code = lsoa_info['WD25CD']
            st.session_state.selected_lsoa_code = lsoa_info['area_code']

    # Reset the postcode search box to None to prevent logic from re-running
    st.session_state.postcode_search = None
    st.rerun()

# Sidebar
with st.sidebar:
    st.markdown("## Dashboard Controls")
    st.info(f"Viewing **2024 Thrive Index Scores**.")
    st.caption("This dashboard focuses exclusively on the 2024 composite scores.")

    # View Mode Toggle
    overview_mode = st.toggle("Regional Overview Mode", value=False,
                              help="Switch to a single view showing all LSOAs across the South West.")
    st.markdown("---")

# Helper to prepare simplified data for overview mode (Cached to avoid re-processing)
# Renamed arg to _df to prevent hashing error
@st.cache_data
def prepare_overview_data(_df):
    """Simplifies geometry and prepares tooltip text for the overview map."""
    df_simple = _df.copy()
    if 'geometry' in df_simple.columns:
        df_simple['geometry'] = df_simple['geometry'].simplify(tolerance=0.001, preserve_topology=True)

    # Pre-calculate integers for cleaner display
    score_cols = [
        'Final_CI_Score',
        'Socio-Economic_Deprivation_Score',
        'Environmental_Safety_Score',
        'Secondary_Education_Score',
        'Primary_Education_Score',
        'Childcare_Quality_Score'
    ]
    for col in score_cols:
        if col in df_simple.columns:
            df_simple[col] = df_simple[col].fillna(0).round(0).astype(int)

    return df_simple


# Sidebar Continued
with st.sidebar:
    st.markdown("## Navigation")
    postcodes = load_postcode_list()

    # Conditional Postcode Search
    st.selectbox(
        "Search by Postcode",
        options=postcodes,
        index=None,
        placeholder="Search for a postcode...",
        key='postcode_search',
        label_visibility="collapsed",
        disabled=overview_mode,
        help="Postcode search is disabled in Regional Overview mode. Switch back to drill-down mode to use."
    )
    if overview_mode:
        st.caption("🔒 *Search disabled in Overview Mode*")

    st.markdown("### Current Selection")

    if overview_mode:
        st.caption(
            "Detailed navigation is disabled in Overview Mode. Hover over any neighbourhood to see detailed scores.")
        # No selection logic needed here for overview mode as clicks are disabled

    elif not st.session_state.get("selected_lad_code"):
        st.info("Click a region on the map or use the search box above.")

    # Local Authority Info
    elif not st.session_state.get("selected_ward_code"):
        lad_row = lad_gdf[lad_gdf["lad_code"] == st.session_state.selected_lad_code].iloc[0]
        st.markdown(f"**Local Authority:** {lad_row['lad_name']}")
        st.markdown(f"**LAD Code:** `{lad_row['lad_code']}`")
        st.info("Now click a **Ward** on the map.")
        if st.button("← Back to South West Overview"):
            st.session_state.selected_lad_code = None
            st.session_state.selected_ward_code = None
            st.session_state.selected_lsoa_code = None
            st.rerun()

    # Ward Info
    elif st.session_state.get("selected_ward_code") and not st.session_state.get("selected_lsoa_code"):
        ward_code = st.session_state.selected_ward_code
        ward_row = ward_to_display[ward_to_display["WD25CD"] == ward_code]
        if not ward_row.empty:
            ward_row = ward_row.iloc[0]
            lad_name = lad_gdf[lad_gdf["lad_code"] == st.session_state.selected_lad_code]["lad_name"].iloc[0]
            st.markdown(f"**Local Authority:** {lad_name}")
            st.markdown(f"**Ward:** {ward_row['WD25NM']}")
            st.metric(label=f"Total Population ({TARGET_YEAR})", value=f"{int(ward_row.get('population', 0)):,}")
            st.metric(label=f"Median House Price ({TARGET_YEAR})",
                      value=f"£{ward_row.get('latest_median_house_price', 0):,}")
            if st.button(f"← Back to {lad_name} Wards"):
                st.session_state.selected_ward_code = None
                st.session_state.selected_lsoa_code = None
                st.rerun()
            with st.container(border=True):
                st.page_link("pages/02_Deep_Dive.py", label="View Ward Deep Dive", icon="📊")
        else:
            st.warning(f"No data available for this ward in {TARGET_YEAR}.")
            lad_name = lad_gdf[lad_gdf["lad_code"] == st.session_state.selected_lad_code]["lad_name"].iloc[0]
            if st.button(f"← Back to {lad_name} Wards"):
                st.session_state.selected_ward_code = None
                st.session_state.selected_lsoa_code = None
                st.rerun()

    # LSOA Info
    elif st.session_state.get("selected_lsoa_code"):
        lsoa_code = st.session_state.selected_lsoa_code
        lsoa_row = lsoa_to_display[lsoa_to_display["area_code"] == lsoa_code]
        if not lsoa_row.empty:
            lsoa_row = lsoa_row.iloc[0]
            lad_name = lad_gdf[lad_gdf["lad_code"] == st.session_state.selected_lad_code]["lad_name"].iloc[0]
            st.markdown(f"**Local Authority:** {lad_name}")
            st.markdown(f"**Ward:** {lsoa_row['WD25NM']}")
            st.markdown(f"**Neighbourhood:** *{lsoa_row['display_name']}*")
            st.markdown(f"**LSOA Code:** `{lsoa_row['area_code']}`")
            st.metric(label=f"Population ({TARGET_YEAR})", value=f"{int(lsoa_row.get('population', 0)):,}")
            st.metric(label=f"Median House Price ({TARGET_YEAR})",
                      value=f"£{lsoa_row.get('latest_median_house_price', 0):,}")
            ward_name = ward_gdf[ward_gdf["WD25CD"] == st.session_state.selected_ward_code]["WD25NM"].iloc[0]
            if st.button(f"← Back to {ward_name} Level"):
                st.session_state.selected_lsoa_code = None
                st.rerun()
            # Deep Dive Page Nav
            with st.container(border=True):
                st.page_link("pages/02_Deep_Dive.py", label="View Deep Dive", icon="📊",
                             use_container_width=True)
        else:
            st.warning(f"No data available for this neighbourhood in {TARGET_YEAR}.")
            ward_name = ward_gdf[ward_gdf["WD25CD"] == st.session_state.selected_ward_code]["WD25NM"].iloc[0]
            if st.button(f"← Back to {ward_name} Level"):
                st.session_state.selected_lsoa_code = None
                st.rerun()

    st.markdown("---")

# Page Layout
# Check if we are in overview mode to adjust layout columns
if overview_mode:
    # Use full width for map, no details pane
    left_col, right_col = st.columns([1, 0.01], gap="small")
else:
    # Standard split layout
    left_col, right_col = st.columns([3, 1], gap="small")

# Map Display (Left Col)
with left_col:
    # Use standard tiles to avoid potential heavy custom tile loading issues
    m = folium.Map(tiles="OpenStreetMap", control_scale=True, min_zoom=7, max_zoom=16)

    # Regional Overview
    if overview_mode:
        # Load pre-processed/simplified data (Cached)
        lsoas_to_draw = prepare_overview_data(lsoa_to_display)

        # Use Choropleth for coloring, but bind to the simplified data
        # Attach the tooltip directly here to avoid needing a separate clickable GeoJson layer
        folium.Choropleth(
            geo_data=lsoas_to_draw,
            name='Thrive Score (Overview)',
            data=lsoas_to_draw,
            columns=['area_code', 'Final_CI_Score'],
            key_on='feature.properties.area_code',
            fill_color='RdYlGn',
            fill_opacity=0.7,
            line_opacity=0.1,  # Very thin lines
            line_weight=0.5,
            legend_name=f'Final Composite Score ({TARGET_YEAR})',
            bins=[0, 20, 40, 60, 80, 100],
            highlight=True  # Enable simple highlight on hover
        ).add_to(m)

        # Add Tooltips using a GeoJson layer that is transparent and NOT clickable (no popup)
        # but provides the hover information.
        # Note: We rely on the fact that Folium/Leaflet handles tooltips on the layer.
        # By NOT returning object from st_folium for this layer interaction, we avoid re-runs.

        folium.GeoJson(
            lsoas_to_draw,
            style_function=lambda x: {'color': 'transparent', 'fillColor': 'transparent', 'weight': 0},
            highlight_function=lambda x: {'weight': 2, 'color': 'black'},
            tooltip=folium.GeoJsonTooltip(
                fields=[
                    "display_name",
                    "Final_CI_Score",
                    "Socio-Economic_Deprivation_Score",
                    "Environmental_Safety_Score",
                    "Secondary_Education_Score",
                    "Primary_Education_Score",
                    "Childcare_Quality_Score"
                ],
                aliases=[
                    "Neighbourhood:",
                    "Thrive Score:",
                    "Socio-Economic:",
                    "Env. Safety:",
                    "Secondary Ed:",
                    "Primary Ed:",
                    "Childcare:"
                ],
                sticky=True,
                style="font-family: sans-serif; font-size: 12px; padding: 10px; background-color: white; border: 1px solid black; border-radius: 5px;"
            )
        ).add_to(m)

        m.fit_bounds(SW_BOUNDS_CORRECTED)

    # Drill-down Navigation
    else:
        # Level 1 - Local Authority (LAD)
        if st.session_state.selected_lad_code is None:
            def lad_style(feature):
                return {
                    "fillColor": get_color(feature["properties"]["lad_code"], LAD_PALETTE),
                    "fillOpacity": 0.4,
                    "color": "#000",
                    "weight": 2
                }


            folium.GeoJson(
                lad_gdf,
                style_function=lad_style,
                highlight_function=lambda f: {"fillOpacity": 0.8, "weight": 3},
                tooltip=folium.GeoJsonTooltip(fields=["lad_name"], aliases=["Local Authority:"], sticky=True)
            ).add_to(m)
            m.fit_bounds(SW_BOUNDS_CORRECTED)

        # Level 2 - Ward
        elif st.session_state.selected_ward_code is None:
            lad_code = st.session_state.selected_lad_code
            wards_to_draw = ward_gdf[ward_gdf["LAD25CD"] == lad_code]
            # Merge scores for display (Final_CI_Score)
            wards_to_draw = wards_to_draw.merge(
                ward_to_display[['WD25CD', 'Final_CI_Score']],
                on="WD25CD",
                how="left"
            )
            wards_to_draw['Final_CI_Score'] = wards_to_draw['Final_CI_Score'].fillna(0)
            wards_to_draw['Final_CI_Score'] = wards_to_draw['Final_CI_Score'].round(0).astype(int)

            # Create a choropleth layer with score bins from 0 to 100
            choropleth = folium.Choropleth(
                geo_data=wards_to_draw,
                name='Thrive Score',
                data=wards_to_draw,
                columns=['WD25CD', 'Final_CI_Score'],
                key_on='feature.properties.WD25CD',
                fill_color='RdYlGn',
                fill_opacity=0.6,
                line_opacity=0.8,
                legend_name=f'Final Composite Score ({TARGET_YEAR})',
                bins=[0, 20, 40, 60, 80, 100]
            ).add_to(m)

            # Hover tooltip
            folium.GeoJson(
                wards_to_draw,
                style_function=lambda x: {"fillOpacity": 0, "color": "#000", "weight": 1.5},
                highlight_function=lambda f: {"fillOpacity": 0.5, "weight": 3},
                tooltip=folium.GeoJsonTooltip(
                    fields=["WD25NM", "Final_CI_Score"],
                    aliases=["Ward:", "Thrive Score:"],
                    sticky=True
                )
            ).add_to(m)

            lad_bounds = lad_gdf[lad_gdf["lad_code"] == lad_code].total_bounds.tolist()
            m.fit_bounds([[lad_bounds[1], lad_bounds[0]], [lad_bounds[3], lad_bounds[2]]])

        # Level 3 - Neighbourhood (LSOA)
        else:
            ward_code = st.session_state.selected_ward_code
            selected_lsoa_code = st.session_state.selected_lsoa_code

            # Get LSOA data for the selected year
            lsoas_to_draw = lsoa_to_display[lsoa_to_display["WD25CD"] == ward_code].copy()
            if lsoas_to_draw.empty:
                st.error(f"No LSOA data to display for this ward in {TARGET_YEAR}.")

            # Use Final_CI_Score
            lsoas_to_draw['Final_CI_Score'] = lsoas_to_draw['Final_CI_Score'].fillna(0).round(0).astype(int)
            lsoas_to_draw['tooltip_text'] = lsoas_to_draw['display_name']

            # Create choropleth layer for LSOA with scoring bins from 0 to 100
            choropleth = folium.Choropleth(
                geo_data=lsoas_to_draw,
                name='Thrive Score',
                data=lsoas_to_draw,
                columns=['area_code', 'Final_CI_Score'],
                key_on='feature.properties.area_code',
                fill_color='RdYlGn',
                fill_opacity=0.6,
                line_opacity=0.8,
                legend_name=f'Final Composite Score ({TARGET_YEAR})',
                bins=[0, 20, 40, 60, 80, 100]
            ).add_to(m)


            # Selected LSOA layout
            def lsoa_style(feature):
                code = feature["properties"]["area_code"]
                if selected_lsoa_code == code:
                    return {"fillColor": "#FFFFFF", "fillOpacity": 0.7, "color": "#FF0000",
                            "weight": 3}  # Selected LSOA
                else:
                    return {"fillColor": "#000000", "fillOpacity": 0, "color": "#333", "weight": 1.5}  # Other LSOAs


            # Hovering over LSOA
            def highlight_lsoa(feature):
                return {"fillOpacity": 0.5, "weight": 3, "color": "#000"}


            # Add LSOA layer to the map as clickable
            folium.GeoJson(
                lsoas_to_draw,
                style_function=lsoa_style,
                highlight_function=highlight_lsoa,
                tooltip=folium.GeoJsonTooltip(
                    fields=["tooltip_text", "Final_CI_Score"],
                    aliases=["Neighbourhood:", "Thrive Score:"],
                    sticky=True
                )
            ).add_to(m)

            # Overlay greenspace geometries
            ward_boundary = ward_gdf[ward_gdf["WD25CD"] == ward_code]
            if not ward_boundary.empty and not greenspace_gdf.empty:
                greenspaces_in_ward = gpd.clip(greenspace_gdf, ward_boundary)
                greenspace_style = {"fillColor": "#3A7D44", "color": "#27502D", "weight": 1, "fillOpacity": 0.7}
                folium.GeoJson(
                    greenspaces_in_ward,
                    style_function=lambda x: greenspace_style,
                    tooltip="Greenspace"
                ).add_to(m)

            # Fit map to bounds of selected ward
            ward_bounds_gdf = ward_gdf[ward_gdf["WD25CD"] == ward_code]
            if not ward_bounds_gdf.empty:
                ward_bounds = ward_bounds_gdf.total_bounds.tolist()
                m.fit_bounds([[ward_bounds[1], ward_bounds[0]], [ward_bounds[3], ward_bounds[2]]])

    # Render the complete Folium map
    # Logic: If in Overview mode, we DON'T care about clicks, so we don't pass them back to Streamlit
    # This prevents the re-run cycle.
    if overview_mode:
        map_output = st_folium(m, width="100%", height=650, key="map_overview", returned_objects=[])
    else:
        map_output = st_folium(m, width="100%", height=650, key="map", returned_objects=["last_clicked"])

# Details Pane (Right Col) - ONLY IN STANDARD MODE
if not overview_mode:
    with right_col:
        with st.container(height=650, border=False):
            st.markdown(f"#### 2024 Thrive Index Score")

            # Logic for determining which row to display (Ward vs LSOA)
            display_row = None
            is_ward = False

            if st.session_state.get("selected_ward_code") and not st.session_state.get("selected_lsoa_code"):
                # Ward Level
                ward_code = st.session_state.selected_ward_code
                ward_row = ward_to_display[ward_to_display["WD25CD"] == ward_code]
                if not ward_row.empty:
                    display_row = ward_row.iloc[0]
                    is_ward = True

            elif st.session_state.get("selected_lsoa_code"):
                # LSOA Level
                lsoa_code = st.session_state.selected_lsoa_code
                lsoa_row = lsoa_to_display[lsoa_to_display["area_code"] == lsoa_code]
                if not lsoa_row.empty:
                    display_row = lsoa_row.iloc[0]
                    is_ward = False

            # Display Metrics
            if display_row is not None:
                # Main Score Check
                final_score_raw = display_row.get('Final_CI_Score')

                # Check for NaN to prevent ValueError in st.progress
                if pd.isna(final_score_raw):
                    st.warning("Composite Score not available for this area.")
                else:
                    final_score = int(round(final_score_raw))
                    label_text = "Thrive Index Score (Ward Avg.)" if is_ward else "Thrive Index Score"
                    st.metric(label=f"**{label_text}**", value=f"{final_score}/100")
                    st.progress(final_score)

                st.markdown("---")
                st.markdown("##### 5 Core Pillars")

                # 1. Socio-Economic Deprivation
                se_score = display_row.get('Socio-Economic_Deprivation_Score')
                se_score = int(round(se_score)) if pd.notna(se_score) else 0

                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown("Socio-Economic Deprivation")
                with col2:
                    st.markdown(f"**{se_score}/100**")
                st.progress(se_score)

                # 2. Environmental Safety
                es_score = display_row.get('Environmental_Safety_Score')
                es_score = int(round(es_score)) if pd.notna(es_score) else 0

                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown("Environmental Safety")
                with col2:
                    st.markdown(f"**{es_score}/100**")
                st.progress(es_score)

                # 3. Secondary Education
                sec_score = display_row.get('Secondary_Education_Score')
                sec_score = int(round(sec_score)) if pd.notna(sec_score) else 0

                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown("Secondary Education")
                with col2:
                    st.markdown(f"**{sec_score}/100**")
                st.progress(sec_score)

                # 4. Primary Education
                pri_score = display_row.get('Primary_Education_Score')
                pri_score = int(round(pri_score)) if pd.notna(pri_score) else 0

                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown("Primary Education")
                with col2:
                    st.markdown(f"**{pri_score}/100**")
                st.progress(pri_score)

                # 5. Childcare Quality
                cc_score = display_row.get('Childcare_Quality_Score')
                cc_score = int(round(cc_score)) if pd.notna(cc_score) else 0

                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown("Childcare Quality")
                with col2:
                    st.markdown(f"**{cc_score}/100**")
                st.progress(cc_score)

                if is_ward:
                    st.info("Click a Neighbourhood on the Map to Dive Deeper")

            # Placeholder text
            elif not st.session_state.get("selected_lad_code"):
                st.info("Select a region on the map, then a ward, to see key indicators.")
            elif not st.session_state.get("selected_ward_code"):
                st.info(f"Select a ward to see its key indicators.")

# Map Click Handling (ONLY ACTIVE IN DRILL-DOWN MODE)
if not overview_mode:
    if map_output and map_output.get("last_clicked"):
        lat, lon = map_output["last_clicked"]["lat"], map_output["last_clicked"]["lng"]

        # If in LAD view, find which LAD was clicked
        if st.session_state.selected_lad_code is None:
            clicked_lad = find_containing_area(lad_gdf, lat, lon)
            if clicked_lad is not None:
                st.session_state.selected_lad_code = clicked_lad["lad_code"]
                st.rerun()

        # If in Ward view, find which Ward was clicked
        elif st.session_state.selected_ward_code is None:
            wards_in_lad = ward_gdf[ward_gdf['LAD25CD'] == st.session_state.selected_lad_code]
            clicked_ward = find_containing_area(wards_in_lad, lat, lon)
            if clicked_ward is not None:
                st.session_state.selected_ward_code = clicked_ward["WD25CD"]
                st.rerun()

        # If in LSOA view, find which LSOA was clicked
        else:
            lsoas_in_ward_geoms = lsoa_index_gdf_base[
                lsoa_index_gdf_base['area_code'].isin(lsoa_to_display['area_code'])
            ]
            clicked_lsoa = find_containing_area(lsoas_in_ward_geoms, lat, lon)
            if clicked_lsoa is not None:
                if st.session_state.selected_lsoa_code != clicked_lsoa["area_code"]:
                    st.session_state.selected_lsoa_code = clicked_lsoa["area_code"]
                    st.rerun()

# Footer for Sourcing and Licensing
with st.expander("Sources & Licensing", expanded=False):
    st.markdown(generate_attribution_markdown())