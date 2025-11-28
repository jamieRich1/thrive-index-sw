#Imports
import streamlit as st
import folium
import geopandas as gpd
from streamlit_folium import st_folium
from licensing import generate_attribution_markdown

from utils import (
    load_master_data,
    load_greenspace_geometries,
    load_postcode_list,
    get_postcode_coords,
    find_containing_area,
    get_color,
    LAD_PALETTE,
    get_scored_data_for_year
)

#Page Settings
st.set_page_config(
    page_title="Map Dashboard - Thrive Index SW",
    layout="wide"
)

#Streamlit Session State Init
if "selected_lad_code" not in st.session_state:
    st.session_state.selected_lad_code = None
if "selected_ward_code" not in st.session_state:
    st.session_state.selected_ward_code = None
if "selected_lsoa_code" not in st.session_state:
    st.session_state.selected_lsoa_code = None

#Data Loading
if 'master_gdf' not in st.session_state:
    load_master_data()
lad_gdf = st.session_state['lad_gdf']
ward_gdf = st.session_state['ward_gdf']
lsoa_index_gdf_base = st.session_state['lsoa_index_gdf_base']  # Base geometries
master_gdf = st.session_state['master_gdf']  # Full data
greenspace_gdf = load_greenspace_geometries()
all_years = sorted(master_gdf['year'].unique(), reverse=True)
latest_year = all_years[0]

#Calculate the bounding box for the entire South West region to set the initial map view
SW_BOUNDS = lad_gdf.total_bounds.tolist()
#Bounds in [lat, lon] format
SW_BOUNDS_CORRECTED = [[SW_BOUNDS[1], SW_BOUNDS[0]], [SW_BOUNDS[3], SW_BOUNDS[2]]]

#Postcode Search
if 'postcode_search' in st.session_state and st.session_state.postcode_search:
    selected_postcode = st.session_state.postcode_search
    coords = get_postcode_coords(selected_postcode)
    if coords:
        lat, lon = coords
        containing_lsoa_geom = find_containing_area(lsoa_index_gdf_base, lat, lon)
        if containing_lsoa_geom is not None:
            #Get LSOA code from geometry
            lsoa_code = containing_lsoa_geom['area_code']
            #Get Ward/LAD info from master_gdf
            lsoa_info = master_gdf[master_gdf['area_code'] == lsoa_code].iloc[0]
            #Set the session state to drill down
            st.session_state.selected_lad_code = lsoa_info['LAD25CD']
            st.session_state.selected_ward_code = lsoa_info['WD25CD']
            st.session_state.selected_lsoa_code = lsoa_info['area_code']

    #Reset the postcode search box to None to prevent logic from re-running
    st.session_state.postcode_search = None
    st.rerun()

#Sidebar
with st.sidebar:
    st.markdown("## Dashboard Controls")
    #Year Selector
    try:
        default_year_index = all_years.index(st.session_state.get('selected_year', latest_year))
    except ValueError:
        default_year_index = 0
    selected_year = st.selectbox(
        "Select Year:",
        all_years,
        index=default_year_index,
        key="selected_year"
    )
    # Customise Thrive Score Weights (2-Pillar System)
    with st.sidebar.expander("Customise 'Thrive Score' Weights"):
        st.markdown("Adjust the balance between the two core pillars.")

        # 1. Set Defaults (Equal Weight: 50/50)
        w_safety_val = st.session_state.get('w_safety', 50)
        w_opp_val = st.session_state.get('w_opportunity', 50)

        # 2. The 2 Sliders (Removed Greenspace)
        w_safety = st.slider("🛡️ Safety & Air", 0, 100, w_safety_val, key="w_safety",
                             help="Crime rates and Air Quality")
        w_opp = st.slider("🚀 Opportunity", 0, 100, w_opp_val, key="w_opportunity",
                          help="Education, Healthcare, and Childcare")

        # 3. Normalisation
        total_weight = w_safety + w_opp
        if total_weight == 0: total_weight = 1

        norm_weights = {
            'safety': w_safety / total_weight,
            'opportunity': w_opp / total_weight,
        }

        # Display breakdown
        st.caption(
            f"**Breakdown:** Safety {norm_weights['safety']:.0%}, Opportunity {norm_weights['opportunity']:.0%}")

    st.markdown("---")

#Data Scoring After Controls Set
norm_weights_tuple = tuple(sorted(norm_weights.items()))
lsoa_to_display, ward_to_display = get_scored_data_for_year(
    selected_year,
    norm_weights_tuple
)

#Sidebar Continued
with st.sidebar:
    st.markdown("## Navigation")
    postcodes = load_postcode_list()
    st.selectbox(
        "Search by Postcode",
        options=postcodes,
        index=None,
        placeholder="Search for a postcode...",
        key='postcode_search',
        label_visibility="collapsed"
    )
    st.markdown("### Current Selection")
    if not st.session_state.get("selected_lad_code"):
        st.info("Click a region on the map or use the search box above.")

    #Local Authority Info
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

    #Ward Info
    elif st.session_state.get("selected_ward_code") and not st.session_state.get("selected_lsoa_code"):
        ward_code = st.session_state.selected_ward_code
        ward_row = ward_to_display[ward_to_display["WD25CD"] == ward_code]
        if not ward_row.empty:
            ward_row = ward_row.iloc[0]
            lad_name = lad_gdf[lad_gdf["lad_code"] == st.session_state.selected_lad_code]["lad_name"].iloc[0]
            st.markdown(f"**Local Authority:** {lad_name}")
            st.markdown(f"**Ward:** {ward_row['WD25NM']}")
            st.metric(label=f"Total Population ({selected_year})", value=f"{ward_row.get('population', 0):,}")
            st.metric(label=f"Median House Price ({selected_year})",
                      value=f"£{ward_row.get('latest_median_house_price', 0):,}")
            if st.button(f"← Back to {lad_name} Wards"):
                st.session_state.selected_ward_code = None
                st.session_state.selected_lsoa_code = None
                st.rerun()
            with st.container(border=True):
                st.page_link("pages/02_Deep_Dive.py", label="View Ward Deep Dive", icon="📊")
        else:
            st.warning(f"No data available for this ward in {selected_year}.")
            lad_name = lad_gdf[lad_gdf["lad_code"] == st.session_state.selected_lad_code]["lad_name"].iloc[0]
            if st.button(f"← Back to {lad_name} Wards"):
                st.session_state.selected_ward_code = None
                st.session_state.selected_lsoa_code = None
                st.rerun()

    #LSOA Info
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
            st.metric(label=f"Population ({selected_year})", value=f"{lsoa_row.get('population', 0):,}")
            st.metric(label=f"Median House Price ({selected_year})",
                      value=f"£{lsoa_row.get('latest_median_house_price', 0):,}")
            ward_name = ward_gdf[ward_gdf["WD25CD"] == st.session_state.selected_ward_code]["WD25NM"].iloc[0]
            if st.button(f"← Back to {ward_name} Level"):
                st.session_state.selected_lsoa_code = None
                st.rerun()
            #Deep Dive Page Nav
            with st.container(border=True):
                st.page_link("pages/02_Deep_Dive.py", label="View Neighbourhood Deep Dive", icon="📊")
        else:
            st.warning(f"No data available for this neighbourhood in {selected_year}.")
            ward_name = ward_gdf[ward_gdf["WD25CD"] == st.session_state.selected_ward_code]["WD25NM"].iloc[0]
            if st.button(f"← Back to {ward_name} Level"):
                st.session_state.selected_lsoa_code = None
                st.rerun()

    st.markdown("---")

#Page Layout
left_col, right_col = st.columns([3, 1], gap="small")

#Map Display (Left Col)
with left_col:
    m = folium.Map(tiles="OpenStreetMap", control_scale=True, min_zoom=7, max_zoom=16)
    #Level 1 - Local Authority (LAD)
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

    #Level 2 - Ward
    elif st.session_state.selected_ward_code is None:
        lad_code = st.session_state.selected_lad_code
        wards_to_draw = ward_gdf[ward_gdf["LAD25CD"] == lad_code]
        wards_to_draw = wards_to_draw.merge(
            ward_to_display[['WD25CD', 'composite_score']],
            on="WD25CD",
            how="left"
        )
        wards_to_draw['composite_score'] = wards_to_draw['composite_score'].fillna(0)
        wards_to_draw['composite_score'] = wards_to_draw['composite_score'].round(0).astype(int)

        #Create a choropleth layer with score bins from 0 to 100
        choropleth = folium.Choropleth(
            geo_data=wards_to_draw,
            name='Thrive Score',
            data=wards_to_draw,
            columns=['WD25CD', 'composite_score'],
            key_on='feature.properties.WD25CD',
            fill_color='RdYlGn',
            fill_opacity=0.6,
            line_opacity=0.8,
            legend_name=f'Thrive Score ({selected_year})',
            bins=[0, 20, 40, 60, 80, 100]
        ).add_to(m)

        #Hover tooltip
        folium.GeoJson(
            wards_to_draw,
            style_function=lambda x: {"fillOpacity": 0, "color": "#000", "weight": 1.5},
            highlight_function=lambda f: {"fillOpacity": 0.5, "weight": 3},
            tooltip=folium.GeoJsonTooltip(
                fields=["WD25NM", "composite_score"],
                aliases=["Ward:", "Thrive Score:"],
                sticky=True
            )
        ).add_to(m)

        lad_bounds = lad_gdf[lad_gdf["lad_code"] == lad_code].total_bounds.tolist()
        m.fit_bounds([[lad_bounds[1], lad_bounds[0]], [lad_bounds[3], lad_bounds[2]]])

    #Level 3 - Neighbourhood (LSOA)
    else:
        ward_code = st.session_state.selected_ward_code
        selected_lsoa_code = st.session_state.selected_lsoa_code

        #Get LSOA data for the selected year
        lsoas_to_draw = lsoa_to_display[lsoa_to_display["WD25CD"] == ward_code].copy()
        if lsoas_to_draw.empty:
            st.error(f"No LSOA data to display for this ward in {selected_year}.")
        lsoas_to_draw['composite_score'] = lsoas_to_draw['composite_score'].round(0).astype(int)
        lsoas_to_draw['tooltip_text'] = lsoas_to_draw['display_name']

        #Create choropleth layer for LSOA with scoring bins from 0 to 100
        choropleth = folium.Choropleth(
            geo_data=lsoas_to_draw,
            name='Thrive Score',
            data=lsoas_to_draw,
            columns=['area_code', 'composite_score'],
            key_on='feature.properties.area_code',
            fill_color='RdYlGn',
            fill_opacity=0.6,
            line_opacity=0.8,
            legend_name=f'Thrive Score ({selected_year})',
            bins=[0, 20, 40, 60, 80, 100]
        ).add_to(m)


        #Selected LSOA layout
        def lsoa_style(feature):
            code = feature["properties"]["area_code"]
            if selected_lsoa_code == code:
                return {"fillColor": "#FFFFFF", "fillOpacity": 0.7, "color": "#FF0000", "weight": 3}  # Selected LSOA
            else:
                return {"fillColor": "#000000", "fillOpacity": 0, "color": "#333", "weight": 1.5}  # Other LSOAs


        #Hovering over LSOA
        def highlight_lsoa(feature):
            return {"fillOpacity": 0.5, "weight": 3, "color": "#000"}


        #Add LSOA layer to the map as clickable
        folium.GeoJson(
            lsoas_to_draw,
            style_function=lsoa_style,
            highlight_function=highlight_lsoa,
            tooltip=folium.GeoJsonTooltip(
                fields=["tooltip_text", "composite_score"],
                aliases=["Neighbourhood:", "Thrive Score:"],
                sticky=True
            )
        ).add_to(m)

        #Overlay greenspace geometries
        ward_boundary = ward_gdf[ward_gdf["WD25CD"] == ward_code]
        if not ward_boundary.empty and not greenspace_gdf.empty:
            greenspaces_in_ward = gpd.clip(greenspace_gdf, ward_boundary)
            greenspace_style = {"fillColor": "#3A7D44", "color": "#27502D", "weight": 1, "fillOpacity": 0.7}
            folium.GeoJson(
                greenspaces_in_ward,
                style_function=lambda x: greenspace_style,
                tooltip="Greenspace"
            ).add_to(m)

        #Fit map to bounds of selected ward
        ward_bounds_gdf = ward_gdf[ward_gdf["WD25CD"] == ward_code]
        if not ward_bounds_gdf.empty:
            ward_bounds = ward_bounds_gdf.total_bounds.tolist()
            m.fit_bounds([[ward_bounds[1], ward_bounds[0]], [ward_bounds[3], ward_bounds[2]]])

    #Render the complete Folium map
    map_output = st_folium(m, width="100%", height=650, key="map", returned_objects=["last_clicked"])

#Details Pane (Right Col)
with right_col:
    with st.container(height=650, border=False):
        st.markdown(f"#### Key Indicators ({selected_year})")

        #Ward-level indicators if a ward is selected.
        if st.session_state.get("selected_ward_code") and not st.session_state.get("selected_lsoa_code"):
            ward_code = st.session_state.selected_ward_code
            ward_row = ward_to_display[ward_to_display["WD25CD"] == ward_code]
            if not ward_row.empty:
                ward_row = ward_row.iloc[0]
                composite_score = ward_row.get('composite_score', 0)
                st.metric(label="**Thrive Index Score (Ward Avg.)**", value=f"{composite_score:.0f}/100")
                st.progress(int(composite_score))
                st.markdown("---")

                #Greenspace (Informational Only)
                greenspace_percent = ward_row.get('greenspace_percentage', 0)
                st.markdown("#### 🌳 Greenspace")
                st.metric(label="Area Coverage", value=f"{greenspace_percent:.1f}%")
                st.caption("Percentage of land covered by accessible greenspace. (Informational)")
                st.markdown("---")

                #Air Quality
                air_quality_score = ward_row.get('air_quality_score', 0)
                no2_conc = ward_row.get('no2_mean_concentration', 0)
                pm25_conc = ward_row.get('pm25_mean_concentration', 0)
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown("Air Quality Score")
                with col2:
                    st.markdown(f"**{air_quality_score:.0f}/100**")
                st.progress(int(air_quality_score))
                st.caption(f"Avg. $NO_2$: {no2_conc:.1f} µg/m³ | $PM_{{2.5}}$: {pm25_conc:.1f} µg/m³")

                #Community Safety
                safety_score = ward_row.get('community_safety_score', 0)
                crime_rate = ward_row.get('crime_rate_per_1000', 0)
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown("Community Safety Score")
                with col2:
                    st.markdown(f"**{safety_score:.0f}/100**")
                st.progress(int(safety_score))
                st.caption(f"Avg. crime rate: {crime_rate:.1f} per 1,000 people")

                #Education
                education_score = ward_row.get('education_score', 0)
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown("Education Score")
                with col2:
                    st.markdown(f"**{education_score:.0f}/100**")
                st.progress(int(education_score))
                st.caption(
                    f"Primary: {ward_row.get('primary_education_score', 0):.0f}/100 | Secondary: {ward_row.get('secondary_education_score', 0):.0f}/100")

                #Healthcare
                healthcare_score = ward_row.get('healthcare_score', 0)
                avg_distance = ward_row.get('avg_distance_to_gp_km', 0)
                avg_satisfaction = ward_row.get('avg_gp_satisfaction', 0)
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown("Healthcare Access Score")
                with col2:
                    st.markdown(f"**{healthcare_score:.0f}/100**")
                st.progress(int(healthcare_score))
                st.caption(f"Avg. distance: {avg_distance:.1f} km | GP satisfaction: {avg_satisfaction:.0f}%")

                #Childcare
                childcare_score = ward_row.get('childcare_score', 0)
                avg_distance = ward_row.get('avg_distance_to_childcare_km', 0)
                avg_quality = ward_row.get('avg_childcare_quality_score', 0)
                total_places = ward_row.get('total_childcare_places_nearby', 0)
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown("Childcare Score")
                with col2:
                    st.markdown(f"**{childcare_score:.0f}/100**")
                st.progress(int(childcare_score))
                st.caption(
                    f"Avg. distance: {avg_distance:.1f} km | Avg. quality: {avg_quality:.1f}/4.0 | Places: {total_places:.0f}")

                st.info("Click a Neighbourhood on the Map to Dive Deeper")

        #LSOA-level details when neighbourhood is selected
        elif st.session_state.get("selected_lsoa_code"):
            lsoa_code = st.session_state.selected_lsoa_code
            lsoa_row = lsoa_to_display[lsoa_to_display["area_code"] == lsoa_code]

            if not lsoa_row.empty:
                lsoa_row = lsoa_row.iloc[0]
                composite_score = lsoa_row.get('composite_score', 0)
                st.metric(label="**Thrive Index Score**", value=f"{composite_score:.0f}/100")
                st.progress(int(composite_score))
                st.markdown("---")

                #Greenspace (Informational Only)
                greenspace_percent = lsoa_row.get('greenspace_percentage', 0)
                st.markdown("#### 🌳 Greenspace")
                st.metric(label="Area Coverage", value=f"{greenspace_percent:.1f}%")
                st.caption("Percentage of land covered by accessible greenspace. (Informational)")
                st.markdown("---")

                #Air Quality
                air_quality_score = lsoa_row.get('air_quality_score', 0)
                no2_conc = lsoa_row.get('no2_mean_concentration', 0)
                pm25_conc = lsoa_row.get('pm25_mean_concentration', 0)
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown("Air Quality Score")
                with col2:
                    st.markdown(f"**{air_quality_score:.0f}/100**")
                st.progress(int(air_quality_score))
                st.caption(f"Avg. $NO_2$: {no2_conc:.1f} µg/m³ | $PM_{{2.5}}$: {pm25_conc:.1f} µg/m³")

                #Community Safety
                safety_score = lsoa_row.get('community_safety_score', 0)
                crime_rate = lsoa_row.get('crime_rate_per_1000', 0)
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown("Community Safety Score")
                with col2:
                    st.markdown(f"**{safety_score:.0f}/100**")
                st.progress(int(safety_score))
                st.caption(f"Crime rate: {crime_rate:.1f} per 1,000 people")

                #Education
                education_score = lsoa_row.get('education_score', 0)
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown("Education Score")
                with col2:
                    st.markdown(f"**{education_score:.0f}/100**")
                st.progress(int(education_score))
                st.caption(
                    f"Primary: {lsoa_row.get('primary_education_score', 0):.0f}/100 | Secondary: {lsoa_row.get('secondary_education_score', 0):.0f}/100")

                #Healthcare
                healthcare_score = lsoa_row.get('healthcare_score', 0)
                avg_distance = lsoa_row.get('avg_distance_to_gp_km', 0)
                avg_satisfaction = lsoa_row.get('avg_gp_satisfaction', 0)
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown("Healthcare Access Score")
                with col2:
                    st.markdown(f"**{healthcare_score:.0f}/100**")
                st.progress(int(healthcare_score))
                st.caption(f"Avg. distance: {avg_distance:.1f} km | GP satisfaction: {avg_satisfaction:.0f}%")

                #Childcare
                childcare_score = lsoa_row.get('childcare_score', 0)
                avg_distance = lsoa_row.get('avg_distance_to_childcare_km', 0)
                avg_quality = lsoa_row.get('avg_childcare_quality_score', 0)
                total_places = lsoa_row.get('total_childcare_places_nearby', 0)
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown("Childcare Score")
                with col2:
                    st.markdown(f"**{childcare_score:.0f}/100**")
                st.progress(int(childcare_score))
                st.caption(
                    f"Avg. distance: {avg_distance:.1f} km | Avg. quality: {avg_quality:.1f}/4.0 | Places: {total_places:.0f}")

        #Placeholder text if no ward/LSOA is selected
        elif not st.session_state.get("selected_lad_code"):
            st.info("Select a region on the map, then a ward, to see key indicators.")
        elif not st.session_state.get("selected_ward_code"):
            st.info(f"Select a ward to see its key indicators.")

#Map Click Handling
if map_output and map_output.get("last_clicked"):
    lat, lon = map_output["last_clicked"]["lat"], map_output["last_clicked"]["lng"]

    #If in LAD view, find which LAD was clicked
    if st.session_state.selected_lad_code is None:
        clicked_lad = find_containing_area(lad_gdf, lat, lon)
        if clicked_lad is not None:
            st.session_state.selected_lad_code = clicked_lad["lad_code"]
            st.rerun()

    #If in Ward view, find which Ward was clicked
    elif st.session_state.selected_ward_code is None:
        wards_in_lad = ward_gdf[ward_gdf['LAD25CD'] == st.session_state.selected_lad_code]
        clicked_ward = find_containing_area(wards_in_lad, lat, lon)
        if clicked_ward is not None:
            st.session_state.selected_ward_code = clicked_ward["WD25CD"]
            st.rerun()

    #If in LSOA view, find which LSOA was clicked
    else:
        lsoas_in_ward_geoms = lsoa_index_gdf_base[
            lsoa_index_gdf_base['area_code'].isin(lsoa_to_display['area_code'])
        ]
        clicked_lsoa = find_containing_area(lsoas_in_ward_geoms, lat, lon)
        if clicked_lsoa is not None:
            if st.session_state.selected_lsoa_code != clicked_lsoa["area_code"]:
                st.session_state.selected_lsoa_code = clicked_lsoa["area_code"]
                st.rerun()

#Footer for Sourcing and Licensing
with st.expander("Sources & Licensing", expanded=False):
    st.markdown(generate_attribution_markdown())