# Imports
import streamlit as st
import pandas as pd
import numpy as np
from utils import (
    load_political_boundaries,
    load_lsoa_boundaries,
    load_attribute_data,
    load_monthly_crime_data,
    load_house_price_history,
    get_postcode_coords,
    find_containing_area,
    load_gp_historical_data,
    load_childcare_historical_data,
    load_primary_historical_data,
    load_secondary_historical_data
)
from licensing import generate_attribution_markdown
import plotly.express as px

# Page Config
st.set_page_config(
    page_title="Deep Dive - Thrive Index SW",
    layout="wide"
)

# Constants
TARGET_YEAR = 2024
COMMUNITY_SAFETY_CRIMES = [
    'Anti-social behaviour', 'Burglary', 'Robbery',
    'Criminal damage and arson', 'Violence and sexual offences', 'Public order'
]
OFSTED_RATING_MAP = {
    '1': 'Outstanding (1)',
    '2': 'Good (2)',
    '3': 'Requires Improvement (3)',
    '4.0': 'Inadequate (4)',
    '4': 'Inadequate (4)',
    '9': 'Not Assessed',
    'nan': 'N/A',
    'N/A': 'N/A'
}

# Data Loading
lad_gdf, ward_gdf = load_political_boundaries()
lsoa_index_gdf_base = load_lsoa_boundaries()
master_gdf = load_attribute_data(year_filter=None)
st.session_state['master_gdf'] = master_gdf
monthly_crime_df = load_monthly_crime_data()
gp_historical_df = load_gp_historical_data()
childcare_historical_df = load_childcare_historical_data()
primary_historical_df = load_primary_historical_data()
secondary_historical_df = load_secondary_historical_data()

# Sidebar
with st.sidebar:
    st.markdown("##### Select an area to explore")
    lad_names = sorted(lad_gdf['lad_name'].unique())
    lad_index = None
    if 'selected_lad_code' in st.session_state and st.session_state.selected_lad_code:
        try:
            selected_name = lad_gdf[lad_gdf['lad_code'] == st.session_state.selected_lad_code]['lad_name'].iloc[0]
            lad_index = lad_names.index(selected_name)
        except (IndexError, ValueError):
            lad_index = None
    selected_lad_name = st.selectbox("Local Authority", options=lad_names, index=lad_index, placeholder="Choose...")
    ward_options = []
    if selected_lad_name:
        selected_lad_code = lad_gdf[lad_gdf['lad_name'] == selected_lad_name]['lad_code'].iloc[0]
        ward_options = sorted(ward_gdf[ward_gdf['LAD25CD'] == selected_lad_code]['WD25NM'].unique())
    ward_index = None
    if 'selected_ward_code' in st.session_state and st.session_state.selected_lad_code and selected_lad_name:
        try:
            selected_name = ward_gdf[ward_gdf['WD25CD'] == st.session_state.selected_ward_code]['WD25NM'].iloc[0]
            if selected_name in ward_options:
                ward_index = ward_options.index(selected_name)
        except (IndexError, ValueError):
            ward_index = None
    selected_ward_name = st.selectbox("Ward", options=ward_options, index=ward_index, placeholder="Choose...",
                                      disabled=not selected_lad_name)
    lsoa_options_names = []
    if selected_ward_name:
        selected_lad_code_for_ward = lad_gdf[lad_gdf['lad_name'] == selected_lad_name]['lad_code'].iloc[0]
        wards_in_lad = ward_gdf[ward_gdf['LAD25CD'] == selected_lad_code_for_ward]
        selected_ward_code = wards_in_lad[wards_in_lad['WD25NM'] == selected_ward_name]['WD25CD'].iloc[0]
        master_gdf = st.session_state['master_gdf']
        lsoas_in_ward = master_gdf[master_gdf['WD25CD'] == selected_ward_code]
        lsoa_options_names = sorted(
            lsoas_in_ward['display_name'].unique())
    lsoa_index = None
    if 'selected_lsoa_code' in st.session_state and st.session_state.selected_lsoa_code and selected_ward_name:
        try:
            selected_name = \
                st.session_state['master_gdf'][
                    st.session_state['master_gdf']['area_code'] == st.session_state.selected_lsoa_code][
                    'display_name'].iloc[0]
            if selected_name in lsoa_options_names:
                lsoa_index = lsoa_options_names.index(selected_name)
        except (IndexError, ValueError):
            lsoa_index = None
    selected_lsoa_name = st.selectbox("Neighbourhood (Optional)", options=lsoa_options_names, index=lsoa_index,
                                      placeholder="All of Ward...", disabled=not selected_ward_name,
                                      key="lsoa_selector")
    if st.button("Show Details", type="primary", disabled=not selected_ward_name):
        st.session_state.selected_lad_code = None
        st.session_state.selected_ward_code = None
        st.session_state.selected_lsoa_code = None
        if selected_lad_name:
            st.session_state.selected_lad_code = lad_gdf[lad_gdf['lad_name'] == selected_lad_name]['lad_code'].iloc[0]
        if selected_ward_name:
            st.session_state.selected_ward_code = ward_gdf[
                (ward_gdf['WD25NM'] == selected_ward_name) & (
                        ward_gdf['LAD25CD'] == st.session_state.selected_lad_code)][
                'WD25CD'].iloc[0]
        if selected_lsoa_name:
            st.session_state.selected_lsoa_code = \
                st.session_state['master_gdf'][st.session_state['master_gdf']['display_name'] == selected_lsoa_name][
                    'area_code'].iloc[0]
        st.rerun()

    # Reset Button
    if st.button("Reset Filters / New Search"):
        keys_to_clear = ["selected_lad_code", "selected_ward_code", "selected_lsoa_code"]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    # Map Dashboard Link
    with st.container(border=True):
        st.page_link("pages/01_Map_Dashboard.py", label="View in Map Dashboard", icon="🗺️")

    st.markdown("---")

# Retrieve Data
latest_lsoa_data = master_gdf[master_gdf['year'] == TARGET_YEAR].copy()
agg_rules = {
    'Final_CI_Score': 'mean',
    'Socio-Economic_Deprivation_Score': 'mean',
    'Environmental_Safety_Score': 'mean',
    'Secondary_Education_Score': 'mean',
    'Primary_Education_Score': 'mean',
    'Childcare_Quality_Score': 'mean',
    'population': 'sum',
    'latest_median_house_price': 'mean',
    'IMD_Decile': 'mean',
    'Income_Decile': 'mean',
    'Employment_Decile': 'mean',
    'Health_Decile': 'mean'
}
valid_aggs = {k: v for k, v in agg_rules.items() if k in latest_lsoa_data.columns}
latest_ward_data = latest_lsoa_data.groupby(['WD25CD', 'WD25NM'], as_index=False).agg(valid_aggs)
all_years = sorted(st.session_state['master_gdf']['year'].unique())

# HELPER FUNCTIONS Shared by LSOA & Ward
# Get and display primary schools
def get_primary_schools(lsoa_df):
    """Extracts primary school data from a dataframe of LSOAs"""
    all_primary_schools = []
    # If no school data columns exist, return empty
    if not any('primary_school_' in col for col in lsoa_df.columns):
        return pd.DataFrame(), [], pd.DataFrame()

    for _, lsoa_row in lsoa_df.iterrows():
        for i in range(1, 4):
            # Try likely column names (primary_school_1_name is standard, but could differ)
            name = lsoa_row.get(f'primary_school_{i}_name')
            urn = lsoa_row.get(f'primary_school_{i}_urn')

            if name and pd.notna(name) and name != 'N/A' and urn and pd.notna(urn):
                all_primary_schools.append({
                    "School Name": name,
                    "% Meeting Standard": lsoa_row.get(f'primary_school_{i}_pass_rate', np.nan),
                    "Read Score": lsoa_row.get(f'primary_school_{i}_read_score', np.nan),
                    "Math Score": lsoa_row.get(f'primary_school_{i}_math_score', np.nan),
                    "URN": str(int(float(urn)))  # Ensure URN is stringified clean integer
                })

    if not all_primary_schools:
        return pd.DataFrame(), [], pd.DataFrame()

    df = pd.DataFrame(all_primary_schools)
    school_urns_df = df[['School Name', 'URN']].drop_duplicates()
    df.drop_duplicates(subset=["School Name"], inplace=True)
    if "Read Score" in df.columns:
        df.sort_values(by="Read Score", ascending=False, inplace=True)
    return df.drop(columns=['URN']), df['School Name'].tolist(), school_urns_df


# Get and display secondary schools
def get_secondary_schools(lsoa_df, school_type='state'):
    """Extracts secondary school data from a dataframe of LSOAs"""
    all_schools = []
    # If no school data columns exist, return empty
    if not any('school_' in col for col in lsoa_df.columns):
        return pd.DataFrame(), [], pd.DataFrame()

    for _, lsoa_row in lsoa_df.iterrows():
        for i in range(1, 4):
            name = lsoa_row.get(f'school_{i}_name')
            nftype = lsoa_row.get(f'school_{i}_nftype', 'NA')
            urn = lsoa_row.get(f'school_{i}_urn')

            # Normalize nftype check
            is_ind = (str(nftype).strip().upper() == 'IND')

            if (school_type == 'independent' and not is_ind) or (school_type == 'state' and is_ind):
                continue

            if name and pd.notna(name) and name != 'N/A' and urn and pd.notna(urn):
                all_schools.append({
                    "School Name": name,
                    "Progress 8": lsoa_row.get(f'school_{i}_progress_8', np.nan),
                    "Attainment 8": lsoa_row.get(f'school_{i}_attainment_8', np.nan),
                    "Type": nftype,
                    "URN": str(int(float(urn)))
                })

    if not all_schools:
        return pd.DataFrame(), [], pd.DataFrame()

    df = pd.DataFrame(all_schools)
    school_urns_df = df[['School Name', 'URN']].drop_duplicates()
    df.drop_duplicates(subset=["School Name"], inplace=True)
    if "Attainment 8" in df.columns:
        df.sort_values(by="Attainment 8", ascending=False, inplace=True)
    return df.drop(columns=['URN']), df['School Name'].tolist(), school_urns_df


# Display historical school data
def display_primary_school_history(school_urns_df, primary_historical_df):
    st.markdown("##### Historical School Performance (Table)")
    if not primary_historical_df.empty and not school_urns_df.empty:
        hist_df = primary_historical_df.rename(columns={
            'URN': 'urn',
            'avg_ks2_pass_rate': '% Meeting Standard',
            'read_score': 'Read Score',
            'math_score': 'Math Score'
        })
        hist_df['urn'] = hist_df['urn'].astype(str)
        unique_urns = school_urns_df['URN'].astype(str).unique()
        chart_data = hist_df[hist_df['urn'].isin(unique_urns)].copy()

        if not chart_data.empty:
            chart_data = chart_data.merge(school_urns_df, left_on='urn', right_on='URN', how='left')

            # Table 1 - % Meeting Standard
            st.markdown("**Historical: % Meeting Standard (R,W,M)**")
            table_pass_rate = chart_data.pivot_table(
                index='School Name',
                columns='year',
                values='% Meeting Standard',
                aggfunc='mean'
            ).round(0)
            table_pass_rate = table_pass_rate.reindex(sorted(table_pass_rate.columns), axis=1)
            # handle NaNs gracefully for display
            st.dataframe(table_pass_rate, width='stretch', column_config={
                year: st.column_config.NumberColumn(f"{year}", format="%d%%")
                for year in table_pass_rate.columns
            })

            # Table 2 - Read Score
            st.markdown("**Historical: Read Score**")
            table_read = chart_data.pivot_table(
                index='School Name',
                columns='year',
                values='Read Score',
                aggfunc='mean'
            ).round(1)
            table_read = table_read.reindex(sorted(table_read.columns), axis=1)
            st.dataframe(table_read, width='stretch', column_config={
                year: st.column_config.NumberColumn(f"{year}", format="%.1f")
                for year in table_read.columns
            })

            # Table 3 - Math Score
            st.markdown("**Historical: Math Score**")
            table_math = chart_data.pivot_table(
                index='School Name',
                columns='year',
                values='Math Score',
                aggfunc='mean'
            ).round(1)
            table_math = table_math.reindex(sorted(table_math.columns), axis=1)
            st.dataframe(table_math, width='stretch', column_config={
                year: st.column_config.NumberColumn(f"{year}", format="%.1f")
                for year in table_math.columns
            })
        else:
            st.info("No historical performance data found for these primary schools.")
    else:
        st.info("Historical primary school data is loading or not available.")


# Display historical secondary school data
def display_secondary_school_history(school_urns_df, secondary_historical_df):
    st.markdown("##### Historical School Performance (Table)")
    if not secondary_historical_df.empty and not school_urns_df.empty:
        hist_df = secondary_historical_df.rename(columns={
            'URN': 'urn',
            'avg_progress_8': 'Progress 8',
            'avg_attainment_8': 'Attainment 8'
        })
        hist_df['urn'] = hist_df['urn'].astype(str)
        unique_urns = school_urns_df['URN'].astype(str).unique()
        chart_data = hist_df[hist_df['urn'].isin(unique_urns)].copy()

        if not chart_data.empty:
            chart_data = chart_data.merge(school_urns_df, left_on='urn', right_on='URN', how='left')

            # Table 1 - Progress 8
            st.markdown("**Historical: Progress 8**")
            table_p8 = chart_data.pivot_table(
                index='School Name',
                columns='year',
                values='Progress 8',
                aggfunc='mean'
            ).round(2)
            table_p8 = table_p8.reindex(sorted(table_p8.columns), axis=1)
            st.dataframe(table_p8, width='stretch', column_config={
                year: st.column_config.NumberColumn(f"{year}", format="%.2f")
                for year in table_p8.columns
            })

            # Table 2 - Attainment 8
            st.markdown("**Historical: Attainment 8**")
            table_a8 = chart_data.pivot_table(
                index='School Name',
                columns='year',
                values='Attainment 8',
                aggfunc='mean'
            ).round(1)
            table_a8 = table_a8.reindex(sorted(table_a8.columns), axis=1)
            st.dataframe(table_a8, width='stretch', column_config={
                year: st.column_config.NumberColumn(f"{year}", format="%.1f")
                for year in table_a8.columns
            })
        else:
            st.info("No historical performance data found for these secondary schools.")
    else:
        st.info("Historical secondary school data is loading or not available.")


# MAIN DISPLAY - Split by whether in Ward or LSOA level view

# View 1: LSOA Level
if st.session_state.get("selected_lsoa_code"):
    lsoa_code = st.session_state.selected_lsoa_code
    # Use 2024 Scored Data for the Top Box
    lsoa_data_series = latest_lsoa_data[latest_lsoa_data["area_code"] == lsoa_code]

    if lsoa_data_series.empty:
        st.error(f"No data found for this LSOA. Please select another area.")
        st.stop()

    lsoa_data = lsoa_data_series.iloc[0]
    st.subheader(f"Neighbourhood: *{lsoa_data['display_name']}*")
    st.caption(f"Data Focus: {TARGET_YEAR}")

    with st.container(border=True):
        if st.button(f"Part of **{lsoa_data['WD25NM']}** ward (click to view ward details)"):
            st.session_state.pop('selected_lsoa_code', None)
            st.rerun()

    # Composite Scoring Section
    st.markdown("### Thrive Index Score (2024)")

    final_score_raw = lsoa_data.get('Final_CI_Score')
    if pd.isna(final_score_raw):
        st.warning("Composite Score not available for this area.")
    else:
        final_score = int(round(final_score_raw))
        m1, m2 = st.columns(2)
        with m1:
            with st.container(border=True):
                st.metric(label="Final Composite Score", value=f"{final_score}/100")
        with m2:
            with st.container(border=True):
                pop_val = lsoa_data.get('population', 0)
                st.metric(label="Population", value=f"{int(pop_val):,}" if pd.notna(pop_val) else "N/A")

        with st.container(border=True):
            st.subheader("5 Core Pillars Breakdown")
            c1, c2, c3, c4, c5 = st.columns(5)


            def get_fmt_score(row, key):
                val = row.get(key)
                if pd.notna(val):
                    return f"{int(round(val))}/100"
                return "N/A"


            with c1:
                st.metric("Socio-Economic", get_fmt_score(lsoa_data, 'Socio-Economic_Deprivation_Score'))
                st.caption("Deprivation & Crime")
            with c2:
                st.metric("Env. Safety", get_fmt_score(lsoa_data, 'Environmental_Safety_Score'))
                st.caption("Air Quality")
            with c3:
                st.metric("Secondary Ed.", get_fmt_score(lsoa_data, 'Secondary_Education_Score'))
                st.caption("Progress 8 & Attainment")
            with c4:
                st.metric("Primary Ed.", get_fmt_score(lsoa_data, 'Primary_Education_Score'))
                st.caption("KS2 Performance")
            with c5:
                st.metric("Childcare", get_fmt_score(lsoa_data, 'Childcare_Quality_Score'))
                st.caption("Ofsted Quality")

    # Historical Context Section
    # House Price Trend Container
    with st.container(border=True):
        st.subheader("Median House Price Trend")
        latest_date_obj = st.session_state.get('latest_house_price_date')
        latest_period_label = latest_date_obj.strftime('%b %Y') if latest_date_obj else 'Latest'
        latest_price = lsoa_data.get('latest_median_house_price')

        with st.container(border=True):
            price_val = f"£{int(latest_price):,}" if pd.notna(latest_price) else "N/A"
            st.metric(label=f"Latest Price ({latest_period_label})", value=price_val)

        lsoa_history_all = load_house_price_history('lsoa')
        sw_history = load_house_price_history('sw')

        if lsoa_history_all is not None and not lsoa_history_all.empty:
            lsoa_history = lsoa_history_all[lsoa_history_all['area_code'] == lsoa_code].copy()
            if not lsoa_history.empty and sw_history is not None:
                lsoa_history_chart = lsoa_history[['date', 'median_house_price']].rename(
                    columns={'median_house_price': 'Neighbourhood'}
                )
                sw_history_chart = sw_history[['date', 'median_house_price']].rename(
                    columns={'median_house_price': 'South West Average'}
                )
                chart_data = pd.merge(lsoa_history_chart, sw_history_chart, on='date', how='outer')
                chart_data['date'] = pd.to_datetime(chart_data['date'])
                chart_data.set_index('date', inplace=True)
                st.line_chart(chart_data)
            else:
                st.info("No historical house price data available for this specific neighbourhood.")
        else:
            st.info("Historical house price data is not available.")

    # Crime Breakdown
    with st.container(border=True):
        st.subheader("Crime Breakdown (Time-Series)")
        lsoa_crime_history = monthly_crime_df[monthly_crime_df['area_code'] == lsoa_code]

        if not lsoa_crime_history.empty:
            crime_years = ['All Time'] + sorted(lsoa_crime_history.index.year.unique(), reverse=True)
            selected_crime_year = st.selectbox(
                "Select Year to View:",
                crime_years,
                key="lsoa_crime_year_filter"
            )

            filtered_crime_history = lsoa_crime_history.copy()
            if selected_crime_year != 'All Time':
                filtered_crime_history = filtered_crime_history[
                    filtered_crime_history.index.year == selected_crime_year]

            crime_c1, crime_c2 = st.columns([2, 1.5])
            with crime_c1:
                if selected_crime_year == 'All Time':
                    time_scale = st.radio(
                        "Select Time Scale",
                        ('Monthly', 'Quarterly', 'Annual'),
                        horizontal=True,
                        key=f"lsoa_crime_scale"
                    )
                else:
                    time_scale = 'Monthly'

                if time_scale == 'Monthly':
                    chart_data = filtered_crime_history['monthly_crime_count']
                elif time_scale == 'Quarterly':
                    chart_data = filtered_crime_history['monthly_crime_count'].resample('QE').sum()
                elif time_scale == 'Annual':
                    chart_data = filtered_crime_history['monthly_crime_count'].resample('YE').sum()

                st.line_chart(chart_data, height=300)

            with crime_c2:
                valid_crime_cols = [col for col in COMMUNITY_SAFETY_CRIMES if col in filtered_crime_history.columns]
                crime_types_data = filtered_crime_history[valid_crime_cols].sum().reset_index()
                crime_types_data.columns = ['Crime Type', 'Total Incidents']
                crime_types_data = crime_types_data.sort_values(by='Total Incidents', ascending=False)
                st.write(f"**Incidents by Type ({selected_crime_year})**")
                st.dataframe(crime_types_data, hide_index=True, height=270)
        else:
            st.info("No detailed crime history available for this neighbourhood.")

    # Air Quality Trend Container
    with st.container(border=True):
        st.subheader(f"Air Quality Trend ({all_years[0]} - {all_years[-1]})")

        # Query MASTER GDF directly for history
        air_quality_history = st.session_state['master_gdf'][
            st.session_state['master_gdf']['area_code'] == lsoa_code
            ].copy()

        if not air_quality_history.empty:
            air_quality_history = air_quality_history.dropna(subset=['year'])
            air_quality_history['Year'] = air_quality_history['year'].astype(int).astype(str)
            air_quality_history = air_quality_history.set_index('Year')
            chart_data = air_quality_history[['no2_mean_concentration', 'pm25_mean_concentration']].rename(columns={
                'no2_mean_concentration': 'NO₂ (µg/m³)',
                'pm25_mean_concentration': 'PM₂.₅ (µg/m³)'
            })

            chart_data = chart_data.dropna(how='all')

            if not chart_data.empty:
                st.line_chart(chart_data)
                with st.expander("View Air Quality Data Table"):
                    st.dataframe(chart_data.style.format("{:.1f}"))
            else:
                st.info("No air quality data available for this area.")
        else:
            st.info("No historical air quality data available to plot.")

    # Healthcare Details Container
    with st.container(border=True):
        st.subheader("Healthcare Details")
        dist_val = lsoa_data.get('avg_distance_to_gp_km')
        dist_str = f"{dist_val:.1f} km" if pd.notna(dist_val) else "Unknown"
        st.caption(
            f"Details of the 3 nearest GP practices. Average distance from this neighbourhood is **{dist_str}**.")

        gp_data = []
        gp_org_codes = []
        for i in range(1, 4):
            # FIX: Updated column name from 'name' to 'gp_name' as per process_healthcare_data.py pivot
            name = lsoa_data.get(f'gp_{i}_gp_name')
            org_code = lsoa_data.get(f'gp_{i}_org_code')

            if name and pd.notna(name) and name != 'N/A' and org_code:
                gp_data.append({
                    "GP Practice Name": name,
                    "Patient Satisfaction": lsoa_data.get(f'gp_{i}_satisfaction', np.nan),
                    "org_code": org_code
                })
                gp_org_codes.append(org_code)

        if gp_data:
            gp_df = pd.DataFrame(gp_data)
            gp_df.sort_values(by="Patient Satisfaction", ascending=False, inplace=True)
            st.dataframe(
                gp_df.drop(columns=['org_code']),
                column_config={"Patient Satisfaction": st.column_config.ProgressColumn(
                    "Patient Satisfaction (%)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                )},
                hide_index=True
            )

            # Historical GP Satisfaction Table
            st.markdown("##### Historical Patient Satisfaction (Table)")
            if not gp_historical_df.empty:
                unique_org_codes = list(set(gp_org_codes))
                chart_data = gp_historical_df[gp_historical_df['org_code'].isin(unique_org_codes)].copy()

                if not chart_data.empty:
                    if chart_data['satisfaction_pct'].max() <= 1:
                        chart_data['satisfaction_pct'] *= 100
                    gp_names_df = gp_df[['GP Practice Name', 'org_code']].drop_duplicates()
                    chart_data = chart_data.merge(gp_names_df, on='org_code', how='left')

                    table = chart_data.pivot_table(
                        index='GP Practice Name',
                        columns='year',
                        values='satisfaction_pct',
                        aggfunc='mean'
                    ).round(1)

                    table = table.reindex(sorted(table.columns), axis=1)
                    st.dataframe(
                        table,
                        width='stretch',
                        column_config={year: st.column_config.NumberColumn(f"{year}", format="%.1f%%")
                                       for year in table.columns},
                        height=250
                    )
                else:
                    st.info("No historical satisfaction data found for these GP practices.")
            else:
                st.info("Historical GP satisfaction data is loading or not available.")
        else:
            st.info("No GP data available for this neighbourhood.")

    # Childcare Details Container
    with st.container(border=True):
        st.subheader("Childcare Details")
        dist_val = lsoa_data.get('avg_distance_to_childcare_km')
        dist_str = f"{dist_val:.1f} km" if pd.notna(dist_val) else "Unknown"
        st.caption(
            f"Details of the 3 nearest childcare providers. Average distance from this neighbourhood is **{dist_str}**.")

        childcare_data = []
        childcare_urns = []
        for i in range(1, 4):
            name = lsoa_data.get(f'childcare_{i}_name')
            urn = lsoa_data.get(f'childcare_{i}_urn')
            if name and pd.notna(name) and name != 'N/A' and urn:
                # FIX: Updated to 'rating_str' as per process_childcare.py
                raw_rating = str(lsoa_data.get(f'childcare_{i}_rating_str', 'N/A')).lower()
                childcare_data.append({
                    "Provider Name": name,
                    "Quality": OFSTED_RATING_MAP.get(raw_rating, 'N/A'),
                    "Places": lsoa_data.get(f'childcare_{i}_places', 0),
                    "Distance (km)": lsoa_data.get(f'childcare_{i}_distance', 0),
                    # Corrected column name from distance_km
                    "provider_urn": urn
                })
                childcare_urns.append(urn)
        if childcare_data:
            childcare_df = pd.DataFrame(childcare_data)
            childcare_df.sort_values(by="Distance (km)", ascending=True, inplace=True)
            st.dataframe(
                childcare_df.drop(columns=['provider_urn']),
                column_config={
                    "Places": st.column_config.NumberColumn(
                        "Registered Places",
                        format="%d",
                    ),
                    "Distance (km)": st.column_config.NumberColumn(
                        "Distance (km)",
                        format="%.1f km",
                    )
                },
                hide_index=True
            )
            # Historical Childcare Table
            st.markdown("##### Historical Provider Data (Table)")
            if not childcare_historical_df.empty:
                hist_df = childcare_historical_df.rename(columns={'Provider URN': 'provider_urn'})
                unique_urns = list(set(childcare_urns))
                chart_data = hist_df[hist_df['provider_urn'].isin(unique_urns)].copy()
                if not chart_data.empty:
                    provider_names_df = childcare_df[['Provider Name', 'provider_urn']].drop_duplicates()
                    chart_data = chart_data.merge(provider_names_df, on='provider_urn', how='left',
                                                  suffixes=('_old', ''))

                    chart_data['Quality'] = chart_data['quality_rating'].astype(str).str.lower().map(
                        OFSTED_RATING_MAP).fillna('N/A')
                    chart_data['Places'] = pd.to_numeric(chart_data['places'], errors='coerce').fillna(0).astype(int)

                    st.markdown("**Historical Quality Rating**")
                    table_quality = chart_data.pivot_table(
                        index='Provider Name',
                        columns='year',
                        values='Quality',
                        aggfunc='first'
                    ).fillna('N/A')
                    table_quality = table_quality.reindex(sorted(table_quality.columns), axis=1)
                    st.dataframe(table_quality, width='stretch')

                    st.markdown("**Historical Registered Places**")
                    table_places = chart_data.pivot_table(
                        index='Provider Name',
                        columns='year',
                        values='Places',
                        aggfunc='mean'
                    ).fillna(0).astype(int)
                    table_places = table_places.reindex(sorted(table_places.columns), axis=1)
                    st.dataframe(table_places, width='stretch', column_config={
                        year: st.column_config.NumberColumn(f"{year}", format="%d")
                        for year in table_places.columns
                    })
                else:
                    st.info("No historical data found for these childcare providers.")
            else:
                st.info("Historical childcare data is loading or not available.")
        else:
            st.info("No childcare data available for this neighbourhood.")

    # Primary Education Details Container
    with st.container(border=True):
        st.subheader("Primary School Details")
        st.caption("Performance of primary schools associated with this neighbourhood (based on latest available data)")

        tab1, tab2 = st.tabs(["Schools near this Neighbourhood", "Schools near Neighbouring Neighbourhoods"])

        # Tab 1 - Primary Schools near this Neighbourhood/LSOA
        with tab1:
            lsoa_df = latest_lsoa_data[latest_lsoa_data["area_code"] == lsoa_code]
            lsoa_primary_schools_df, schools_near_this_lsoa_primary, lsoa_primary_urns_df = get_primary_schools(lsoa_df)

            if not lsoa_primary_schools_df.empty:
                st.dataframe(
                    lsoa_primary_schools_df,
                    hide_index=True,
                    column_config={
                        "% Meeting Standard": st.column_config.NumberColumn(format="%.0f%%"),
                        "Read Score": st.column_config.NumberColumn(format="%.1f"),
                        "Math Score": st.column_config.NumberColumn(format="%.1f")
                    }
                )
                display_primary_school_history(lsoa_primary_urns_df, primary_historical_df)
            else:
                st.info("No primary school data available for this neighbourhood.")

        # Tab 2 - Primary Schools near Neighbouring LSOAs
        with tab2:
            try:
                selected_lsoa_geom = lsoa_index_gdf_base[lsoa_index_gdf_base['area_code'] == lsoa_code].geometry.iloc[0]
                intersecting_lsoas_gdf = lsoa_index_gdf_base[lsoa_index_gdf_base.intersects(selected_lsoa_geom)]
                neighbouring_lsoas_gdf = intersecting_lsoas_gdf[intersecting_lsoas_gdf['area_code'] != lsoa_code]
                neighbouring_lsoa_codes = neighbouring_lsoas_gdf['area_code'].unique()

                if len(neighbouring_lsoa_codes) > 0:
                    neighbour_lsoa_data = latest_lsoa_data[latest_lsoa_data["area_code"].isin(neighbouring_lsoa_codes)]
                    n_primary_schools_df, _, n_primary_urns_df = get_primary_schools(neighbour_lsoa_data)
                    n_primary_schools_df_filtered = n_primary_schools_df[
                        ~n_primary_schools_df['School Name'].isin(schools_near_this_lsoa_primary)
                    ]
                    n_primary_urns_df_filtered = n_primary_urns_df[
                        n_primary_urns_df['School Name'].isin(n_primary_schools_df_filtered['School Name'])
                    ]

                    if not n_primary_schools_df_filtered.empty:
                        st.dataframe(
                            n_primary_schools_df_filtered,
                            hide_index=True,
                            column_config={
                                "% Meeting Standard": st.column_config.NumberColumn(format="%.0f%%"),
                                "Read Score": st.column_config.NumberColumn(format="%.1f"),
                                "Math Score": st.column_config.NumberColumn(format="%.1f")
                            }
                        )
                        display_primary_school_history(n_primary_urns_df_filtered, primary_historical_df)
                    else:
                        st.info("No unique primary school data found for neighbouring neighbourhoods.")
                else:
                    st.info("This neighbourhood has no neighbouring neighbourhoods in the dataset.")

            except Exception as e:
                st.error("Could not process neighbouring neighbourhood data.")

    # Secondary Education Details Container
    with st.container(border=True):
        st.subheader("Secondary School Details")
        st.caption(
            "Performance of secondary schools associated with this neighbourhood (based on latest available data)")

        tab1, tab2, tab3 = st.tabs(
            ["State Schools near this Neighbourhood", "State Schools near Neighbouring Neighbourhoods",
             "Independent Schools"])

        # Tab 1 - State Secondary Schools near this Neighbourhood/LSOA
        with tab1:
            if 'lsoa_df' not in locals():
                lsoa_df = latest_lsoa_data[latest_lsoa_data["area_code"] == lsoa_code]
            lsoa_schools_df, schools_near_this_lsoa_sec, lsoa_secondary_urns_df = get_secondary_schools(lsoa_df,
                                                                                                        school_type='state')

            if not lsoa_schools_df.empty:
                st.dataframe(
                    lsoa_schools_df.drop(columns=['Type']),
                    hide_index=True,
                    column_config={
                        "Progress 8": st.column_config.NumberColumn(format="%.2f"),
                        "Attainment 8": st.column_config.NumberColumn(format="%.1f")
                    }
                )
                display_secondary_school_history(lsoa_secondary_urns_df, secondary_historical_df)
            else:
                st.info("No state secondary school data available for this neighbourhood.")

        # Tab 2 - State Secondary Schools near Neighbouring LSOAs
        with tab2:
            try:
                if 'neighbouring_lsoa_codes' not in locals():
                    selected_lsoa_geom = \
                        lsoa_index_gdf_base[lsoa_index_gdf_base['area_code'] == lsoa_code].geometry.iloc[0]
                    intersecting_lsoas_gdf = lsoa_index_gdf_base[lsoa_index_gdf_base.intersects(selected_lsoa_geom)]
                    neighbouring_lsoas_gdf = intersecting_lsoas_gdf[intersecting_lsoas_gdf['area_code'] != lsoa_code]
                    neighbouring_lsoa_codes = neighbouring_lsoas_gdf['area_code'].unique()

                if 'neighbour_lsoa_data' not in locals():
                    if len(neighbouring_lsoa_codes) > 0:
                        neighbour_lsoa_data = latest_lsoa_data[
                            latest_lsoa_data["area_code"].isin(neighbouring_lsoa_codes)]
                    else:
                        neighbour_lsoa_data = pd.DataFrame()

                if not neighbour_lsoa_data.empty:
                    n_schools_df, _, n_secondary_urns_df = get_secondary_schools(neighbour_lsoa_data,
                                                                                 school_type='state')
                    n_schools_df_filtered = n_schools_df[
                        ~n_schools_df['School Name'].isin(schools_near_this_lsoa_sec)
                    ]
                    n_secondary_urns_df_filtered = n_secondary_urns_df[
                        n_secondary_urns_df['School Name'].isin(n_schools_df_filtered['School Name'])
                    ]

                    if not n_schools_df_filtered.empty:
                        st.dataframe(
                            n_schools_df_filtered.drop(columns=['Type']),
                            hide_index=True,
                            column_config={
                                "Progress 8": st.column_config.NumberColumn(format="%.2f"),
                                "Attainment 8": st.column_config.NumberColumn(format="%.1f")
                            }
                        )
                        display_secondary_school_history(n_secondary_urns_df_filtered, secondary_historical_df)
                    else:
                        st.info("No unique state secondary school data found for neighbouring neighbourhoods.")
                else:
                    st.info("This neighbourhood has no neighbouring neighbourhoods in the dataset.")
            except Exception as e:
                st.error(f"Could not process neighbouring neighbourhood data")

        # Tab 3 - Independent Secondary Schools
        with tab3:
            st.info("""
            Independent schools ('IND') are not required to publish the same performance data as state schools 
            and are not included in the Education Score. The schools listed below are located 
            near this neighbourhood or neighbouring neighbourhoods.
            """)

            ind_schools_this_lsoa_df, _, _ = get_secondary_schools(
                lsoa_df, school_type='independent'
            )

            if 'neighbour_lsoa_data' in locals() and not neighbour_lsoa_data.empty:
                ind_schools_neighbour_df, _, _ = get_secondary_schools(
                    neighbour_lsoa_data, school_type='independent'
                )
            else:
                ind_schools_neighbour_df = pd.DataFrame()

            all_ind_schools_df = pd.concat([ind_schools_this_lsoa_df, ind_schools_neighbour_df])
            if not all_ind_schools_df.empty:
                all_ind_schools_df.drop_duplicates(subset=["School Name"], inplace=True)
                all_ind_schools_df.sort_values(by="School Name", ascending=True, inplace=True)

                st.dataframe(
                    all_ind_schools_df[['School Name']],
                    hide_index=True,
                    width='stretch'
                )
            else:
                st.info("No independent secondary schools found near this neighbourhood or its neighbours.")

    # Deprivation Details Container
    with st.container(border=True):
        st.subheader("Deprivation Details (IMD 2019)")
        st.caption("""
        The Index of Multiple Deprivation (IMD) shows relative deprivation. 
        Areas are grouped into 10 'deciles', where **1 is the most deprived** 10% in England, 
        and **10 is the least deprived** 10%.
        """)
        overall_decile = lsoa_data.get('IMD_Decile')

        if pd.isna(overall_decile):
            st.metric(label="Overall Deprivation Decile", value="N/A")
            st.info("IMD data is not available for this area (it may be outside England).")
        else:
            st.metric(label="Overall Deprivation Decile", value=f"{int(overall_decile)} / 10")
            decile_data = {
                'Deprivation Type': ['Income', 'Employment', 'Health'],
                'Decile': [
                    lsoa_data.get('Income_Decile', 0),
                    lsoa_data.get('Employment_Decile', 0),
                    lsoa_data.get('Health_Decile', 0)
                ]
            }
            decile_df = pd.DataFrame(decile_data).fillna(0)
            st.write("**Deprivation Breakdown (Deciles)**")
            fig = px.bar(
                decile_df,
                x='Deprivation Type',
                y='Decile',
                color='Deprivation Type',
                text='Decile',
                height=300
            )
            fig.update_layout(yaxis_range=[0, 10.5])
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, width='stretch')

# View 2 - Ward Level
elif st.session_state.get("selected_ward_code"):
    ward_code = st.session_state.selected_ward_code
    ward_data_series = latest_ward_data[latest_ward_data["WD25CD"] == ward_code]

    if ward_data_series.empty:
        st.error(f"No data found for this Ward. Please select another area.")
        st.stop()

    ward_data = ward_data_series.iloc[0]

    st.subheader(f"Ward: *{ward_data['WD25NM']}*")
    st.caption(f"Data Focus: {TARGET_YEAR}")

    # Top Scoring Section
    st.markdown("### Thrive Index Score (Ward Average, 2024)")

    final_score_raw = ward_data.get('Final_CI_Score')
    if pd.isna(final_score_raw):
        st.warning("Composite Score not available for this area.")
    else:
        final_score = int(round(final_score_raw))
        m1, m2 = st.columns(2)
        with m1:
            with st.container(border=True):
                st.metric(label="Thrive Index Score", value=f"{final_score}/100")
        with m2:
            with st.container(border=True):
                pop_val = ward_data.get('population', 0)
                st.metric(label="Total Population", value=f"{int(pop_val):,}" if pd.notna(pop_val) else "N/A")

        with st.container(border=True):
            st.subheader("5 Core Pillars Breakdown (Average)")
            c1, c2, c3, c4, c5 = st.columns(5)


            def get_fmt_score(row, key):
                val = row.get(key)
                if pd.notna(val):
                    return f"{int(round(val))}/100"
                return "N/A"


            with c1:
                st.metric("Socio-Economic", get_fmt_score(ward_data, 'Socio-Economic_Deprivation_Score'))
            with c2:
                st.metric("Env. Safety", get_fmt_score(ward_data, 'Environmental_Safety_Score'))
            with c3:
                st.metric("Secondary Ed.", get_fmt_score(ward_data, 'Secondary_Education_Score'))
            with c4:
                st.metric("Primary Ed.", get_fmt_score(ward_data, 'Primary_Education_Score'))
            with c5:
                st.metric("Childcare", get_fmt_score(ward_data, 'Childcare_Quality_Score'))

    # House Price Trend Container
    with st.container(border=True):
        st.subheader("Median House Price Trend")
        latest_date_obj = st.session_state.get('latest_house_price_date')
        latest_period_label = latest_date_obj.strftime('%b %Y') if latest_date_obj else 'Latest'
        latest_price = ward_data.get('latest_median_house_price')

        with st.container(border=True):
            price_val = f"£{int(latest_price):,}" if pd.notna(latest_price) else "N/A"
            st.metric(label=f"Latest Avg. Price ({latest_period_label})", value=price_val)

        ward_history_all = load_house_price_history('ward')
        sw_history = load_house_price_history('sw')
        if ward_history_all is not None and not ward_history_all.empty:
            ward_history = ward_history_all[ward_history_all['area_code'] == ward_code].copy()

            if not ward_history.empty and sw_history is not None:
                ward_history_chart = ward_history[['date', 'median_house_price']].rename(
                    columns={'median_house_price': 'Ward'}
                )
                sw_history_chart = sw_history[['date', 'median_house_price']].rename(
                    columns={'median_house_price': 'South West Average'}
                )
                chart_data = pd.merge(ward_history_chart, sw_history_chart, on='date', how='outer')
                chart_data['date'] = pd.to_datetime(chart_data['date'])
                chart_data.set_index('date', inplace=True)
                st.line_chart(chart_data)
            else:
                st.info("No historical house price data available for this ward.")
        else:
            st.info("Historical house price data is not available.")

    # Crime Breakdown
    with st.container(border=True):
        st.subheader("Crime Breakdown (Time-Series)")
        master_gdf = st.session_state['master_gdf']
        lsoas_in_ward = master_gdf[master_gdf['WD25CD'] == ward_code]['area_code'].unique()

        ward_crime_history = monthly_crime_df[monthly_crime_df['area_code'].isin(lsoas_in_ward)].copy()

        if not ward_crime_history.empty:
            crime_years = ['All Time'] + sorted(ward_crime_history.index.year.unique(), reverse=True)
            selected_crime_year = st.selectbox(
                "Select Year to View:",
                crime_years,
                key="ward_crime_year_filter"
            )

            filtered_crime_history = ward_crime_history.copy()
            if selected_crime_year != 'All Time':
                filtered_crime_history = filtered_crime_history[
                    filtered_crime_history.index.year == selected_crime_year]

            crime_c1, crime_c2 = st.columns([2, 1.5])
            with crime_c1:
                if selected_crime_year == 'All Time':
                    time_scale = st.radio(
                        "Select Time Scale",
                        ('Monthly', 'Quarterly', 'Annual'),
                        horizontal=True,
                        key=f"ward_crime_scale"
                    )
                else:
                    time_scale = 'Monthly'

                ward_monthly_total = filtered_crime_history.groupby('period')['monthly_crime_count'].sum()

                if time_scale == 'Monthly':
                    chart_data = ward_monthly_total
                elif time_scale == 'Quarterly':
                    chart_data = ward_monthly_total.resample('QE').sum()
                elif time_scale == 'Annual':
                    chart_data = ward_monthly_total.resample('YE').sum()

                st.line_chart(chart_data, height=300)

            with crime_c2:
                valid_crime_cols = [col for col in COMMUNITY_SAFETY_CRIMES if col in filtered_crime_history.columns]
                crime_types_data = filtered_crime_history[valid_crime_cols].sum().reset_index()
                crime_types_data.columns = ['Crime Type', 'Total Incidents']
                crime_types_data = crime_types_data.sort_values(by='Total Incidents', ascending=False)
                st.write(f"**Total Incidents by Type ({selected_crime_year})**")
                st.dataframe(crime_types_data, hide_index=True, height=270)
        else:
            st.info("No detailed crime history available for this ward.")

    # Air Quality Trend Container
    with st.container(border=True):
        st.subheader(f"Air Quality Trend (Ward Average, {all_years[0]} - {all_years[-1]})")

        master_gdf = st.session_state['master_gdf']
        lsoas_in_ward_codes = master_gdf[master_gdf['WD25CD'] == ward_code]['area_code'].unique()

        air_quality_history_lsoas = master_gdf[
            master_gdf['area_code'].isin(lsoas_in_ward_codes)
        ].copy()

        if not air_quality_history_lsoas.empty:
            air_quality_history = air_quality_history_lsoas.groupby('year')[[
                'no2_mean_concentration', 'pm25_mean_concentration'
            ]].mean()

            air_quality_history['Year'] = air_quality_history.index.astype(str)
            air_quality_history = air_quality_history.set_index('Year')

            chart_data = air_quality_history[['no2_mean_concentration', 'pm25_mean_concentration']].rename(columns={
                'no2_mean_concentration': 'NO₂ (µg/m³)',
                'pm25_mean_concentration': 'PM₂.₅ (µg/m³)'
            })

            chart_data = chart_data.dropna(how='all')

            if not chart_data.empty:
                st.line_chart(chart_data)
                with st.expander("View Air Quality Data Table"):
                    st.dataframe(chart_data.style.format("{:.1f}"))
            else:
                st.info("No air quality data available.")
        else:
            st.info("No historical air quality data available to plot.")

    # Healthcare Details Container
    with st.container(border=True):
        st.subheader("Healthcare Details")
        dist_val = ward_data.get('avg_distance_to_gp_km')
        dist_str = f"{dist_val:.1f} km" if pd.notna(dist_val) else "Unknown"
        st.caption(
            f"Unique GP practices associated with this ward. The average distance to a GP across the ward is **{dist_str}**.")

        neighbourhoods_in_ward_df = latest_lsoa_data[latest_lsoa_data["WD25CD"] == ward_code]
        all_gps = []
        gp_org_codes = []
        for _, lsoa_row in neighbourhoods_in_ward_df.iterrows():
            for i in range(1, 4):
                name = lsoa_row.get(f'gp_{i}_gp_name')
                org_code = lsoa_row.get(f'gp_{i}_org_code')
                if name and pd.notna(name) and name != 'N/A' and org_code:
                    all_gps.append({
                        "GP Practice Name": name,
                        "Patient Satisfaction": lsoa_row.get(f'gp_{i}_satisfaction', np.nan),
                        "org_code": org_code
                    })
                    gp_org_codes.append(org_code)

        if all_gps:
            ward_gp_df = pd.DataFrame(all_gps)
            ward_gp_df.drop_duplicates(subset=["GP Practice Name"], inplace=True)
            ward_gp_df.sort_values(by="Patient Satisfaction", ascending=False, inplace=True)
            st.dataframe(
                ward_gp_df.drop(columns=['org_code']),
                column_config={"Patient Satisfaction": st.column_config.ProgressColumn(
                    "Patient Satisfaction (%)",
                    format="%.1f%%",
                    min_value=0,
                    max_value=100,
                )},
                hide_index=True
            )

            st.markdown("##### Historical Patient Satisfaction (Table)")
            if not gp_historical_df.empty:
                unique_org_codes = list(set(gp_org_codes))
                chart_data = gp_historical_df[gp_historical_df['org_code'].isin(unique_org_codes)].copy()

                if not chart_data.empty:
                    if chart_data['satisfaction_pct'].max() <= 1:
                        chart_data['satisfaction_pct'] *= 100
                    gp_names_df = ward_gp_df[['GP Practice Name', 'org_code']].drop_duplicates()
                    chart_data = chart_data.merge(gp_names_df, on='org_code', how='left')

                    table = chart_data.pivot_table(
                        index='GP Practice Name',
                        columns='year',
                        values='satisfaction_pct',
                        aggfunc='mean'
                    ).round(1)

                    table = table.reindex(sorted(table.columns), axis=1)
                    st.dataframe(
                        table,
                        width='stretch',
                        column_config={year: st.column_config.NumberColumn(f"{year}", format="%.1f%%")
                                       for year in table.columns},
                        height=200
                    )
                else:
                    st.info("No historical satisfaction data found for these GP practices.")
            else:
                st.info("Historical GP satisfaction data is loading or not available.")
        else:
            st.info("No GP data available for this ward.")

    # Childcare Details Container
    with st.container(border=True):
        st.subheader("Childcare Details")
        dist_val = ward_data.get('avg_distance_to_childcare_km')
        dist_str = f"{dist_val:.1f} km" if pd.notna(dist_val) else "Unknown"
        st.caption(
            f"Unique childcare providers associated with this ward. The average distance to a provider across the ward is **{dist_str}**.")

        neighbourhoods_in_ward_df = latest_lsoa_data[latest_lsoa_data["WD25CD"] == ward_code]
        all_childcare = []
        childcare_urns = []
        for _, lsoa_row in neighbourhoods_in_ward_df.iterrows():
            for i in range(1, 4):
                name = lsoa_row.get(f'childcare_{i}_name')
                urn = lsoa_row.get(f'childcare_{i}_urn')
                if name and pd.notna(name) and name != 'N/A' and urn:
                    # FIX: Correct column name
                    raw_rating = str(lsoa_row.get(f'childcare_{i}_rating_str', 'N/A')).lower()
                    all_childcare.append({
                        "Provider Name": name,
                        "Quality": OFSTED_RATING_MAP.get(raw_rating, 'N/A'),
                        "Quality_Sort": raw_rating,
                        "Places": lsoa_row.get(f'childcare_{i}_places', 0),
                        "provider_urn": urn
                    })
                    childcare_urns.append(urn)
        if all_childcare:
            ward_childcare_df = pd.DataFrame(all_childcare)
            ward_childcare_df.drop_duplicates(subset=["Provider Name"], inplace=True)
            ward_childcare_df.sort_values(by="Quality_Sort", ascending=True, inplace=True)
            st.dataframe(
                ward_childcare_df.drop(columns=['Quality_Sort', 'provider_urn']),
                column_config={
                    "Places": st.column_config.NumberColumn(
                        "Registered Places",
                        format="%d",
                    )
                },
                hide_index=True
            )

            st.markdown("##### Historical Provider Data (Table)")
            if not childcare_historical_df.empty:
                hist_df = childcare_historical_df.rename(columns={'Provider URN': 'provider_urn'})
                unique_urns = list(set(childcare_urns))
                chart_data = hist_df[hist_df['provider_urn'].isin(unique_urns)].copy()
                if not chart_data.empty:
                    provider_names_df = ward_childcare_df[['Provider Name', 'provider_urn']].drop_duplicates()
                    chart_data = chart_data.merge(provider_names_df, on='provider_urn', how='left',
                                                  suffixes=('_old', ''))

                    chart_data['Quality'] = chart_data['quality_rating'].astype(str).str.lower().map(
                        OFSTED_RATING_MAP).fillna('N/A')
                    chart_data['Places'] = pd.to_numeric(chart_data['places'], errors='coerce').fillna(0).astype(int)

                    st.markdown("**Historical Quality Rating**")
                    table_quality = chart_data.pivot_table(
                        index='Provider Name',
                        columns='year',
                        values='Quality',
                        aggfunc='first'
                    ).fillna('N/A')
                    table_quality = table_quality.reindex(sorted(table_quality.columns), axis=1)
                    st.dataframe(table_quality, width='stretch')

                    st.markdown("**Historical Registered Places**")
                    table_places = chart_data.pivot_table(
                        index='Provider Name',
                        columns='year',
                        values='Places',
                        aggfunc='mean'
                    ).fillna(0).astype(int)
                    table_places = table_places.reindex(sorted(table_places.columns), axis=1)
                    st.dataframe(table_places, width='stretch', column_config={
                        year: st.column_config.NumberColumn(f"{year}", format="%d")
                        for year in table_places.columns
                    })
                else:
                    st.info("No historical data found for these childcare providers.")
            else:
                st.info("Historical childcare data is loading or not available.")
        else:
            st.info("No childcare data available for this ward.")

    # Primary Education Details Container - Ward
    with st.container(border=True):
        st.subheader("Primary School Details")
        st.caption("Performance of primary schools associated with this ward (based on latest available FINAL data)")

        tab1, tab2 = st.tabs(["Schools in this Ward", "Schools in Neighbouring Wards"])

        # Tab 1 - Primary Schools in this Ward
        with tab1:
            neighbourhoods_in_ward_df = latest_lsoa_data[latest_lsoa_data["WD25CD"] == ward_code]
            ward_primary_schools_df, schools_in_this_ward_primary, ward_primary_urns_df = get_primary_schools(
                neighbourhoods_in_ward_df)

            if not ward_primary_schools_df.empty:
                st.dataframe(
                    ward_primary_schools_df,
                    hide_index=True,
                    column_config={
                        "% Meeting Standard": st.column_config.NumberColumn(format="%.0f%%"),
                        "Read Score": st.column_config.NumberColumn(format="%.1f"),
                        "Math Score": st.column_config.NumberColumn(format="%.1f")
                    }
                )
                display_primary_school_history(ward_primary_urns_df, primary_historical_df)
            else:
                st.info("No primary school data available for this ward.")

        # Tab 2 - Primary Schools in Neighbouring Wards
        with tab2:
            try:
                selected_ward_geom = ward_gdf[ward_gdf['WD25CD'] == ward_code].geometry.iloc[0]
                intersecting_wards_gdf = ward_gdf[ward_gdf.intersects(selected_ward_geom)]
                neighbouring_wards_gdf = intersecting_wards_gdf[intersecting_wards_gdf['WD25CD'] != ward_code]
                neighbouring_ward_codes = neighbouring_wards_gdf['WD25CD'].unique()

                if len(neighbouring_ward_codes) > 0:
                    neighbour_lsoa_data = latest_lsoa_data[latest_lsoa_data["WD25CD"].isin(neighbouring_ward_codes)]
                    n_primary_schools_df, _, n_primary_urns_df = get_primary_schools(neighbour_lsoa_data)
                    n_primary_schools_df_filtered = n_primary_schools_df[
                        ~n_primary_schools_df['School Name'].isin(schools_in_this_ward_primary)
                    ]
                    n_primary_urns_df_filtered = n_primary_urns_df[
                        n_primary_urns_df['School Name'].isin(n_primary_schools_df_filtered['School Name'])
                    ]

                    if not n_primary_schools_df_filtered.empty:
                        st.dataframe(
                            n_primary_schools_df_filtered,
                            hide_index=True,
                            column_config={
                                "% Meeting Standard": st.column_config.NumberColumn(format="%.0f%%"),
                                "Read Score": st.column_config.NumberColumn(format="%.1f"),
                                "Math Score": st.column_config.NumberColumn(format="%.1f")
                            }
                        )
                        display_primary_school_history(n_primary_urns_df_filtered, primary_historical_df)
                    else:
                        st.info("No unique primary school data found for neighbouring wards.")
                else:
                    st.info("This ward has no neighbouring wards in the dataset.")

            except Exception as e:
                st.error("Could not process neighbouring ward data.")

    # Secondary Education Details Container - Ward
    with st.container(border=True):
        st.subheader("Secondary School Details")
        st.caption("Performance of secondary schools associated with this ward (based on latest available FINAL data)")

        tab1, tab2, tab3 = st.tabs(
            ["State Schools in this Ward", "State Schools in Neighbouring Wards", "Independent Schools"])

        # Tab 1 - State Secondary Schools in this Ward
        with tab1:
            if 'neighbourhoods_in_ward_df' not in locals():
                neighbourhoods_in_ward_df = latest_lsoa_data[latest_lsoa_data["WD25CD"] == ward_code]
            ward_schools_df, schools_in_this_ward_sec, ward_secondary_urns_df = get_secondary_schools(
                neighbourhoods_in_ward_df,
                school_type='state')

            if not ward_schools_df.empty:
                st.dataframe(
                    ward_schools_df.drop(columns=['Type']),
                    hide_index=True,
                    column_config={
                        "Progress 8": st.column_config.NumberColumn(format="%.2f"),
                        "Attainment 8": st.column_config.NumberColumn(format="%.1f")
                    }
                )
                display_secondary_school_history(ward_secondary_urns_df, secondary_historical_df)
            else:
                st.info("No state secondary school data available for this ward.")

        # Tab 2 - State Secondary Schools in Neighbouring Wards
        with tab2:
            try:
                if 'neighbouring_ward_codes' not in locals():
                    selected_ward_geom = ward_gdf[ward_gdf['WD25CD'] == ward_code].geometry.iloc[0]
                    intersecting_wards_gdf = ward_gdf[ward_gdf.intersects(selected_ward_geom)]
                    neighbouring_wards_gdf = intersecting_wards_gdf[intersecting_wards_gdf['WD25CD'] != ward_code]
                    neighbouring_ward_codes = neighbouring_wards_gdf['WD25CD'].unique()

                if 'neighbour_lsoa_data' not in locals():
                    if len(neighbouring_ward_codes) > 0:
                        neighbour_lsoa_data = latest_lsoa_data[latest_lsoa_data["WD25CD"].isin(neighbouring_ward_codes)]
                    else:
                        neighbour_lsoa_data = pd.DataFrame()

                if not neighbour_lsoa_data.empty:
                    n_schools_df, _, n_secondary_urns_df = get_secondary_schools(neighbour_lsoa_data,
                                                                                 school_type='state')

                    n_schools_df_filtered = n_schools_df[
                        ~n_schools_df['School Name'].isin(schools_in_this_ward_sec)
                    ]
                    n_secondary_urns_df_filtered = n_secondary_urns_df[
                        n_secondary_urns_df['School Name'].isin(n_schools_df_filtered['School Name'])
                    ]

                    if not n_schools_df_filtered.empty:
                        st.dataframe(
                            n_schools_df_filtered.drop(columns=['Type']),
                            hide_index=True,
                            column_config={
                                "Progress 8": st.column_config.NumberColumn(format="%.2f"),
                                "Attainment 8": st.column_config.NumberColumn(format="%.1f")
                            }
                        )
                        display_secondary_school_history(n_secondary_urns_df_filtered, secondary_historical_df)
                    else:
                        st.info("No unique state secondary school data found for neighbouring wards.")
                else:
                    st.info("This ward has no neighbouring wards in the dataset.")
            except Exception as e:
                st.error(f"Could not process neighbouring ward data: {e}")

        # Tab 3 - Independent Secondary Schools
        with tab3:
            st.info("""
            Independent schools ('IND') are not required to publish the same performance data as state schools 
            and are not included in the Education Score. The schools listed below are located 
            in this ward or neighbouring wards.
            """)

            ind_schools_this_ward_df, _, _ = get_secondary_schools(
                neighbourhoods_in_ward_df, school_type='independent'
            )

            if 'neighbour_lsoa_data' in locals() and not neighbour_lsoa_data.empty:
                ind_schools_neighbour_df, _, _ = get_secondary_schools(
                    neighbour_lsoa_data, school_type='independent'
                )
            else:
                ind_schools_neighbour_df = pd.DataFrame()

            all_ind_schools_df = pd.concat([ind_schools_this_ward_df, ind_schools_neighbour_df])
            if not all_ind_schools_df.empty:
                all_ind_schools_df.drop_duplicates(subset=["School Name"], inplace=True)
                all_ind_schools_df.sort_values(by="School Name", ascending=True, inplace=True)

                st.dataframe(
                    all_ind_schools_df[['School Name']],
                    hide_index=True,
                    width='stretch'
                )
            else:
                st.info("No independent secondary schools found in this ward or neighbouring wards.")

    # Deprivation Details Container
    with st.container(border=True):
        st.subheader("Deprivation Details (IMD 2019 - Ward Average)")
        st.caption("""
        The Index of Multiple Deprivation (IMD) shows relative deprivation. 
        Areas are grouped into 10 'deciles', where **1 is the most deprived** 10% in England, 
        and **10 is the least deprived** 10%. Values shown here are averages for the neighbourhoods in this ward.
        """)
        # Deciles calculated in utils.py
        overall_decile = ward_data.get('IMD_Decile')
        if pd.isna(overall_decile) or overall_decile == 0:
            st.metric(label="Average Overall Deprivation Decile", value="N/A")
            st.info("IMD data is not available for this area (it may be outside England).")
        else:
            st.metric(label="Average Overall Deprivation Decile", value=f"{int(overall_decile)} / 10")
            decile_data = {
                'Deprivation Type': ['Income', 'Employment', 'Health'],
                'Decile': [
                    ward_data.get('Income_Decile', 0),
                    ward_data.get('Employment_Decile', 0),
                    ward_data.get('Health_Decile', 0)
                ]
            }
            decile_df = pd.DataFrame(decile_data).fillna(0)
            st.write("**Average Deprivation Breakdown (Deciles)**")
            fig = px.bar(
                decile_df,
                x='Deprivation Type',
                y='Decile',
                color='Deprivation Type',
                text='Decile',
                height=300
            )
            fig.update_layout(yaxis_range=[0, 10.5])
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, width='stretch')

    # Neighbourhood Comparison Table for Selected Ward
    with st.container(border=True):
        st.subheader(f"Neighbourhoods in this Ward")

        # Default to 2024 if available, else last year in list
        default_idx = len(all_years) - 1
        if TARGET_YEAR in all_years:
            default_idx = all_years.index(TARGET_YEAR)

        year_for_table = st.selectbox(
            "Select Year to Display:",
            options=all_years,
            index=default_idx,
            key="table_year_selector"
        )
        # This contains the non-imputed raw metrics
        master_gdf = st.session_state['master_gdf']
        lsoa_for_table = master_gdf[master_gdf['year'] == year_for_table].copy()

        # Also need to merge in the 2024 Scores if available for reference
        if year_for_table == TARGET_YEAR:
            neighbourhoods_in_ward_df = latest_lsoa_data[latest_lsoa_data["WD25CD"] == ward_code]
        else:
            # Just raw data for other years
            neighbourhoods_in_ward_df = lsoa_for_table[lsoa_for_table["WD25CD"] == ward_code]

        cols_to_show = [
            'display_name',
            'Final_CI_Score',
            # Pillar Scores
            'Socio-Economic_Deprivation_Score',
            'Environmental_Safety_Score',
            'Secondary_Education_Score',
            'Primary_Education_Score',
            'Childcare_Quality_Score',
            # Context Data
            'latest_median_house_price',
            'population',
            'greenspace_percentage',
            'no2_mean_concentration',
            'pm25_mean_concentration',
            'crime_rate_per_1000',
            'avg_distance_to_gp_km',
            'avg_gp_satisfaction',
            'avg_progress_8',
            'avg_attainment_8',
            'avg_ks2_pass_rate',
            'avg_primary_scaled_score',
            'avg_distance_to_childcare_km',
            'avg_childcare_quality_score',
            'total_childcare_places_nearby'
        ]

        # Filter cols that exist
        cols_to_show = [col for col in cols_to_show if col in neighbourhoods_in_ward_df.columns]

        rename_map = {
            'display_name': 'Neighbourhood',
            'Final_CI_Score': 'Thrive Score',
            # Pillar Renames
            'Socio-Economic_Deprivation_Score': 'Socio-Ec Score',
            'Environmental_Safety_Score': 'Env Safety Score',
            'Secondary_Education_Score': 'Sec Ed Score',
            'Primary_Education_Score': 'Pri Ed Score',
            'Childcare_Quality_Score': 'Childcare Score',
            # Context Renames
            'latest_median_house_price': 'Latest House Price',
            'population': 'Population',
            'greenspace_percentage': 'Greenspace %',
            'no2_mean_concentration': 'NO₂ (µg/m³)',
            'pm25_mean_concentration': 'PM₂.₅ (µg/m³)',
            'crime_rate_per_1000': 'Crime Rate (per 1k)',
            'avg_distance_to_gp_km': 'Avg. GP Dist. (km)',
            'avg_gp_satisfaction': 'GP Satisfaction %',
            'avg_progress_8': 'Progress 8',
            'avg_attainment_8': 'Attainment 8',
            'avg_ks2_pass_rate': 'KS2 Pass %',
            'avg_primary_scaled_score': 'KS2 Scaled Score',
            'avg_distance_to_childcare_km': 'Childcare Dist. (km)',
            'avg_childcare_quality_score': 'Childcare Quality (1-4)',
            'total_childcare_places_nearby': 'Childcare Places'
        }

        df_to_display = neighbourhoods_in_ward_df[cols_to_show].rename(columns=rename_map)

        st.dataframe(
            df_to_display,
            hide_index=True,
            column_config={
                "Thrive Score": st.column_config.NumberColumn(format="%.0f"),
                "Socio-Ec Score": st.column_config.NumberColumn(format="%.0f"),
                "Env Safety Score": st.column_config.NumberColumn(format="%.0f"),
                "Sec Ed Score": st.column_config.NumberColumn(format="%.0f"),
                "Pri Ed Score": st.column_config.NumberColumn(format="%.0f"),
                "Childcare Score": st.column_config.NumberColumn(format="%.0f"),
                "Latest House Price": st.column_config.NumberColumn(format="£%d"),
                "Population": st.column_config.NumberColumn(format="%d"),
                "Greenspace %": st.column_config.NumberColumn(format="%.1f%%"),
                "NO₂ (µg/m³)": st.column_config.NumberColumn(format="%.1f"),
                "PM₂.₅ (µg/m³)": st.column_config.NumberColumn(format="%.1f"),
                "Crime Rate (per 1k)": st.column_config.NumberColumn(format="%.1f"),
                "Avg. GP Dist. (km)": st.column_config.NumberColumn(format="%.1f km"),
                "GP Satisfaction %": st.column_config.NumberColumn(format="%.0f%%"),
                "Progress 8": st.column_config.NumberColumn(format="%.2f"),
                "Attainment 8": st.column_config.NumberColumn(format="%.1f"),
                "KS2 Pass %": st.column_config.NumberColumn(format="%.0f%%"),
                "KS2 Scaled Score": st.column_config.NumberColumn(format="%.1f"),
                "Childcare Dist. (km)": st.column_config.NumberColumn(format="%.1f km"),
                "Childcare Quality (1-4)": st.column_config.NumberColumn(format="%.1f"),
                "Childcare Places": st.column_config.NumberColumn(format="%d"),
            }
        )

# View 3 - No Selection
else:
    st.info("Use the dropdowns above to select a Ward to begin.")

# Footer for Sources & Licensing
with st.expander("Sources & Licensing", expanded=False):
    st.markdown(generate_attribution_markdown())