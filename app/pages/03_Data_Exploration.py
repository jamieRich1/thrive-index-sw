# Imports
import streamlit as st
import pandas as pd
import plotly.express as px
from utils import (
    load_attribute_data,
    load_political_boundaries
)
from licensing import generate_attribution_markdown

# Page Config
st.set_page_config(
    page_title="Data Explorer - Thrive Index SW",
    layout="wide"
)

# Load Data into Session State
lad_gdf, ward_gdf = load_political_boundaries()
lsoa_to_display = load_attribute_data(year_filter=2024)
ward_to_display = lsoa_to_display.groupby(['WD25CD', 'WD25NM', 'LAD25CD'], as_index=False).mean(numeric_only=True)

# Hardcode Year to 2024
TARGET_YEAR = 2024

# Indicator Mappings
INDICATORS = {
    # Top Level Scores
    'Thrive Index Score': {'col': 'Final_CI_Score', 'ascending': False, 'desc': 'The overall composite score.'},
    'Socio-Economic Score': {'col': 'Socio-Economic_Deprivation_Score', 'ascending': False,
                             'desc': 'Score based on deprivation & crime.'},
    'Environmental Safety Score': {'col': 'Environmental_Safety_Score', 'ascending': False,
                                   'desc': 'Score based on air quality & pollution.'},
    'Secondary Education Score': {'col': 'Secondary_Education_Score', 'ascending': False,
                                  'desc': 'Score based on secondary school performance.'},
    'Primary Education Score': {'col': 'Primary_Education_Score', 'ascending': False,
                                'desc': 'Score based on primary school performance.'},
    'Childcare Quality Score': {'col': 'Childcare_Quality_Score', 'ascending': False,
                                'desc': 'Score based on childcare quality & access.'},

    # Detailed Metrics
    'Greenspace Percentage (%)': {'col': 'greenspace_percentage', 'ascending': False,
                                  'desc': 'Percentage of area covered by greenspace.'},
    'NO₂ Mean Concentration (µg/m³)': {'col': 'no2_mean_concentration', 'ascending': True,
                                       'desc': 'Average Nitrogen Dioxide level. Lower is better.'},
    'PM₂.₅ Mean Concentration (µg/m³)': {'col': 'pm25_mean_concentration', 'ascending': True,
                                         'desc': 'Average PM₂.₅ particulate level. Lower is better.'},

    'Crime Rate (per 1,000)': {'col': 'crime_rate_per_1000', 'ascending': True,
                               'desc': 'Crimes per 1,000 people. Lower is better.'},

    'Avg. Distance to GP (km)': {'col': 'avg_distance_to_gp_km', 'ascending': True,
                                 'desc': 'Average distance to nearest GPs. Lower is better.'},
    'Avg. GP Satisfaction (%)': {'col': 'avg_gp_satisfaction', 'ascending': False,
                                 'desc': 'Average patient satisfaction score. Higher is better.'},

    # Education
    'Avg. KS2 Pass Rate (%)': {'col': 'primary_pass_rate_weighted', 'ascending': False,
                               'desc': 'Weighted average % meeting expected standard at KS2.'},
    'Avg. Primary Scaled Score': {'col': 'primary_read_score_weighted', 'ascending': False,
                                  'desc': 'Weighted average reading score at KS2 (proxy for overall scaled).'},

    'Avg. Progress 8': {'col': 'secondary_progress_8_weighted', 'ascending': False,
                        'desc': 'Weighted Average Progress 8 score.'},
    'Avg. Attainment 8': {'col': 'secondary_attainment_8_weighted', 'ascending': False,
                          'desc': 'Weighted Average Attainment 8 score.'},

    'Avg. Distance to Childcare (km)': {'col': 'avg_distance_to_childcare_km', 'ascending': True,
                                        'desc': 'Average distance to nearest childcare. Lower is better.'},
    'Avg. Childcare Quality': {'col': 'avg_childcare_quality_score', 'ascending': False,
                               'desc': 'Average Ofsted rating score (4=Outstanding). Higher is better.'},
    'Total Childcare Places Nearby': {'col': 'total_childcare_places_nearby', 'ascending': False,
                                      'desc': 'Sum of places at nearest providers. Higher is better.'},

    'Population': {'col': 'population', 'ascending': False, 'desc': 'Estimated resident population.'},
    'Latest Median House Price': {
        'col': 'latest_median_house_price',
        'ascending': True,
        'desc': 'Latest available median house price. Lower is more affordable.'
    },

    # IMD Deciles
    'IMD Decile (Overall)': {
        'col': 'IMD_Decile', 'ascending': True,
        'desc': 'Overall deprivation decile. 1 = most deprived 10% in SW, 10 = least deprived.'
    },
    'Income Decile': {
        'col': 'Income_Decile', 'ascending': True,
        'desc': 'Income deprivation decile. 1 = most deprived, 10 = least deprived.'
    },
    'Employment Decile': {
        'col': 'Employment_Decile', 'ascending': True,
        'desc': 'Employment deprivation decile. 1 = most deprived, 10 = least deprived.'
    },
    'Health Decile': {
        'col': 'Health_Decile', 'ascending': True,
        'desc': 'Health deprivation decile. 1 = most deprived, 10 = least deprived.'
    }
}
# Get list of names for dropdowns
INDICATOR_NAMES = list(INDICATORS.keys())

# Sidebar
st.sidebar.title("Data Explorer Controls")

mode = st.sidebar.radio(
    "Choose Exploration Mode",
    ("Rank Areas", "Compare Areas", "Explore Correlations", "View Distribution"),
    help="Select a mode to analyze the data in different ways."
)
st.sidebar.markdown("---")
st.sidebar.markdown("## Global Data Controls")
st.sidebar.info(f"Viewing Data for **{TARGET_YEAR}**")
st.sidebar.caption("This tool uses the final 2024 composite scores and data.")

# Data Retrieval
ward_stats_df_page = ward_to_display.copy()
lsoa_index_gdf_page = lsoa_to_display.copy()

if 'crime_rate_per_1000' not in lsoa_index_gdf_page.columns:
    if 'crime_count' in lsoa_index_gdf_page.columns and 'population' in lsoa_index_gdf_page.columns:
        # Calculate rate: (Count / Population) * 1000
        lsoa_index_gdf_page['crime_rate_per_1000'] = (
                                                             lsoa_index_gdf_page['crime_count'] / lsoa_index_gdf_page[
                                                         'population']
                                                     ) * 1000

cols_to_check = [
    'crime_rate_per_1000',
    'avg_gp_satisfaction',
    'avg_distance_to_gp_km',
    'avg_childcare_quality_score',
    'avg_distance_to_childcare_km',
    'total_childcare_places_nearby',
    'secondary_progress_8_weighted',
    'secondary_attainment_8_weighted',
    'primary_pass_rate_weighted',
    'primary_read_score_weighted'
]

# Identify missing columns in ward dataframe but present in LSOA dataframe
missing_ward_cols = [col for col in cols_to_check
                     if col not in ward_stats_df_page.columns and col in lsoa_index_gdf_page.columns]

if missing_ward_cols:
    ward_aggs = lsoa_index_gdf_page.groupby('WD25CD')[missing_ward_cols].mean(numeric_only=True).reset_index()
    ward_stats_df_page = ward_stats_df_page.merge(ward_aggs, on='WD25CD', how='left')

# LAD Code -> LAD Name lookup
lad_name_map = lad_gdf.set_index('lad_code')['lad_name']

# Add display names to Ward dataframe
if 'LAD25CD' in ward_stats_df_page.columns:
    ward_stats_df_page['lad_name'] = ward_stats_df_page['LAD25CD'].map(lad_name_map)
    ward_stats_df_page['ward_display_name'] = ward_stats_df_page['lad_name'] + " - " + ward_stats_df_page['WD25NM']

# Add display names and helper columns to LSOA dataframe
lsoa_index_gdf_page['ward_display_name'] = lsoa_index_gdf_page['LAD25NM'] + " - " + lsoa_index_gdf_page['WD25NM']
lsoa_index_gdf_page['neighbourhood_num'] = lsoa_index_gdf_page['display_name'].str.split(' - ').str[-1].str.replace(
    'Neighbourhood ', '')
lsoa_index_gdf_page['lsoa_dropdown_name'] = lsoa_index_gdf_page['WD25NM'] + " - " + lsoa_index_gdf_page[
    'neighbourhood_num']
lsoa_index_gdf_page['lsoa_full_display_name'] = lsoa_index_gdf_page['ward_display_name'] + " - " + lsoa_index_gdf_page[
    'neighbourhood_num']


# Filtering Helper
def filter_df_by_lad(df, selected_lad_name):
    """Filters a dataframe to include only areas within a selected Local Authority."""
    if selected_lad_name == "All of South West":
        return df
    try:
        selected_lad_code = lad_gdf.loc[lad_gdf['lad_name'] == selected_lad_name, 'lad_code'].iloc[0]
        if 'LAD25CD' in df.columns:
            return df[df["LAD25CD"] == selected_lad_code]
        else:
            st.error("Cannot filter by LAD: Dataframe missing 'LAD25CD'.")
            return df
    except (IndexError, KeyError):
        st.error("Error filtering by Local Authority.")
        return df


# Main Page Layout

# Rank Areas
if 'Rank' in mode:
    st.header(f"Rank Areas for {TARGET_YEAR}")
    st.markdown(
        f"Find the top or bottom areas across the South West based on a chosen indicator for **{TARGET_YEAR}**.")

    # Sidebar Controls
    rank_level = st.sidebar.radio("Area Level", ("Wards", "Neighbourhoods (LSOAs)"))
    rank_order = st.sidebar.radio("Rank Order", ("Top", "Bottom"),
                                  help="Top ranks best scores, Bottom ranks worst scores.")
    selected_indicator_name = st.sidebar.selectbox("Select Indicator to Rank By", INDICATOR_NAMES)
    lad_names = ["All of South West"] + sorted(lad_gdf['lad_name'].unique())
    selected_lad = st.sidebar.selectbox("Filter by Local Authority (Optional)", lad_names)
    num_to_show = st.sidebar.slider("Number of areas to show", 5, 50, 10)

    # Prepare data based on selections
    indicator = INDICATORS[selected_indicator_name]
    df_to_rank = ward_stats_df_page if rank_level == "Wards" else lsoa_index_gdf_page
    name_col = 'ward_display_name' if rank_level == "Wards" else 'lsoa_full_display_name'
    filtered_df = filter_df_by_lad(df_to_rank, selected_lad)

    # Sorting Based on Top/Bottom Selection
    sort_ascending = indicator['ascending']
    if rank_order == "Top":
        sort_ascending = indicator['ascending']
    else:
        sort_ascending = not indicator['ascending']

    filtered_df = filtered_df.dropna(subset=[name_col])

    # Check if column exists
    if indicator['col'] not in filtered_df.columns:
        st.error(f"Data for '{selected_indicator_name}' ({indicator['col']}) is missing.")
    else:
        ranked_df = filtered_df.sort_values(by=indicator['col'], ascending=sort_ascending).head(num_to_show)

        # Display Results
        st.subheader(f"{rank_order} {num_to_show} {rank_level} for {selected_indicator_name}")
        st.caption(
            f"Sorted by values in the **'{selected_indicator_name}'** column for **{TARGET_YEAR}**. Filtered by: **{selected_lad}**.")

        # Bar Chart
        PIXELS_PER_BAR, BASE_HEIGHT = 28, 250
        dynamic_height = min(max((len(ranked_df) * PIXELS_PER_BAR), BASE_HEIGHT), 800)
        fig = px.bar(
            ranked_df, x=indicator['col'], y=name_col, orientation='h',
            labels={indicator['col']: selected_indicator_name, name_col: rank_level[:-1]},
            height=dynamic_height
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        # Table View
        with st.expander("View Data Table", expanded=True):
            display_cols_dict = {props['col']: name for name, props in INDICATORS.items()}
            cols_to_display_ranked = [name_col] + [col for col in display_cols_dict.keys() if col in ranked_df.columns]

            st.dataframe(
                ranked_df[cols_to_display_ranked]
                .rename(columns=display_cols_dict)
                .rename(columns={name_col: rank_level[:-1]}),
                hide_index=True
            )

# Compare Areas
elif 'Compare' in mode:
    st.header(f"Compare Areas for {TARGET_YEAR}")
    st.markdown(f"Select multiple areas to compare their indicators side-by-side for **{TARGET_YEAR}**.")

    # Sidebar Controls
    compare_level = st.sidebar.radio("Select Area Level", ("Wards", "Neighbourhoods (LSOAs)"))
    lad_names = ["All of South West"] + sorted(lad_gdf['lad_name'].unique())
    selected_lad_compare = st.sidebar.selectbox("Filter Area List by Local Authority", lad_names, key="compare_lad")

    # Prepare data based on selections
    df_to_compare = ward_stats_df_page if compare_level == "Wards" else lsoa_index_gdf_page
    option_col = 'WD25NM' if compare_level == "Wards" else 'lsoa_dropdown_name'
    display_col = 'ward_display_name' if compare_level == "Wards" else 'lsoa_full_display_name'
    filtered_df = filter_df_by_lad(df_to_compare, selected_lad_compare)
    filtered_df = filtered_df.dropna(subset=[option_col, display_col])

    # Multiselect
    options = sorted(filtered_df[option_col].unique())
    selected_areas = st.sidebar.multiselect(f"Select up to 5 {compare_level} to Compare", options, max_selections=5)
    if not selected_areas:
        st.info("Please select areas from the sidebar to begin comparison.")
    else:
        # Filter dataframe to only the selected areas
        comparison_df = filtered_df[filtered_df[option_col].isin(selected_areas)]
        st.subheader(f"Comparison of {len(selected_areas)} {compare_level}")

        # Bar chart for comparing 0-100 index scores
        st.markdown("##### Overall Score Comparison")
        score_cols = {name: props['col'] for name, props in INDICATORS.items()
                      if 'Score' in name and name != 'Avg. Primary Scaled Score'}

        cols_to_plot_with_display = [display_col] + [col for col in score_cols.values() if col in comparison_df.columns]
        df_to_plot = comparison_df[cols_to_plot_with_display]

        melted_df = pd.melt(df_to_plot, id_vars=[display_col], var_name='Indicator', value_name='Score')
        col_to_name_map = {v: k for k, v in score_cols.items()}
        melted_df['Indicator'] = melted_df['Indicator'].map(col_to_name_map)

        fig = px.bar(
            melted_df, x=display_col, y='Score', color='Indicator', barmode='group',
            title=f"Indicator Scores by Area ({TARGET_YEAR}) (Higher is Better)",
            labels={display_col: compare_level[:-1], 'Score': 'Score (0-100)'}
        )
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

        # Table View
        with st.expander("View Detailed Data Table", expanded=True):
            # Also exclude from the table as requested
            display_cols_dict = {props['col']: name for name, props in INDICATORS.items() if
                                 name != 'Avg. Primary Scaled Score'}

            cols_to_display_compare = [display_col] + [col for col in display_cols_dict.keys() if
                                                       col in comparison_df.columns]
            cols_to_display_compare = [col for col in cols_to_display_compare if col in comparison_df.columns]

            # Transpose table for better comparison across areas
            table_df = (
                comparison_df[cols_to_display_compare]
                .set_index(display_col)
                .rename(columns=display_cols_dict)
                .T
            )
            st.dataframe(table_df)

# Explore Correlations
elif 'Correlations' in mode:
    st.header(f"Explore Correlations for {TARGET_YEAR}")
    st.markdown(
        f"Select two indicators to see if there is a relationship between them across all areas for **{TARGET_YEAR}**.")

    # Sidebar Controls
    corr_level = st.sidebar.selectbox("Select Area Level", ["Wards", "Neighbourhoods (LSOAs)"])
    lad_names = ["All of South West"] + sorted(lad_gdf['lad_name'].unique())
    selected_lad_corr = st.sidebar.selectbox("Filter by Local Authority (Optional)", lad_names, key="corr_lad")
    st.sidebar.markdown("Select two different indicators to plot against each other:")
    x_axis_name = st.sidebar.selectbox("Horizontal Axis (X)", INDICATOR_NAMES,
                                       index=INDICATOR_NAMES.index('Thrive Index Score'))
    y_axis_name = st.sidebar.selectbox("Vertical Axis (Y)", INDICATOR_NAMES,
                                       index=INDICATOR_NAMES.index('IMD Decile (Overall)'))

    # Prepare data
    df_to_corr = ward_stats_df_page if corr_level == "Wards" else lsoa_index_gdf_page
    name_col = 'ward_display_name' if corr_level == "Wards" else 'lsoa_full_display_name'
    filtered_df = filter_df_by_lad(df_to_corr, selected_lad_corr)

    if x_axis_name == y_axis_name:
        st.warning("Please select two different indicators.")
    else:
        x_col = INDICATORS[x_axis_name]['col']
        y_col = INDICATORS[y_axis_name]['col']

        # Check if columns exist in the dataframe
        if x_col not in filtered_df.columns or y_col not in filtered_df.columns:
            st.error(f"Error: One or both of the selected indicators are not available in the current dataset.")
            missing_cols = []
            if x_col not in filtered_df.columns: missing_cols.append(f"{x_axis_name} ({x_col})")
            if y_col not in filtered_df.columns: missing_cols.append(f"{y_axis_name} ({y_col})")
            st.write(f"Missing data for: {', '.join(missing_cols)}")
        else:
            # Scatter plot
            # Ensure numeric types for correlation plotting to avoid errors with OLS
            try:
                filtered_df[x_col] = pd.to_numeric(filtered_df[x_col], errors='coerce')
                filtered_df[y_col] = pd.to_numeric(filtered_df[y_col], errors='coerce')

                # Drop NaNs for the plot to avoid issues
                plot_df = filtered_df.dropna(subset=[x_col, y_col])

                if plot_df.empty:
                    st.warning(
                        f"No valid data points found for correlation between {x_axis_name} and {y_axis_name} after filtering.")
                else:
                    fig = px.scatter(
                        plot_df, x=x_col, y=y_col, hover_name=name_col,
                        title=f"Correlation between {x_axis_name} and {y_axis_name} ({TARGET_YEAR})",
                        labels={x_col: x_axis_name, y_col: y_axis_name},
                        trendline="ols",
                        trendline_color_override="red"  # Ensures line is red even in LSOA view
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Interpretation Guide
                    with st.expander("Interpretation Guide"):
                        st.info(
                            f"Each dot represents a {corr_level[:-1]}. This chart helps visualize potential patterns between the two selected indicators.")
                        st.markdown(
                            f"- **Upward sloping red line:** Suggests a *positive correlation* (as **{x_axis_name}** increases, **{y_axis_name}** tends to increase).")
                        st.markdown(
                            f"- **Downward sloping red line:** Suggests a *negative correlation* (as **{x_axis_name}** increases, **{y_axis_name}** tends to decrease).")
                        st.markdown("- **Scattered dots / flat line:** Suggests little to no correlation.")
            except Exception as e:
                st.error(f"An error occurred while creating the chart: {e}")

# View Distribution
elif 'Distribution' in mode:
    st.header(f"View Distribution for {TARGET_YEAR}")
    st.markdown(f"See how a specific area's score compares to the overall distribution for **{TARGET_YEAR}**.")

    # Sidebar Controls
    st.sidebar.markdown("##### Area Selection")
    dist_level = st.sidebar.radio("Select Area Level", ("Ward", "Neighbourhood (LSOA)"), key="dist_level_toggle")
    indicator_dist_name = st.sidebar.selectbox("Select Indicator", INDICATOR_NAMES, index=0)
    lad_names = sorted(lad_gdf['lad_name'].unique())
    selected_lad_name = st.sidebar.selectbox("1. Filter by Local Authority", lad_names)

    # Get wards within the selected LAD
    wards_in_lad_df = ward_gdf[
        ward_gdf["LAD25CD"] == lad_gdf.loc[lad_gdf['lad_name'] == selected_lad_name, 'lad_code'].iloc[0]]
    ward_options = sorted(wards_in_lad_df['WD25NM'].unique())
    selected_ward_name = st.sidebar.selectbox("2. Select Ward", ward_options)

    # Determine the selected area based on Ward/LSOA level
    selected_area_filter_val = None
    selected_area_display_val = None
    df = ward_stats_df_page if dist_level == "Ward" else lsoa_index_gdf_page

    if dist_level == "Ward":
        selected_area_filter_val = selected_ward_name
        if selected_area_filter_val:
            try:
                # Get full display name for selected ward
                selected_area_display_val = df.loc[df['WD25NM'] == selected_area_filter_val, 'ward_display_name'].iloc[
                    0]
            except IndexError:
                st.sidebar.error(f"Could not find display name for ward: {selected_area_filter_val}")
                selected_area_filter_val = None
    else:  # LSOA level
        if selected_ward_name:
            try:
                selected_ward_code = \
                    wards_in_lad_df.loc[wards_in_lad_df['WD25NM'] == selected_ward_name, 'WD25CD'].iloc[0]
                lsoas_in_ward = lsoa_index_gdf_page[lsoa_index_gdf_page["WD25CD"] == selected_ward_code]
                lsoas_in_ward = lsoas_in_ward.dropna(subset=['lsoa_dropdown_name'])

                lsoa_options = sorted(lsoas_in_ward['lsoa_dropdown_name'].unique())
                if lsoa_options:
                    selected_area_filter_val = st.sidebar.selectbox("3. Select Neighbourhood (LSOA)", lsoa_options)
                    if selected_area_filter_val:
                        # Get full display name for selected LSOA
                        selected_area_display_val = lsoas_in_ward.loc[
                            lsoas_in_ward[
                                'lsoa_dropdown_name'] == selected_area_filter_val, 'lsoa_full_display_name'].iloc[0]
                else:
                    st.sidebar.warning(f"No Neighbourhood data found for ward: {selected_ward_name}")
            except IndexError:
                st.sidebar.error(f"Error retrieving data for ward: {selected_ward_name}")
                selected_area_filter_val = None

    # If an area has been successfully selected, display the chart
    if selected_area_filter_val and selected_area_display_val:
        filter_col = 'WD25NM' if dist_level == "Ward" else 'lsoa_dropdown_name'
        indicator_col = INDICATORS[indicator_dist_name]['col']

        # Check if column exists
        if indicator_col not in df.columns:
            st.error(f"Data for '{indicator_dist_name}' is missing in the dataset.")
        else:
            # Find the specific row and value for the selected area
            selected_area_row = df[df[filter_col] == selected_area_filter_val]

            if not selected_area_row.empty:
                area_value = selected_area_row[indicator_col].iloc[0]
                st.subheader(f"Distribution of '{indicator_dist_name}' across all {dist_level.lower()}s")
                st.markdown(f"The red line marks the score for **{selected_area_display_val}**.")

                # Histogram
                fig = px.histogram(df, x=indicator_col, nbins=50,
                                   title=f"Distribution of {indicator_dist_name} ({TARGET_YEAR})")
                # Add vertical line for the selected area value
                fig.add_vline(x=area_value, line_width=3, line_dash="dash", line_color="red",
                              annotation_text=f"{selected_area_display_val}: {area_value:.1f}",
                              annotation_position="top left")
                st.plotly_chart(fig, use_container_width=True)

                # Calculate and display percentile rank
                percentile = df[indicator_col].rank(pct=True).loc[df[filter_col] == selected_area_filter_val].iloc[
                                 0] * 100
                is_ascending = INDICATORS[indicator_dist_name]['ascending']
                percentile_rank = percentile if not is_ascending else (100 - percentile)
                comparison_word = "top" if not is_ascending else "bottom"

                col1, col2 = st.columns(2)
                col1.metric(label=f"Score for {selected_area_display_val}", value=f"{area_value:.2f}",
                            help=INDICATORS[indicator_dist_name]['desc'])
                col2.metric(label="Percentile Rank", value=f"{comparison_word.title()} {percentile_rank:.0f}%",
                            help=f"This {dist_level.lower()} performs better than {percentile_rank:.0f}% of all {dist_level.lower()}s for this indicator.")
            else:
                st.error(f"Data for '{selected_area_display_val}' could not be found. Please check selections.")
    else:
        st.info("Please complete the selection in the sidebar to view the distribution chart.")

# Footer for Sources & Licensing
with st.expander("Sources & Licensing", expanded=False):
    st.markdown(generate_attribution_markdown())