# Imports
import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import numpy as np

# Constants
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
LAD_PALETTE = ["#377eb8", "#e41a1c", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999", "#6a3d9a", "#dede00"]
AREA_CODES_CSV = DATA_DIR / "area_codes.csv"
POSTCODE_FILE = DATA_DIR / "sw_postcodes.parquet"
WARD_GJSON = DATA_DIR / "boundaries_ward.geojson"
LAD_GDF_FILE = DATA_DIR / "lad_sw_outline.geojson"
LSOA_BOUNDARIES_FILE = DATA_DIR / "boundaries_lsoa.geoparquet"
GREENSPACE_GEOMETRIES_FILE = DATA_DIR / "sw_greenspace_geometries.geoparquet"
LSOA_HOUSE_PRICE_TIMESERIES_FILE = DATA_DIR / "lsoa_house_prices_timeseries.parquet"
WARD_HOUSE_PRICE_TIMESERIES_FILE = DATA_DIR / "ward_house_prices_timeseries.parquet"
SW_HOUSE_PRICE_TIMESERIES_FILE = DATA_DIR / "sw_house_prices_timeseries.parquet"
LSOA_MONTHLY_CRIME_FILE = DATA_DIR / "lsoa_monthly_crime.parquet"
LSOA_FINAL_SCORES_FILE = DATA_DIR / "lsoa_final_composite_scores_2024.parquet"
LSOA_CONTEXT_DATA_FILE = DATA_DIR / "lsoa_annual_indicators_non_imputed.parquet"
HISTORICAL_GP_SCORES_FILE = DATA_DIR / "gp_historical_satisfaction.parquet"
HISTORICAL_CHILDCARE_FILE = DATA_DIR / "childcare_historical_data.parquet"
HISTORICAL_PRIMARY_SCORES_FILE = DATA_DIR / "primary_school_historical_data.parquet"
HISTORICAL_SECONDARY_SCORES_FILE = DATA_DIR / "secondary_school_historical_data.parquet"

# Data Loaders
@st.cache_data(show_spinner="Loading greenspace areas...")
def load_greenspace_geometries():
    """Loads and re-projects the greenspace geometry file for map overlays."""
    if not GREENSPACE_GEOMETRIES_FILE.exists():
        st.warning("Greenspace geometry file not found. Skipping greenspace layer.")
        return gpd.GeoDataFrame([], geometry=[], crs="EPSG:4326")
    gdf = gpd.read_parquet(GREENSPACE_GEOMETRIES_FILE)
    return gdf.to_crs(4326)

@st.cache_data(show_spinner=False)
def load_area_codes(valid_codes):
    """Loads the LSOA to Ward/LAD lookup table, filtered to valid codes."""
    df = pd.read_csv(AREA_CODES_CSV)
    df["LSOA21CD"] = df["LSOA21CD"].astype(str).str.strip().str.upper()
    valid = pd.Series(list(valid_codes), dtype=str).str.strip().str.upper()
    return df[df["LSOA21CD"].isin(valid)].copy()

@st.cache_data
def load_postcode_list():
    """Loads a unique, sorted list of all postcodes for the search box."""
    if not POSTCODE_FILE.exists():
        return []
    df = pd.read_parquet(POSTCODE_FILE, columns=['Postcode'])
    return sorted(df['Postcode'].unique().tolist())

@st.cache_data
def get_postcode_coords(postcode: str):
    """Fetches the latitude and longitude for a given postcode."""
    if not POSTCODE_FILE.exists():
        return None
    df = pd.read_parquet(POSTCODE_FILE)
    match = df[df['Postcode'] == postcode]
    if not match.empty:
        return match.iloc[0]['latitude'], match.iloc[0]['longitude']
    return None

@st.cache_data(show_spinner="Loading detailed crime history...")
def load_monthly_crime_data():
    """Loads the pre-aggregated monthly crime data for deep-dive charts."""
    if not LSOA_MONTHLY_CRIME_FILE.exists():
        st.warning("Monthly crime file not found. Deep-dive charts will be empty.")
        return pd.DataFrame()
    df = pd.read_parquet(LSOA_MONTHLY_CRIME_FILE)
    df['period'] = pd.to_datetime(df['period'])
    df = df.set_index('period')
    return df

@st.cache_data(show_spinner="Loading GP satisfaction history...")
def load_gp_historical_data():
    """Loads the pre-processed historical satisfaction data for all GPs."""
    if not HISTORICAL_GP_SCORES_FILE.exists():
        st.warning("GP historical satisfaction file not found. Charts will be empty.")
        return pd.DataFrame()
    df = pd.read_parquet(HISTORICAL_GP_SCORES_FILE)
    df['year'] = df['year'].astype(str)
    return df

@st.cache_data(show_spinner="Loading childcare history...")
def load_childcare_historical_data():
    """Loads the pre-processed historical data for all childcare providers."""
    if not HISTORICAL_CHILDCARE_FILE.exists():
        st.warning("Childcare historical file not found. Charts will be empty.")
        return pd.DataFrame()
    df = pd.read_parquet(HISTORICAL_CHILDCARE_FILE)
    df['year'] = df['year'].astype(str)
    return df


@st.cache_data(show_spinner="Loading primary school history...")
def load_primary_historical_data():
    """Loads the pre-processed historical performance data for all primary schools."""
    if not HISTORICAL_PRIMARY_SCORES_FILE.exists():
        st.warning("Primary school historical file not found. Charts will be empty.")
        return pd.DataFrame()
    df = pd.read_parquet(HISTORICAL_PRIMARY_SCORES_FILE)
    df['year'] = df['year'].astype(str)
    return df


@st.cache_data(show_spinner="Loading secondary school history...")
def load_secondary_historical_data():
    """Loads the pre-processed historical performance data for all secondary schools."""
    if not HISTORICAL_SECONDARY_SCORES_FILE.exists():
        st.warning("Secondary school historical file not found. Charts will be empty.")
        return pd.DataFrame()
    df = pd.read_parquet(HISTORICAL_SECONDARY_SCORES_FILE)
    df['year'] = df['year'].astype(str)
    return df


# Master Data Loader
@st.cache_data(show_spinner="Loading and preparing all map data...")
def load_master_data():
    """
    Loads all base geographies, the context data, and the final 2024 composite scores.
    Merges them into a single master_gdf stored in session_state.
    """
    # Part 1 - Load base geographic data
    lad_gdf = gpd.read_file(LAD_GDF_FILE).to_crs(4326)
    lsoa_index_gdf_base = gpd.read_parquet(LSOA_BOUNDARIES_FILE).to_crs(4326)
    ward_gdf = gpd.read_file(WARD_GJSON).to_crs(4326)

    # Store base geos in session state
    st.session_state['lad_gdf'] = lad_gdf
    st.session_state['ward_gdf'] = ward_gdf
    st.session_state['lsoa_index_gdf_base'] = lsoa_index_gdf_base

    # Part 2 - Load Context Data (Non-imputed annual indicators)
    if not LSOA_CONTEXT_DATA_FILE.exists():
        # Fallback logic to prevent crash during file migration
        fallback = DATA_DIR / "lsoa_annual_indicators.parquet"
        if fallback.exists():
            st.warning(f"Context file {LSOA_CONTEXT_DATA_FILE.name} not found. Using fallback: {fallback.name}")
            context_df = pd.read_parquet(fallback)
        else:
            st.error(f"Context Data file not found: {LSOA_CONTEXT_DATA_FILE.name}")
            st.stop()
    else:
        context_df = pd.read_parquet(LSOA_CONTEXT_DATA_FILE)

    # Part 3 - Load Final Scores (2024)
    if not LSOA_FINAL_SCORES_FILE.exists():
        st.warning(f"Final scores file not found: {LSOA_FINAL_SCORES_FILE.name}. Dashboard will lack scores.")
        scores_df = pd.DataFrame()
    else:
        scores_df = pd.read_parquet(LSOA_FINAL_SCORES_FILE)

    # Part 4 - Merge Context and Scores
    if not scores_df.empty:
        # Ensure year is int in both to guarantee successful merge
        if 'year' in scores_df.columns:
            scores_df['year'] = scores_df['year'].astype(int)
        context_df['year'] = context_df['year'].astype(int)
        # Merge
        master_df = context_df.merge(scores_df, on=['area_code', 'year'], how='left', suffixes=('', '_score_file'))
    else:
        master_df = context_df

    imd_vars = {
        'IMD_Score': 'IMD_Decile',
        'Income_Rate': 'Income_Decile',
        'Employment_Rate': 'Employment_Decile',
        'Health_Score': 'Health_Decile',
        'IDACI_Rate': 'IDACI_Decile'
    }

    for score_col, decile_col in imd_vars.items():
        if score_col in master_df.columns:
            try:
                # Check for sufficient unique values to bin
                if master_df[score_col].nunique() > 1:
                    # qcut with labels [10...1] ensures Highest Score gets Label 1
                    master_df[decile_col] = pd.qcut(
                        master_df[score_col].rank(method='first'),
                        10,
                        labels=[10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
                    )
                    # Convert to numeric to allow aggregation (mean) in get_scored_data_for_year
                    master_df[decile_col] = pd.to_numeric(master_df[decile_col])
                else:
                    master_df[decile_col] = np.nan
            except Exception as e:
                print(f"Warning: Could not calculate {decile_col}: {e}")
                master_df[decile_col] = np.nan

    # Part 5 - Merge LSOA geometries onto the master data
    master_gdf = lsoa_index_gdf_base.merge(master_df, on="area_code", how="left")

    # Part 6 - Add Ward/LAD info
    area_codes_df = pd.read_csv(AREA_CODES_CSV)
    area_codes_df.columns = area_codes_df.columns.str.strip()
    master_gdf = master_gdf.merge(
        area_codes_df[["LSOA21CD", "WD25CD", "WD25NM", "LAD25CD", "LAD25NM"]],
        left_on="area_code",
        right_on="LSOA21CD",
        how="left"
    )
    master_gdf['WD25NM'] = master_gdf['WD25NM'].fillna('Uncategorised')

    # Sorting and Display Names
    master_gdf = master_gdf.sort_values(by=['WD25NM', 'area_code', 'year']).reset_index(drop=True)
    master_gdf['neighbourhood_num'] = master_gdf.groupby(['WD25NM', 'year']).cumcount() + 1
    master_gdf['neighbourhood_num'] = master_gdf['neighbourhood_num'].astype(int)
    master_gdf['display_name'] = master_gdf.apply(
        lambda row: f"{row['WD25NM']} - Neighbourhood {row['neighbourhood_num']}",
        axis=1
    )

    # Store the master gdf in session state
    st.session_state['master_gdf'] = master_gdf

    # Part 7 - Load House Price Time Series
    def load_price_history(filepath):
        if filepath.exists():
            try:
                df = pd.read_parquet(filepath)
                if 'area_code' in df.columns:
                    df['area_code'] = df['area_code'].astype(str).str.strip()
                return df
            except Exception as e:
                print(f"Error loading house price file {filepath.name}: {e}")
        else:
            print(f"Warning: House price history file not found: {filepath.name}")
        return pd.DataFrame()

    st.session_state['lsoa_house_price_history'] = load_price_history(LSOA_HOUSE_PRICE_TIMESERIES_FILE)
    st.session_state['ward_house_price_history'] = load_price_history(WARD_HOUSE_PRICE_TIMESERIES_FILE)
    st.session_state['sw_house_price_history'] = load_price_history(SW_HOUSE_PRICE_TIMESERIES_FILE)

    if not st.session_state['lsoa_house_price_history'].empty:
        latest_date = st.session_state['lsoa_house_price_history']['date'].max()
        st.session_state['latest_house_price_date'] = latest_date

    st.session_state['gp_historical_df'] = load_gp_historical_data()
    st.session_state['childcare_historical_df'] = load_childcare_historical_data()
    st.session_state['primary_historical_df'] = load_primary_historical_data()
    st.session_state['secondary_historical_df'] = load_secondary_historical_data()

# Data Retrieval Function (Scoring Logic Removed)
@st.cache_data
def get_scored_data_for_year(selected_year: int):
    """
    Retrieves the master data for a specific year.
    Since scores are now pre-calculated in the file, this function simply
    filters the dataframe and performs Ward-level aggregation.
    """
    if 'master_gdf' not in st.session_state:
        st.error("Master data not loaded. Please refresh.")
        return pd.DataFrame(), pd.DataFrame()

    # Filter data for the selected year
    gdf_year = st.session_state['master_gdf'][
        st.session_state['master_gdf']['year'] == selected_year
        ].copy()

    if gdf_year.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Aggregation columns
    agg_cols = {
        # Final Scores
        'Final_CI_Score': 'mean',
        'Socio-Economic_Deprivation_Score': 'mean',
        'Environmental_Safety_Score': 'mean',
        'Secondary_Education_Score': 'mean',
        'Primary_Education_Score': 'mean',
        'Childcare_Quality_Score': 'mean',

        # Context Data (Keep raw metrics for UI)
        'greenspace_percentage': 'mean',
        'no2_mean_concentration': 'mean',
        'pm25_mean_concentration': 'mean',
        'crime_rate_per_1000': 'mean',
        'avg_primary_scaled_score': 'mean',
        'avg_ks2_pass_rate': 'mean',
        'avg_progress_8': 'mean',
        'avg_attainment_8': 'mean',
        'avg_distance_to_gp_km': 'mean',
        'avg_gp_satisfaction': 'mean',
        'avg_distance_to_childcare_km': 'mean',
        'avg_childcare_quality_score': 'mean',
        'total_childcare_places_nearby': 'mean',
        'population': 'sum',
        'latest_median_house_price': 'mean',
        # Deciles (Aggregated by Mean for Ward view)
        'IMD_Decile': 'mean',
        'Income_Decile': 'mean',
        'Employment_Decile': 'mean',
        'Health_Decile': 'mean',
        'IDACI_Decile': 'mean'
    }

    # Group by Ward codes, then add the LAD codes back
    group_cols = ['WD25CD', 'WD25NM']
    # Filter agg_cols to only those present in the dataframe
    valid_agg_cols = {k: v for k, v in agg_cols.items() if k in gdf_year.columns}
    if not valid_agg_cols:
        ward_stats_df = pd.DataFrame(columns=group_cols + ['LAD25CD'])
    else:
        ward_stats_df = gdf_year.groupby(group_cols, as_index=False).agg(valid_agg_cols)
        ward_lad_lookup = gdf_year[['WD25CD', 'LAD25CD']].drop_duplicates().dropna()
        ward_stats_df = ward_stats_df.merge(ward_lad_lookup, on='WD25CD', how='left')
        # Cleanup: Rounding and Integers for display
        if 'latest_median_house_price' in ward_stats_df.columns:
            ward_stats_df['latest_median_house_price'] = ward_stats_df['latest_median_house_price'].fillna(0).astype(
                int)

        # Round Deciles
        imd_cols_to_int = ['IMD_Decile', 'Income_Decile', 'Employment_Decile', 'Health_Decile', 'IDACI_Decile']
        for col in imd_cols_to_int:
            if col in ward_stats_df.columns:
                ward_stats_df[col] = ward_stats_df[col].round(0).fillna(0).astype(int)

    return gdf_year, ward_stats_df

# Helper Functions
def find_containing_area(gdf: gpd.GeoDataFrame, lat: float, lon: float):
    """Finds which geometry in a GeoDataFrame contains a given lat/lon point."""
    point = Point(lon, lat)
    possible_matches_idx = list(gdf.sindex.intersection(point.bounds))
    if not possible_matches_idx:
        return None
    possible_matches = gdf.iloc[possible_matches_idx]
    precise_match = possible_matches[possible_matches.contains(point)]
    return precise_match.iloc[0] if not precise_match.empty else None

def get_color(key: str, palette):
    """Gets a consistent color for a given key from a color palette."""
    return palette[hash(str(key)) % len(palette)]