#Imports
import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
from datetime import date

#Constants
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
LSOA_ANNUAL_INDICATORS_FILE = DATA_DIR / "lsoa_annual_indicators.parquet"
LSOA_MONTHLY_CRIME_FILE = DATA_DIR / "lsoa_monthly_crime.parquet"
HISTORICAL_GP_SCORES_FILE = DATA_DIR / "gp_historical_satisfaction.parquet"
HISTORICAL_CHILDCARE_FILE = DATA_DIR / "childcare_historical_data.parquet"
HISTORICAL_PRIMARY_SCORES_FILE = DATA_DIR / "primary_school_historical_data.parquet"
HISTORICAL_SECONDARY_SCORES_FILE = DATA_DIR / "secondary_school_historical_data.parquet"


#Data Loaders
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


#Scoring Functions
def calculate_greenspace_score(df):
    df['greenspace_score'] = df['greenspace_percentage'].rank(pct=True) * 100
    return df

def calculate_air_quality_score(row):
    MAX_NO2, MAX_PM25 = 50, 25
    no2_conc, pm25_conc = row.get('no2_mean_concentration', 0), row.get('pm25_mean_concentration', 0)
    no2_norm, pm25_norm = min(no2_conc / MAX_NO2, 1.0), min(pm25_conc / MAX_PM25, 1.0)
    avg_pollutant_norm = (no2_norm + pm25_norm) / 2
    return (1 - avg_pollutant_norm) * 100

def calculate_community_safety_score(df):
    # --- FIX 1: Fillna(1) to avoid divide by zero if population is missing ---
    safe_population = df['population'].replace(0, 1).fillna(1)
    df['crime_rate_per_1000'] = (df['crime_count'] / safe_population) * 1000
    df['community_safety_score'] = df['crime_rate_per_1000'].rank(pct=True, ascending=False) * 100
    return df

def calculate_secondary_education_score(df):
    df['progress_8_percentile'] = df['avg_progress_8'].rank(pct=True, ascending=True) * 100
    df['attainment_8_percentile'] = df['avg_attainment_8'].rank(pct=True, ascending=True) * 100
    df['secondary_education_score'] = (df['progress_8_percentile'] + df['attainment_8_percentile']) / 2
    return df

def calculate_primary_education_score(df):
    df['pass_rate_percentile'] = df['avg_ks2_pass_rate'].rank(pct=True, ascending=True) * 100
    df['scaled_score_percentile'] = df['avg_primary_scaled_score'].rank(pct=True, ascending=True) * 100
    df['primary_education_score'] = (df['pass_rate_percentile'] + df['scaled_score_percentile']) / 2
    return df

def calculate_healthcare_score(df):
    df['distance_percentile'] = df['avg_distance_to_gp_km'].rank(pct=True, ascending=False) * 100
    df['satisfaction_percentile'] = df['avg_gp_satisfaction'].rank(pct=True, ascending=True) * 100
    df['healthcare_score'] = (df['distance_percentile'] + df['satisfaction_percentile']) / 2
    return df

def calculate_childcare_score(df):
    df['childcare_distance_percentile'] = df['avg_distance_to_childcare_km'].rank(pct=True, ascending=False) * 100
    df['childcare_quality_percentile'] = df['avg_childcare_quality_score'].rank(pct=True, ascending=True) * 100
    df['childcare_places_percentile'] = df['total_childcare_places_nearby'].rank(pct=True, ascending=True) * 100
    df['childcare_score'] = (df['childcare_distance_percentile'] + df['childcare_quality_percentile'] + df[
        'childcare_places_percentile']) / 3
    return df

def calculate_thrive_score(row, weights_dict):
    """Calculates the composite score based on user-defined weights."""
    score = (row.get('greenspace_score', 0) * weights_dict['greenspace'] +
             row.get('air_quality_score', 0) * weights_dict['air_quality'] +
             row.get('community_safety_score', 0) * weights_dict['safety'] +
             row.get('education_score', 0) * weights_dict['education'] +
             row.get('healthcare_score', 0) * weights_dict['healthcare'] +
             row.get('childcare_score', 0) * weights_dict['childcare'])
    return score

#Master Data Loader
@st.cache_data(show_spinner="Loading and preparing all map data...")
def load_master_data():
    """
    Loads all base geographies and the master indicator time-series.
    This function loads the RAW data and stores it in session state.
    Scoring is handled by get_scored_data_for_year().
    """
    print("Running load_master_data()...")
    #Part 1 - Load base geographic data
    lad_gdf = gpd.read_file(LAD_GDF_FILE).to_crs(4326)
    lsoa_index_gdf_base = gpd.read_parquet(LSOA_BOUNDARIES_FILE).to_crs(4326)
    ward_gdf = gpd.read_file(WARD_GJSON).to_crs(4326)

    #Store base geos in session state for lookups
    st.session_state['lad_gdf'] = lad_gdf
    st.session_state['ward_gdf'] = ward_gdf
    st.session_state['lsoa_index_gdf_base'] = lsoa_index_gdf_base

    #Part 2 - Load the master annual indicators file
    if not LSOA_ANNUAL_INDICATORS_FILE.exists():
        st.error(f"Master data file not found: {LSOA_ANNUAL_INDICATORS_FILE.name}")
        st.info("Please run `scripts/build_master_timeseries.py` to create it.")
        st.stop()
    master_df = pd.read_parquet(LSOA_ANNUAL_INDICATORS_FILE)

    #Part 3 - Merge LSOA geometries onto the master data
    master_gdf = lsoa_index_gdf_base.merge(master_df, on="area_code", how="left")

    #Part 4 - Add Ward/LAD info
    area_codes_df = pd.read_csv(AREA_CODES_CSV)
    area_codes_df.columns = area_codes_df.columns.str.strip()
    master_gdf = master_gdf.merge(
        area_codes_df[["LSOA21CD", "WD25CD", "WD25NM", "LAD25CD", "LAD25NM"]],
        left_on="area_code",
        right_on="LSOA21CD",
        how="left"
    )
    master_gdf['WD25NM'] = master_gdf['WD25NM'].fillna('Uncategorised')
    master_gdf = master_gdf.sort_values(by=['WD25NM', 'area_code', 'year']).reset_index(drop=True)
    master_gdf['neighbourhood_num'] = master_gdf.groupby(['WD25NM', 'year']).cumcount() + 1
    master_gdf['neighbourhood_num'] = master_gdf['neighbourhood_num'].astype(int)
    master_gdf['display_name'] = master_gdf.apply(
        lambda row: f"{row['WD25NM']} - Neighbourhood {row['neighbourhood_num']}",
        axis=1
    )

    # Store the master gdf in session state
    st.session_state['master_gdf'] = master_gdf

    #Part 5 - Load House Price Time Series
    def load_price_history(filepath):
        if filepath.exists():
            try:
                df = pd.read_parquet(filepath)
                if 'area_code' in df.columns:
                    df['area_code'] = df['area_code'].astype(str).str.strip()
                return df
            except Exception as e:
                print(f"⚠️ Error loading house price file {filepath.name}: {e}")
        else:
            print(f"⚠️ Warning: House price history file not found: {filepath.name}")
        return pd.DataFrame()

    st.session_state['lsoa_house_price_history'] = load_price_history(LSOA_HOUSE_PRICE_TIMESERIES_FILE)
    st.session_state['ward_house_price_history'] = load_price_history(WARD_HOUSE_PRICE_TIMESERIES_FILE)
    st.session_state['sw_house_price_history'] = load_price_history(SW_HOUSE_PRICE_TIMESERIES_FILE)

    if not st.session_state['lsoa_house_price_history'].empty:
        latest_date = st.session_state['lsoa_house_price_history']['date'].max()
        st.session_state['latest_house_price_date'] = latest_date
        print(f"Loaded house price history. Latest period: {latest_date}")

    st.session_state['gp_historical_df'] = load_gp_historical_data()
    st.session_state['childcare_historical_df'] = load_childcare_historical_data()
    st.session_state['primary_historical_df'] = load_primary_historical_data()
    st.session_state['secondary_historical_df'] = load_secondary_historical_data()


#Scoring Function
@st.cache_data
def get_scored_data_for_year(selected_year: int, thrive_weights_tuple: tuple):
    """
    Takes the master GDF from session state, filters to a year,
    and calculates all scores for that year's data.
    This function is cached, so it only re-runs when the year or weights change.
    Weights must be passed as a tuple to be hashable for caching.
    """
    print(f"--- Running get_scored_data_for_year for {selected_year} ---")
    thrive_weights = dict(thrive_weights_tuple)
    if 'master_gdf' not in st.session_state:
        st.error("Master data not loaded. Please refresh.")
        return pd.DataFrame(), pd.DataFrame()

    #Filter data for the selected year
    gdf_year = st.session_state['master_gdf'][
        st.session_state['master_gdf']['year'] == selected_year
        ].copy()

    #Apply all individual scoring functions
    gdf_year = calculate_community_safety_score(gdf_year)
    gdf_year = calculate_greenspace_score(gdf_year)
    gdf_year = calculate_secondary_education_score(gdf_year)
    gdf_year = calculate_primary_education_score(gdf_year)
    gdf_year = calculate_healthcare_score(gdf_year)
    gdf_year = calculate_childcare_score(gdf_year)
    gdf_year['air_quality_score'] = gdf_year.apply(calculate_air_quality_score, axis=1)

    #Calculate combined scores
    gdf_year['education_score'] = (gdf_year['primary_education_score'] + gdf_year['secondary_education_score']) / 2

    #Calculate final composite score using the provided weights
    gdf_year['composite_score'] = gdf_year.apply(
        lambda row: calculate_thrive_score(row, thrive_weights), axis=1
    )

    #Calculate ward-level stats for the selected year
    agg_cols = {
        'greenspace_score': 'mean', 'greenspace_percentage': 'mean', 'air_quality_score': 'mean',
        'no2_mean_concentration': 'mean', 'pm25_mean_concentration': 'mean',
        'community_safety_score': 'mean', 'crime_rate_per_1000': 'mean',
        'education_score': 'mean', 'primary_education_score': 'mean', 'secondary_education_score': 'mean',
        'avg_primary_scaled_score': 'mean', 'avg_ks2_pass_rate': 'mean',
        'avg_progress_8': 'mean', 'avg_attainment_8': 'mean',
        'healthcare_score': 'mean', 'avg_distance_to_gp_km': 'mean', 'avg_gp_satisfaction': 'mean',
        'childcare_score': 'mean', 'avg_distance_to_childcare_km': 'mean',
        'avg_childcare_quality_score': 'mean', 'total_childcare_places_nearby': 'mean',
        'composite_score': 'mean', 'population': 'sum', 'latest_median_house_price': 'mean',
        'IMD_Decile': 'mean', 'Income_Decile': 'mean', 'Employment_Decile': 'mean', 'Health_Decile': 'mean'
    }

    #Group by Ward codes, then add the LAD codes back
    group_cols = ['WD25CD', 'WD25NM']
    valid_agg_cols = {k: v for k, v in agg_cols.items() if k in gdf_year.columns}

    if not valid_agg_cols:
        ward_stats_df = pd.DataFrame(columns=group_cols + ['LAD25CD'])
    else:
        #Aggregate the data
        ward_stats_df = gdf_year.groupby(group_cols, as_index=False).agg(valid_agg_cols)
        #Create a lookup to find the LAD code for each Ward code
        ward_lad_lookup = gdf_year[['WD25CD', 'LAD25CD']].drop_duplicates().dropna()
        #Merge the LAD code back onto the aggregated dataframe
        ward_stats_df = ward_stats_df.merge(ward_lad_lookup, on='WD25CD', how='left')
        #Clean up aggregated data types
        if 'latest_median_house_price' in ward_stats_df.columns:
            ward_stats_df['latest_median_house_price'] = ward_stats_df['latest_median_house_price'].fillna(0).astype(
                int)

        imd_cols_to_int = ['IMD_Decile', 'Income_Decile', 'Employment_Decile', 'Health_Decile']
        for col in imd_cols_to_int:
            if col in ward_stats_df.columns:
                ward_stats_df[col] = ward_stats_df[col].round(0).fillna(0).astype(int)
    return gdf_year, ward_stats_df


#Helper Functions ---
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