# Imports
import streamlit as st
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path
import numpy as np
import glob

# Constants
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
LAD_PALETTE = ["#377eb8", "#e41a1c", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999", "#6a3d9a", "#dede00"]
AREA_CODES_CSV = DATA_DIR / "area_codes.csv"
POSTCODE_FILE = DATA_DIR / "sw_postcodes.parquet"
WARD_GJSON = DATA_DIR / "boundaries_ward.geojson"
LAD_GDF_FILE = DATA_DIR / "lad_sw_outline.geojson"
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


#Core Geometry Loaders
@st.cache_data(show_spinner="Loading geometries...")
def load_lsoa_boundaries():
    """Loads LSOA geometries (Shapes only)."""
    # Helper to load chunked files
    base_name = "boundaries_lsoa"
    search_pattern = str(DATA_DIR / f"{base_name}_part*.geoparquet")
    parts = sorted(glob.glob(search_pattern), key=lambda x: int(x.split('part')[-1].split('.')[0]))
    if not parts:
        # Fallback to single file if parts don't exist
        single_file = DATA_DIR / f"{base_name}.geoparquet"
        if single_file.exists():
            return gpd.read_parquet(single_file).to_crs(4326)
        st.error(f"Could not find LSOA boundaries in {DATA_DIR}")
        return gpd.GeoDataFrame()
    gdfs = [gpd.read_parquet(p) for p in parts]
    gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    return gdf.to_crs(4326)

@st.cache_data(show_spinner=False)
def load_political_boundaries():
    """Loads LAD and Ward boundaries."""
    lad_gdf = gpd.read_file(LAD_GDF_FILE).to_crs(4326)
    ward_gdf = gpd.read_file(WARD_GJSON).to_crs(4326)
    return lad_gdf, ward_gdf

@st.cache_data(show_spinner=False)
def load_greenspace_geometries():
    if not GREENSPACE_GEOMETRIES_FILE.exists():
        return gpd.GeoDataFrame([], geometry=[], crs="EPSG:4326")
    return gpd.read_parquet(GREENSPACE_GEOMETRIES_FILE).to_crs(4326)

# Attribute Data Loaders
@st.cache_data(show_spinner="Loading data attributes...")
def load_attribute_data(year_filter=None):
    """
    Loads scores and context data.
    """
    # Load Area Codes (Lookup)
    area_codes_df = pd.read_csv(AREA_CODES_CSV)
    area_codes_df.columns = area_codes_df.columns.str.strip()
    area_codes_df["LSOA21CD"] = area_codes_df["LSOA21CD"].astype(str).str.strip().str.upper()

    # Load Context Data
    if not LSOA_CONTEXT_DATA_FILE.exists():
        st.error(f"Context file missing: {LSOA_CONTEXT_DATA_FILE}")
        return pd.DataFrame()

    context_df = pd.read_parquet(LSOA_CONTEXT_DATA_FILE)
    context_df['year'] = context_df['year'].astype(int)

    # Load Scores
    if LSOA_FINAL_SCORES_FILE.exists():
        scores_df = pd.read_parquet(LSOA_FINAL_SCORES_FILE)
        if 'year' in scores_df.columns:
            scores_df['year'] = scores_df['year'].astype(int)

        # Merge scores
        master_df = context_df.merge(scores_df, on=['area_code', 'year'], how='left', suffixes=('', '_score_file'))
    else:
        master_df = context_df

    # Filter Year (Crucial for Memory)
    if year_filter:
        master_df = master_df[master_df['year'] == year_filter].copy()

    # Merge Area Info (LAD/Ward Names)
    master_df = master_df.merge(
        area_codes_df[["LSOA21CD", "WD25CD", "WD25NM", "LAD25CD", "LAD25NM"]],
        left_on="area_code",
        right_on="LSOA21CD",
        how="left"
    )
    master_df['WD25NM'] = master_df['WD25NM'].fillna('Uncategorised')

    # Calculate IMD Deciles
    imd_vars = {
        'IMD_Score': 'IMD_Decile', 'Income_Rate': 'Income_Decile',
        'Employment_Rate': 'Employment_Decile', 'Health_Score': 'Health_Decile',
        'IDACI_Rate': 'IDACI_Decile'
    }
    for score_col, decile_col in imd_vars.items():
        if score_col in master_df.columns and master_df[score_col].nunique() > 1:
            try:
                master_df[decile_col] = pd.qcut(master_df[score_col].rank(method='first'), 10,
                                                labels=list(range(10, 0, -1)))
                master_df[decile_col] = pd.to_numeric(master_df[decile_col])
            except:
                master_df[decile_col] = np.nan

    # Display Names
    master_df = master_df.sort_values(by=['WD25NM', 'area_code', 'year']).reset_index(drop=True)
    master_df['neighbourhood_num'] = master_df.groupby(['WD25NM', 'year']).cumcount() + 1
    master_df['display_name'] = master_df.apply(
        lambda row: f"{row['WD25NM']} - Neighbourhood {row['neighbourhood_num']}", axis=1)

    return master_df

# Aggregation
@st.cache_data
def get_2024_map_data():
    """
    Specific optimized loader for the Map Dashboard.
    """
    df_2024 = load_attribute_data(year_filter=2024)
    lsoa_geo = load_lsoa_boundaries()
    lsoa_gdf_2024 = lsoa_geo.merge(df_2024, on="area_code", how="inner")
    ward_stats = _calculate_ward_stats(df_2024)
    return lsoa_gdf_2024, ward_stats

def _calculate_ward_stats(df):
    """Internal helper to aggregate LSOA data to Ward level."""
    agg_cols = {
        'Final_CI_Score': 'mean',
        'Socio-Economic_Deprivation_Score': 'mean',
        'Environmental_Safety_Score': 'mean',
        'Secondary_Education_Score': 'mean',
        'Primary_Education_Score': 'mean',
        'Childcare_Quality_Score': 'mean',
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
        'IMD_Decile': 'mean', 'Income_Decile': 'mean',
        'Employment_Decile': 'mean', 'Health_Decile': 'mean', 'IDACI_Decile': 'mean'
    }
    valid_cols = {k: v for k, v in agg_cols.items() if k in df.columns}
    if not valid_cols:
        return pd.DataFrame()
    ward_stats = df.groupby(['WD25CD', 'WD25NM'], as_index=False).agg(valid_cols)

    # Re-attach LAD info
    lookup = df[['WD25CD', 'LAD25CD']].drop_duplicates().dropna()
    ward_stats = ward_stats.merge(lookup, on='WD25CD', how='left')

    # Cleanup rounding
    int_cols = ['latest_median_house_price', 'IMD_Decile', 'Income_Decile', 'Employment_Decile', 'Health_Decile',
                'IDACI_Decile']
    for col in int_cols:
        if col in ward_stats.columns:
            ward_stats[col] = ward_stats[col].fillna(0).round(0).astype(int)

    return ward_stats


# Secondary Data Loaders (Historical/Specific)
@st.cache_data
def load_postcode_list():
    if not POSTCODE_FILE.exists(): return []
    df = pd.read_parquet(POSTCODE_FILE, columns=['Postcode'])
    return sorted(df['Postcode'].unique().tolist())

@st.cache_data
def get_postcode_coords(postcode: str):
    if not POSTCODE_FILE.exists(): return None
    df = pd.read_parquet(POSTCODE_FILE)
    match = df[df['Postcode'] == postcode]
    return (match.iloc[0]['latitude'], match.iloc[0]['longitude']) if not match.empty else None

@st.cache_data
def load_monthly_crime_data():
    if not LSOA_MONTHLY_CRIME_FILE.exists(): return pd.DataFrame()
    df = pd.read_parquet(LSOA_MONTHLY_CRIME_FILE)
    df['period'] = pd.to_datetime(df['period'])
    return df.set_index('period')

@st.cache_data
def load_house_price_history(level='lsoa'):
    files = {
        'lsoa': LSOA_HOUSE_PRICE_TIMESERIES_FILE,
        'ward': WARD_HOUSE_PRICE_TIMESERIES_FILE,
        'sw': SW_HOUSE_PRICE_TIMESERIES_FILE
    }
    fpath = files.get(level)
    if fpath and fpath.exists():
        df = pd.read_parquet(fpath)
        if 'area_code' in df.columns:
            df['area_code'] = df['area_code'].astype(str).str.strip()
        return df
    return pd.DataFrame()

@st.cache_data
def load_gp_historical_data():
    return pd.read_parquet(HISTORICAL_GP_SCORES_FILE).assign(
        year=lambda x: x['year'].astype(str)) if HISTORICAL_GP_SCORES_FILE.exists() else pd.DataFrame()

@st.cache_data
def load_childcare_historical_data():
    return pd.read_parquet(HISTORICAL_CHILDCARE_FILE).assign(
        year=lambda x: x['year'].astype(str)) if HISTORICAL_CHILDCARE_FILE.exists() else pd.DataFrame()

@st.cache_data
def load_primary_historical_data():
    return pd.read_parquet(HISTORICAL_PRIMARY_SCORES_FILE).assign(
        year=lambda x: x['year'].astype(str)) if HISTORICAL_PRIMARY_SCORES_FILE.exists() else pd.DataFrame()

@st.cache_data
def load_secondary_historical_data():
    return pd.read_parquet(HISTORICAL_SECONDARY_SCORES_FILE).assign(
        year=lambda x: x['year'].astype(str)) if HISTORICAL_SECONDARY_SCORES_FILE.exists() else pd.DataFrame()

# Helpers
def find_containing_area(gdf: gpd.GeoDataFrame, lat: float, lon: float):
    """Spatial lookup using sindex."""
    point = Point(lon, lat)
    # Using spatial index for speed
    possible_matches_index = list(gdf.sindex.intersection(point.bounds))
    possible_matches = gdf.iloc[possible_matches_index]
    precise_match = possible_matches[possible_matches.contains(point)]
    return precise_match.iloc[0] if not precise_match.empty else None

def get_color(key: str, palette):
    return palette[hash(str(key)) % len(palette)]