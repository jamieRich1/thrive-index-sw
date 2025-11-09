from pathlib import Path
import pandas as pd
import geopandas as gpd

#Paths and Constants
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "postcodes"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
POSTCODE_PREFIXES = [
    "BA", "BH", "BS", "DT",
    "EX", "PL", "SP", "TA",
    "TQ", "TR"
]
OUTPUT_FILE = PROCESSED_DIR / "sw_postcodes.parquet"


def main():
    """
    Reads headerless Code-Point Open CSVs, converts coordinates,
    and saves the result to a Parquet file.
    """
    print("[INFO] Starting postcode processing...")

    existing_files = [
        f for f in [RAW_DIR / f"{p.lower()}.csv" for p in POSTCODE_PREFIXES] if f.exists()
    ]
    print(f"[INFO] Found {len(existing_files)} postcode CSV files.")

    if not existing_files:
        print("[ERROR] No postcode CSV files found. Please check the paths.")
        return

    col_names = [
        "Postcode", "Positional_quality_indicator", "Eastings", "Northings",
        "Country_code", "NHS_regional_HA_code", "NHS_HA_code",
        "Admin_county_code", "Admin_district_code", "Admin_ward_code"
    ]

    df = pd.concat(
        [pd.read_csv(f, header=None, names=col_names, usecols=["Postcode", "Eastings", "Northings"]) for f in
         existing_files],
        ignore_index=True
    )

    print(f"[INFO] Loaded {len(df):,} postcodes from CSVs.")

    #Coordinate Conversion using GeoPandas
    df.dropna(subset=['Eastings', 'Northings'], inplace=True)
    df['Eastings'] = pd.to_numeric(df['Eastings'], errors='coerce')
    df['Northings'] = pd.to_numeric(df['Northings'], errors='coerce')
    df.dropna(subset=['Eastings', 'Northings'], inplace=True)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.Eastings, df.Northings),
        crs="EPSG:27700"
    )

    print("[INFO] Converting coordinates to Latitude/Longitude...")
    gdf = gdf.to_crs("EPSG:4326")
    gdf['longitude'] = gdf.geometry.x
    gdf['latitude'] = gdf.geometry.y

    final_df = gdf[['Postcode', 'latitude', 'longitude']].copy()
    final_df.to_parquet(OUTPUT_FILE, index=False)

    print(f"[SUCCESS] Saved {len(final_df):,} processed postcodes to:")
    print(f"  -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()