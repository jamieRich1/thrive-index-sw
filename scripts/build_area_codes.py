from pathlib import Path
import pandas as pd
import geopandas as gpd

#Paths
DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
RAW_LOOKUP = DATA_ROOT / "raw" / "lookups" / "LSOA_(2021)_to_Electoral_Ward_(2025)_to_LAD_(2025)_Best_Fit_Lookup_in_EW_v2.csv"
BOUND_GPQ = DATA_ROOT / "processed" / "boundaries_lsoa.geoparquet"
BOUND_GJSON = DATA_ROOT / "processed" / "boundaries_lsoa_simplified.geojson"
OUT_CSV = DATA_ROOT / "processed" / "area_codes.csv"

#Load boundary codes
if BOUND_GPQ.exists():
    lsoa_gdf = gpd.read_parquet(BOUND_GPQ).to_crs(4326)
else:
    lsoa_gdf = gpd.read_file(BOUND_GJSON).to_crs(4326)

codes = lsoa_gdf[
    "area_code" if "area_code" in lsoa_gdf.columns else "LSOA21CD"
].astype(str).str.strip().str.upper().unique()

#Load and trim the raw lookup file
df = pd.read_csv(RAW_LOOKUP)
df["LSOA21CD"] = df["LSOA21CD"].astype(str).str.strip().str.upper()
trimmed = df[df["LSOA21CD"].isin(codes)]

#Save all columns to output file
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
trimmed.to_csv(OUT_CSV, index=False)
print(f"Saved {len(trimmed):,} rows to {OUT_CSV}")
