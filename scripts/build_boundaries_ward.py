from pathlib import Path
import pandas as pd
import geopandas as gpd

#Paths and Constants
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
RAW = DATA / "raw"
WARD_SOURCE = RAW / "geometry" / "Wards_(May_2025)_Boundaries_UK_BFC_(V2).geojson"
AREA_CODES = PROCESSED / "area_codes.csv"
WARD_OUT = PROCESSED / "boundaries_ward.geojson"


#Load GeoJSON
print(f"Loading source ward boundaries from {WARD_SOURCE}...")
if not WARD_SOURCE.exists():
    raise FileNotFoundError(
        f"The source file was not found. Please place it at: {WARD_SOURCE}"
    )
wards_raw = gpd.read_file(WARD_SOURCE)
print(f"Loaded {len(wards_raw):,} raw ward polygons.")
required_ward_columns = ["WD25CD", "WD25NM", "geometry"]
missing_ward_cols = [col for col in required_ward_columns if col not in wards_raw.columns]
if missing_ward_cols:
    raise ValueError(
        f"The source GeoJSON file is missing required columns: {missing_ward_cols}.\n"
        f"Please check the file. The available columns are: {wards_raw.columns.tolist()}"
    )
wards = wards_raw[required_ward_columns].copy()


#Load area codes
print(f"Loading area codes from {AREA_CODES} for LAD mapping...")
ac = pd.read_csv(AREA_CODES)

#Add check for required columns
required_ac_columns = ["WD25CD", "LAD25CD", "LAD25NM"]
missing_ac_cols = [col for col in required_ac_columns if col not in ac.columns]
if missing_ac_cols:
    raise ValueError(
        f"The 'area_codes.csv' file is missing required columns: {missing_ac_cols}.\n"
        f"Please check the file. The available columns are: {ac.columns.tolist()}"
    )

#Create mapping from Ward to LAD
ward_to_lad_map = ac[required_ac_columns].drop_duplicates(subset=["WD25CD"]).copy()

#Merge the LAD onto the ward geometry
print("Merging LAD information onto ward boundaries...")
wards["WD25CD"] = wards["WD25CD"].astype(str)
ward_to_lad_map["WD25CD"] = ward_to_lad_map["WD25CD"].astype(str)

#Using inner merge to filter wards to only those in area_code
wards_with_lad = wards.merge(
    ward_to_lad_map,
    on="WD25CD",
    how="inner",
)

#Check for any wards that didn't get a LAD match
unmatched_wards = wards_with_lad[wards_with_lad["LAD25CD"].isnull()]
if not unmatched_wards.empty:
    print(f"Warning: {len(unmatched_wards)} wards could not be matched to a Local Authority.")
    print(unmatched_wards[["WD25CD", "WD25NM"]].head())

#Final columns
final_columns = ["WD25CD", "WD25NM", "LAD25CD", "LAD25NM", "geometry"]
try:
    processed_wards = wards_with_lad[final_columns]
except KeyError as e:
    print(f"Error: A required column is missing from the source file or the merge. Missing column: {e}")
    print(f"Available columns are: {wards_with_lad.columns.tolist()}")
    exit()

#Reproject to WGS 84 (CRS 4326)
print("Reprojecting boundaries to CRS:4326...")
processed_wards = processed_wards.to_crs(4326)

#Simplify the geometries to reduce file size for UI
print("Simplifying ward boundary geometries...")
processed_wards.geometry = processed_wards.simplify(tolerance=0.0001, preserve_topology=True)

#Save GeoJSON
WARD_OUT.parent.mkdir(parents=True, exist_ok=True)
processed_wards.to_file(WARD_OUT, driver="GeoJSON")
print("-" * 30)
print(f"Success! Wrote {len(processed_wards):,} processed wards to {WARD_OUT}")
print("-" * 30)

