from pathlib import Path
import sys
import pandas as pd
import geopandas as gpd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
BOUND = PROC / "boundaries_lsoa.geoparquet"

if not BOUND.exists():
    print(f"[ERROR] Missing {BOUND}")
    sys.exit(1)

gdf = gpd.read_parquet(BOUND)

print("[INFO] CRS:", gdf.crs)
print("[INFO] Rows (LSOAs):", len(gdf))
print("[INFO] Columns:", list(gdf.columns))

expected_lads = [
    "Bristol, City of",
    "Cornwall",
    "East Devon","Exeter","Mid Devon","North Devon","South Hams","Teignbridge","Torridge","West Devon",
    "Dorset",
    "Cheltenham","Cotswold","Forest of Dean","Gloucester","Stroud","Tewkesbury",
    "Somerset",
    "Wiltshire",
    "Plymouth","Torbay","Bath and North East Somerset","North Somerset",
]

present = set(gdf["lad_name"].dropna().unique())
missing = [n for n in expected_lads if n not in present]
extras  = [n for n in sorted(present) if n not in expected_lads]

print("\n[CHECK] LAD presence")
print(f"  Present LADs (unique): {len(present)}")
if missing:
    print("  MISSING:", missing)
else:
    print("  MISSING: none")
if extras:
    print("  EXTRA (in data but not in your expected list):", extras)
else:
    print("  EXTRA: none")

counts = gdf.groupby("lad_name", dropna=False)["area_code"].nunique().sort_values(ascending=False)
print("\n[SUMMARY] LSOA count by LAD (top 30):")
print(counts.head(30))

counts.to_csv(PROC / "qc_lsoa_counts_by_lad.csv")
dupes = gdf["area_code"].duplicated().sum()
null_codes = gdf["area_code"].isna().sum()
null_lads  = gdf["lad_name"].isna().sum()

print("\n[INTEGRITY]")
print(f"  Duplicate area_code rows: {dupes}")
print(f"  Null area_code rows     : {null_codes}")
print(f"  Null lad_name rows      : {null_lads}")

invalid = (~gdf.geometry.is_valid).sum()
empty   = gdf.geometry.is_empty.sum()
print("\n[GEOMETRY]")
print(f"  Invalid geometries: {invalid}")
print(f"  Empty geometries  : {empty}")

if invalid or empty or null_lads:
    print("\n[HINT] If you see invalid/empty or null lad_name:")
    print("  - Re-run the build script; ensure spatial join used predicate='within' in EPSG:27700.")
    print("  - For null lad_name near borders, try predicate='intersects' or majority-area assignment.")