from pathlib import Path
import geopandas as gpd
import pyogrio
from shapely.ops import unary_union

#Paths and Constants
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)
lad_path = RAW / "geometry" / "LAD_MAY_2025_UK_BFC_V2_6634550694215771101.gpkg"
lsoa_path = RAW / "geometry" / "Lower_layer_Super_Output_Areas_(December_2021)_Boundaries_EW_BFC_(V10).geojson"
assert lad_path.exists(), f"LAD file not found: {lad_path}"
assert lsoa_path.exists(), f"LSOA file not found: {lsoa_path}"
lad = gpd.read_file(lad_path).to_crs(4326)
name_col = "LAD25NM" if "LAD25NM" in lad.columns else next(c for c in lad.columns if c.lower().endswith("nm"))
code_col = "LAD25CD" if "LAD25CD" in lad.columns else next(c for c in lad.columns if c.lower().endswith("cd"))

keep = [
    "Bristol, City of", "Cornwall",
    "East Devon","Exeter","Mid Devon","North Devon","South Hams","Teignbridge","Torridge","West Devon",
    "Dorset",
    "Somerset",
    "Plymouth","Torbay","Bath and North East Somerset","North Somerset","South Gloucestershire"
]

#Filter to keep LADs
lad_sw = lad[lad[name_col].isin(keep)][[name_col, code_col, "geometry"]].rename(
    columns={name_col:"lad_name", code_col:"lad_code"}
)
if lad_sw.empty:
    raise SystemExit("[ERROR] No matching LADs found; check names/columns.")

mask_geom = unary_union(lad_sw.geometry)
mask = mask_geom.buffer(0.005)

#Read only LSOAs intersecting the slightly larger mask
wanted_cols = ["LSOA21CD", "LSOA21NM"]
lsoa_sw = pyogrio.read_dataframe(lsoa_path, mask=mask, columns=wanted_cols, read_geometry=True)
lsoa_sw = lsoa_sw.rename(columns={"LSOA21CD":"area_code","LSOA21NM":"area_name"})

#Robust spatial join to add LAD name/code
lsoa_sw_27700 = lsoa_sw.to_crs(27700).copy()
lad_sw_27700  = lad_sw.to_crs(27700).copy()

lsoa_sw_27700["geometry"] = lsoa_sw_27700.geometry.buffer(0)
lad_sw_27700["geometry"]  = lad_sw_27700.geometry.buffer(0)

joined = (
    lsoa_sw_27700
    .sjoin(lad_sw_27700[["lad_code","lad_name","geometry"]], how="left", predicate="within")
    .drop(columns=["index_right"])
)

null_mask = joined["lad_name"].isna()
if null_mask.any():
    cent = joined.loc[null_mask, ["area_code", "geometry"]].copy()
    cent["geometry"] = cent.geometry.centroid
    cent_fix = (
        cent.sjoin(lad_sw_27700[["lad_code", "lad_name", "geometry"]], how="left", predicate="within")
        .drop(columns=["index_right"])
    )
    cent_fix = cent_fix.reindex(cent.index)
    joined.loc[null_mask, ["lad_code", "lad_name"]] = cent_fix[["lad_code", "lad_name"]].values

null_mask = joined["lad_name"].isna()
if null_mask.any():
    to_fix = joined.loc[null_mask, ["area_code", "geometry"]].copy()
    overlaps = gpd.overlay(to_fix, lad_sw_27700[["lad_code", "lad_name", "geometry"]], how="intersection", keep_geom_type=False)
    if not overlaps.empty:
        overlaps["overlap_area"] = overlaps.area
        idxmax = overlaps.groupby("area_code")["overlap_area"].idxmax()
        best = overlaps.loc[idxmax, ["area_code", "lad_code", "lad_name"]]
        joined = joined.merge(best, on="area_code", how="left", suffixes=("", "_best"))
        for col in ("lad_code", "lad_name"):
            joined[col] = joined[col].fillna(joined[f"{col}_best"])
        joined = joined.drop(columns=[c for c in joined.columns if c.endswith("_best")])

null_mask = joined["lad_name"].isna()
if null_mask.any():
    print(f"[INFO] {null_mask.sum()} LSOAs remain. Using 'nearest' as final fallback...")
    to_fix_nearest = joined.loc[null_mask, ["area_code", "geometry"]].copy()
    nearest_fix = gpd.sjoin_nearest(to_fix_nearest, lad_sw_27700[["lad_code", "lad_name", "geometry"]], how="left").drop(columns=["index_right"])
    nearest_fix = nearest_fix.set_index('area_code')
    joined = joined.set_index('area_code')
    joined.update(nearest_fix)
    joined = joined.reset_index()

lsoa_sw_joined = joined.to_crs(4326)

still_null = lsoa_sw_joined["lad_name"].isna().sum()
if still_null > 0:
    print(f"[WARN] After all fallbacks, {still_null} LSOAs could not be assigned a LAD.")
else:
    print("[INFO] All LSOAs successfully assigned to a LAD.")

#Save GeoParquet
parquet_out = OUT / "boundaries_lsoa.geoparquet"
lsoa_sw_joined[["area_code","area_name","lad_code","lad_name","geometry"]].to_parquet(parquet_out)
print(f"[INFO] Wrote: {parquet_out} ({parquet_out.stat().st_size/1e6:.1f} MB)")

#Save simplified LAD outline (UI)
lad_outline = lad_sw_27700.copy()
lad_outline["geometry"] = lad_outline.geometry.simplify(20)
lad_outline = lad_outline.to_crs(4326)
lad_outline = lad_outline[ lad_outline.geometry.notna() & (~lad_outline.geometry.is_empty) ]
lad_outline_out = OUT / "lad_sw_outline.geojson"
lad_outline.to_file(lad_outline_out, driver="GeoJSON")
print(f"[INFO] Wrote simplified: {lad_outline_out} ({lad_outline_out.stat().st_size/1e6:.1f} MB)")
print("[INFO] Done.")