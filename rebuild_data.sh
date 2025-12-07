#!/bin/bash
#
# This script performs a full, clean rebuild of all processed data.
# It should be run from the root of the 'thrive-index-sw' directory.
#
# It will:
# 1. Delete all *processed* data (it will not touch 'data/raw').
# 2. Re-create the empty 'processed' directory.
# 3. Run all the 'build' and 'process' scripts in the correct order.
#
# Use 'bash rebuild_data.sh' to run this.
set -e # This makes the script exit immediately if any command fails

# --- 1. CLEAN UP ---
echo "--- STEP 1: Deleting old processed data... ---"
rm -rf data/processed
mkdir data/processed
echo "  -> 'data/processed' directory is now clean."


# --- 2. BUILD BASE GEOGRAPHIES ---
echo "\n--- STEP 2: Building base geographies... ---"
python scripts/build_boundaries_sw.py
python scripts/build_area_codes.py
python scripts/build_boundaries_ward.py
python scripts/build_postcodes_sw.py
echo "  -> Base geographies (LSOA, Ward, Postcodes) built."


# --- 3. PROCESS ALL 'SPARSE' INDICATORS ---
echo "\n--- STEP 3: Processing all sparse indicator data... ---"
python scripts/process_population.py
python scripts/process_crime_data.py
python scripts/process_air_quality.py

# --- Greenspace Pre-processing ---
echo "  -> Running Greenspace pre-processing (clipping UK file)..."
python scripts/process_sw_greenspace_geometries.py
python scripts/process_greenspace.py


python scripts/process_imd.py
python scripts/process_house_prices.py
python scripts/process_healthcare_data.py
python scripts/process_primary_school_data.py
python scripts/process_secondary_school_data.py
python scripts/process_childcare.py
echo "  -> All individual indicator files created."


# --- 4. BUILD FINAL MASTER FILE
echo "\n--- STEP 4: Running Imputation Engine, Normalisation & Weighting and Aggregation ---"
python scripts/imputation_engine.py
python scripts/normalisation_script.py
python scripts/weighting_and_aggregation.py


echo "\n--- ALL DONE! ---"
echo "The 'data/processed' directory has been fully rebuilt."