import pandas as pd
import numpy as np
from pathlib import Path
import sys

#Paths and Constants
PROJECT_DIR = Path(__file__).resolve().parent.parent
MASTER_FILE = PROJECT_DIR / "data" / "processed" / "lsoa_annual_indicators.parquet"

def display_final_data():
    if not MASTER_FILE.exists():
        sys.exit("Error: Master file not found. Please run 'rebuild_data.sh' first.")
    df = pd.read_parquet(MASTER_FILE)
    #Select 5 random LSOAs for the final check
    lsoas = df['area_code'].unique()
    sample_lsoas = list(np.random.choice(lsoas, 5, replace=False))
    #Filter the data for the sample LSOAs across all years
    final_data = df[df['area_code'].isin(sample_lsoas)].sort_values(['area_code', 'year']).reset_index(drop=True)
    print("FINAL DATA ROWS FOR SAMPLE LSOAS ACROSS ALL YEARS (2018 - 2025)")
    print(f"LSOAs Sampled: {sample_lsoas}")
    print("=" * 120)
    #Displaying the full DataFrame content for audit purposes
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(final_data.to_string())
    print("END OF DATA AUDIT")

if __name__ == '__main__':
    display_final_data()