"""
tools/verify_submission.py
==========================
Verifies that the single-file CS2 deliverable reconstructs
the exact numeric dataset as the main pipeline.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path so we can import from submission/
sys.path.append(str(Path(__file__).resolve().parents[1]))
from submission.cs2_pipeline import build_dataset

def main():
    db_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "foodprice_nutrition.sqlite"
    csv_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "clean_data.csv"
    
    if not db_path.exists():
        print("Error: DB not found.")
        sys.exit(1)
    if not csv_path.exists():
        print("Error: clean_data.csv not found.")
        sys.exit(1)
        
    print("Building dataset via single-file pipeline...")
    sub_df = build_dataset(db_path)
    sub_df = sub_df.sort_values("district").reset_index(drop=True)
    
    print("Loading clean_data.csv...")
    main_df = pd.read_csv(csv_path)
    main_df = main_df.sort_values("district").reset_index(drop=True)
    
    # Get common numeric columns
    numeric_cols = main_df.select_dtypes(include=[np.number]).columns
    common_cols = [c for c in numeric_cols if c in sub_df.columns]
    
    print(f"Comparing {len(common_cols)} common numeric columns over 32 districts...")
    
    if len(sub_df) != 32 or len(main_df) != 32:
        print(f"Error: Row mismatch (sub={len(sub_df)}, main={len(main_df)})")
        sys.exit(1)
        
    errors = 0
    for col in common_cols:
        main_vals = main_df[col].values
        sub_vals = sub_df[col].values
        
        # handle nan equality
        mask_nan_main = np.isnan(main_vals)
        mask_nan_sub = np.isnan(sub_vals)
        
        if not np.array_equal(mask_nan_main, mask_nan_sub):
            print(f"Mismatch in NaNs for column {col}")
            errors += 1
            continue
            
        valid_mask = ~mask_nan_main
        if valid_mask.any():
            diff = np.abs(main_vals[valid_mask] - sub_vals[valid_mask])
            max_diff = np.max(diff)
            if max_diff > 1e-9:
                print(f"Mismatch in column {col}: Max absolute difference = {max_diff}")
                errors += 1
                
    if errors == 0:
        print("SUCCESS: The single-file submission perfectly recreates the dataset (within 1e-9 tolerance)!")
        sys.exit(0)
    else:
        print(f"FAILED: {errors} columns had mismatches.")
        sys.exit(1)

if __name__ == "__main__":
    main()
