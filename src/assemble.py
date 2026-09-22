"""
src/assemble.py
===============
Merges NFHS data with district-level price features into a final dataset.
"""

import sys
import pandas as pd
from src.config import OUTCOMES, PROCESSED, EXPECTED_NFHS_ROWS

def assemble_dataset(sql_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforces the final column ordering.
    """
    assert len(sql_df) == EXPECTED_NFHS_ROWS, f"Expected {EXPECTED_NFHS_ROWS} rows, got {len(sql_df)}"
    assert sql_df["district"].is_unique, "Districts in sql_df are not unique"
    
    df = sql_df.copy()
        
    ctrl_cols = [c for c in df.columns if c.startswith("ctrl_")]
    p_cols = [c for c in df.columns if c.startswith("p_")]
    e_cols = [c for c in df.columns if c.startswith("e_")]
    
    # Check outcomes
    missing_outcomes = [o for o in OUTCOMES if o not in df.columns]
    assert not missing_outcomes, f"Missing outcomes: {missing_outcomes}"
    
    ordered_cols = (
        ["district"] +
        OUTCOMES + 
        ["households_surveyed"] +
        ctrl_cols +
        p_cols +
        e_cols
    )
    
    # Drop any extra columns not explicitly accounted for, or keep them? 
    # Instructions: Column order: district, outcomes, households_surveyed, ctrl_, p_, e_
    # Let's ensure all ordered cols exist
    for c in ordered_cols:
        assert c in df.columns, f"Expected column {c} missing from merged df"
        
    df = df[ordered_cols]
    return df

def main():
    print("=" * 60)
    print("DATASET ASSEMBLY")
    print("=" * 60)
    
    # We load NFHS from src.nfhs.main() or files directly
    from src.nfhs import load_outcomes, load_controls
    
    outcomes = load_outcomes()
    canonical = set(outcomes["district"])
    controls = load_controls(canonical)
    
    nfhs_merged = pd.merge(outcomes, controls, on="district", how="inner")
    
    # Load features
    features_path = PROCESSED / "district_price_features.csv"
    features = pd.read_csv(features_path)
    
    # Left join onto NFHS
    sql_df = pd.merge(nfhs_merged, features, on="district", how="left")
    
    final_df = assemble_dataset(sql_df)
    
    out_path = PROCESSED / "clean_data.csv"
    final_df.to_csv(out_path, index=False)
    
    print(f"Saved {out_path.name}: {len(final_df)} rows, {len(final_df.columns)} columns.")
    
if __name__ == "__main__":
    main()
