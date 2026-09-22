"""
src/nfhs.py
===========
Extract NFHS outcomes and controls, validate values, and merge them.
"""

import sys
import pandas as pd
from src.config import INTERIM, EXTERNAL, NFHS_CONTROL_RULES, EXPECTED_NFHS_ROWS

def load_outcomes() -> pd.DataFrame:
    """Load and validate the pre-extracted NFHS outcomes."""
    df = pd.read_csv(INTERIM / "nfhs_raw.csv")
    df["district"] = df["district"].str.strip()
    
    assert len(df) == EXPECTED_NFHS_ROWS, f"Expected {EXPECTED_NFHS_ROWS} outcomes, got {len(df)}"
    
    pct_cols = [c for c in df.columns if c.endswith("_pct")]
    for col in pct_cols:
        invalid = df[(df[col] < 0) | (df[col] > 100)]
        assert len(invalid) == 0, f"Column {col} has values outside [0, 100]"
        
    return df

def parse_nfhs_value(val):
    """
    Parse an NFHS factsheet string value.
    - strip whitespace
    - (x) -> x
    - * or empty -> NaN
    """
    if pd.isna(val):
        return pd.NA, False
    val = str(val).strip()
    if val == "" or val == "*":
        return pd.NA, False
    
    is_paren = False
    if val.startswith("(") and val.endswith(")"):
        val = val[1:-1].strip()
        is_paren = True
        
    return val, is_paren

def load_controls(canonical_districts: set) -> pd.DataFrame:
    """
    Load external factsheet data, extract Tamil Nadu rows, 
    dynamically match columns from NFHS_CONTROL_RULES, 
    parse their values, and return a DataFrame.
    """
    df = pd.read_csv(EXTERNAL / "NFHS_5_India_Districts_Factsheet_Data.csv", dtype=str)
    # Strip column names
    df.columns = df.columns.str.strip()
    
    # Filter for Tamil Nadu
    state_col = "State/UT"
    dist_col = "District Names"
    
    mask_state = df[state_col].str.strip().str.lower() == "tamil nadu"
    mask_header = df[dist_col].str.strip() != "District Names"
    
    tn_df = df[mask_state & mask_header].copy()
    tn_df["district"] = tn_df[dist_col].str.strip()
    
    cols_lower = {c: c.lower() for c in tn_df.columns}
    
    controls_df = tn_df[["district"]].copy()
    
    for concept, rule in NFHS_CONTROL_RULES.items():
        keywords = [kw.lower() for kw in rule["keywords"]]
        exclude_keywords = [kw.lower() for kw in rule.get("exclude_keywords", [])]
        
        matches = []
        for orig_col, lower_col in cols_lower.items():
            if all(kw in lower_col for kw in keywords):
                if not any(ex_kw in lower_col for ex_kw in exclude_keywords):
                    matches.append(orig_col)
                
        if len(matches) == 1:
            orig_name = matches[0]
            print(f"Match found for '{concept}': {orig_name}")
            
            # Parse values
            vals = tn_df[orig_name].apply(parse_nfhs_value)
            raw_vals = [v[0] for v in vals]
            parens = [v[1] for v in vals]
            
            numeric_vals = pd.to_numeric(raw_vals, errors="coerce")
            
            n_paren = sum(parens)
            n_nan = pd.isna(numeric_vals).sum()
            
            controls_df[f"ctrl_{concept}"] = numeric_vals
            print(f"  -> {n_paren} parenthesized, {n_nan} NaN")
            
        elif len(matches) == 0:
            if rule["required"]:
                raise ValueError(f"Required concept '{concept}' had zero matches. Keywords: {keywords}")
            else:
                print(f"Match NOT found for optional concept '{concept}'")
        else:
            raise ValueError(f"Concept '{concept}' had multiple matches: {matches}")
            
    # Validate canonical match
    districts_in_ctrl = set(controls_df["district"])
    unmatched = canonical_districts - districts_in_ctrl
    assert len(controls_df) == EXPECTED_NFHS_ROWS and not unmatched, \
        f"Control extraction got {len(controls_df)} rows. Missing districts: {unmatched}"
        
    return controls_df

def main():
    print("=" * 60)
    print("NFHS DATA EXTRACTION")
    print("=" * 60)
    
    outcomes = load_outcomes()
    canonical = set(outcomes["district"])
    
    print(f"Loaded {len(outcomes)} NFHS outcomes.")
    
    controls = load_controls(canonical)
    
    merged = pd.merge(outcomes, controls, on="district", how="inner")
    assert len(merged) == EXPECTED_NFHS_ROWS, f"Merged expected {EXPECTED_NFHS_ROWS} but got {len(merged)}"
    
    return merged

if __name__ == "__main__":
    main()
