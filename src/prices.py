"""
src/prices.py
=============
Price cleaning and staple commodity filter for the Food Price Volatility pipeline.

Input:  data/interim/prices_raw.csv  (1,506,952 Tamil Nadu rows, 2018-2024)
Output: data/processed/prices_clean.csv  (staple-only, cleaned, 2018-2023)
        reports/tables/commodities_all.csv  (all commodities before staple filter)

Cleaning steps applied in order:
  1. Strip whitespace on string columns
  2. Parse arrival_date; drop NaT rows
  3. Date filter: keep MIN_DATE <= arrival_date <= MAX_DATE (removes 2024)
  4. Numeric coercion; drop rows where modal_price is NaN or <= 0
  5. Map district to canonical name via district_alias.csv
  6. Staple keyword filter: assign commodity_group; drop non-staples
  7. Outlier removal: drop if |log10(price / commodity median)| > OUTLIER_LOG10_BAND
"""

import sys
import numpy as np
import pandas as pd

from src.config import (
    INTERIM,
    PROCESSED,
    EXTERNAL,
    REPORTS,
    MIN_DATE,
    MAX_DATE,
    OUTLIER_LOG10_BAND,
    STAPLE_KEYWORDS,
    STAPLE_EXCLUDE_KEYWORDS,
    EXPECTED_RAW_PRICE_ROWS,
)


# ---------------------------------------------------------------------------
# Staple commodity classification
# ---------------------------------------------------------------------------

def classify_commodity(name: str) -> str | None:
    """Return the commodity_group for a commodity name, or None if not a staple.

    Checks STAPLE_EXCLUDE_KEYWORDS first; a commodity matching any exclusion
    keyword is not a staple regardless of what else it matches.
    Then scans STAPLE_KEYWORDS in the fixed order: cereals, pulses, veg, other.
    Returns the FIRST matching group name, or None.
    """
    lower = name.lower().strip()

    # Exclusion check (takes priority over any keyword match)
    for excl in STAPLE_EXCLUDE_KEYWORDS:
        if excl in lower:
            return None

    # Keyword scan in config order
    for group in ["cereals", "pulses", "veg", "other"]:
        for kw in STAPLE_KEYWORDS[group]:
            if kw in lower:
                return group

    return None


def build_commodity_group_series(commodities: pd.Series) -> pd.Series:
    """Apply classify_commodity to a Series of commodity names. Returns Series of str|None."""
    return commodities.map(classify_commodity)


# ---------------------------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------------------------

def clean_prices() -> pd.DataFrame:
    """Run all 7 cleaning steps and return the cleaned DataFrame."""
    prices_path = INTERIM / "prices_raw.csv"
    alias_path = PROCESSED / "district_alias.csv"
    assert prices_path.exists(), f"Missing: {prices_path}"
    assert alias_path.exists(), f"Missing: {alias_path}"

    # ------------------------------------------------------------------
    # Load raw data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("PRICE CLEANING")
    print("=" * 60)
    print(f"\nLoading {prices_path.name} ...")
    df = pd.read_csv(prices_path, dtype=str)
    print(f"  Header   : {list(df.columns)}")
    print(f"  Dtypes   : {df.dtypes.to_dict()}")
    print(f"  First 5 rows:")
    print(df.head().to_string(index=False))

    # Row count raw row count
    raw_count = len(df)
    print(f"\n  Raw row count: {raw_count:,}")
    assert raw_count == EXPECTED_RAW_PRICE_ROWS, (
        f"Expected {EXPECTED_RAW_PRICE_ROWS:,} raw rows, got {raw_count:,}"
    )
    print(f"  ASSERT PASS: raw count == {EXPECTED_RAW_PRICE_ROWS:,}")

    # ------------------------------------------------------------------
    # Step 1: Strip whitespace on string columns
    # ------------------------------------------------------------------
    print("\n--- Step 1: Strip whitespace ---")
    str_cols = ["district", "market", "commodity", "variety", "grade"]
    for col in str_cols:
        df[col] = df[col].str.strip()
    after_step1 = len(df)
    print(f"  Rows after step 1 (strip): {after_step1:,}  (no rows dropped)")

    # ------------------------------------------------------------------
    # Step 2: Parse arrival_date; drop NaT
    # ------------------------------------------------------------------
    print("\n--- Step 2: Parse arrival_date ---")
    df["arrival_date"] = pd.to_datetime(df["arrival_date"], errors="coerce")
    nat_count = df["arrival_date"].isna().sum()
    df = df[df["arrival_date"].notna()].copy()
    after_step2 = len(df)
    print(f"  NaT dates dropped        : {nat_count:,}")
    print(f"  Rows after step 2        : {after_step2:,}")

    # ------------------------------------------------------------------
    # Step 3: Date filter MIN_DATE to MAX_DATE (removes 2024)
    # ------------------------------------------------------------------
    print(f"\n--- Step 3: Date filter [{MIN_DATE}, {MAX_DATE}] ---")
    min_dt = pd.Timestamp(MIN_DATE)
    max_dt = pd.Timestamp(MAX_DATE)

    # Count removed rows by year for reporting
    removed = df[(df["arrival_date"] < min_dt) | (df["arrival_date"] > max_dt)].copy()
    removed_by_year = removed.groupby(removed["arrival_date"].dt.year).size()
    print(f"  Rows removed by year:")
    for yr, cnt in removed_by_year.items():
        print(f"    {yr}: {cnt:,}")

    df = df[(df["arrival_date"] >= min_dt) & (df["arrival_date"] <= max_dt)].copy()
    after_step3 = len(df)
    print(f"  Rows after step 3        : {after_step3:,}")

    # Row counts by year (for Row counts by year)
    print("  Row counts by year (2018-2023):")
    year_counts = df.groupby(df["arrival_date"].dt.year).size()
    for yr, cnt in year_counts.items():
        print(f"    {yr}: {cnt:,}")

    # ------------------------------------------------------------------
    # Step 4: Numeric coercion; drop modal_price NaN or <= 0
    # ------------------------------------------------------------------
    print("\n--- Step 4: Numeric coercion ---")
    for col in ["min_price", "max_price", "modal_price"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_modal = df["modal_price"].isna() | (df["modal_price"] <= 0)
    n_bad = bad_modal.sum()
    df = df[~bad_modal].copy()
    after_step4 = len(df)
    print(f"  Rows with modal_price NaN or <=0 dropped: {n_bad:,}")
    print(f"  Rows after step 4        : {after_step4:,}")

    # ------------------------------------------------------------------
    # Step 5: Map district to canonical via alias table
    # ------------------------------------------------------------------
    print("\n--- Step 5: District canonicalization ---")
    alias_df = pd.read_csv(alias_path)
    raw_to_canonical = dict(zip(alias_df["raw_name"], alias_df["canonical"]))

    df["district_raw"] = df["district"].copy()
    df["district"] = df["district"].map(raw_to_canonical)

    empty_canonical = df["district"].isna() | (df["district"] == "")
    n_empty = empty_canonical.sum()
    assert n_empty == 0, f"Zero canonical assertion failed: {n_empty} rows with empty canonical"
    print(f"  ASSERT PASS: zero rows with empty canonical district")
    after_step5 = len(df)
    print(f"  Rows after step 5        : {after_step5:,}  (no rows dropped)")

    # ------------------------------------------------------------------
    # Build commodities_all.csv BEFORE staple filter (on 2018-2023 data)
    # ------------------------------------------------------------------
    print("\n--- Building commodities_all.csv (all commodities, pre-staple filter) ---")
    all_groups = build_commodity_group_series(df["commodity"])
    all_comm_df = (
        df.assign(commodity_group_or_unmatched=all_groups.fillna("unmatched"))
        .groupby(["commodity", "commodity_group_or_unmatched"])
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
        .reset_index(drop=True)
    )
    (REPORTS / "tables").mkdir(parents=True, exist_ok=True)
    all_comm_out = REPORTS / "tables" / "commodities_all.csv"
    all_comm_df.to_csv(all_comm_out, index=False)
    print(f"  Written: {all_comm_out} ({len(all_comm_df)} distinct commodities)")

    # Print top 30 unmatched commodities (for report review)
    unmatched_comms = (
        all_comm_df[all_comm_df["commodity_group_or_unmatched"] == "unmatched"]
        .nlargest(30, "rows")
    )
    print("\n  Top 30 unmatched (non-staple) commodities by row count:")
    print(unmatched_comms.to_string(index=False))

    # Print matched staple commodities grouped by commodity_group
    print("\n  Matched staple commodities by group:")
    staple_comms = all_comm_df[all_comm_df["commodity_group_or_unmatched"] != "unmatched"]
    for grp in ["cereals", "pulses", "veg", "other"]:
        grp_comms = staple_comms[staple_comms["commodity_group_or_unmatched"] == grp]
        print(f"    {grp} ({len(grp_comms)} commodities, {grp_comms['rows'].sum():,} rows):")
        for _, r in grp_comms.iterrows():
            print(f"      '{r['commodity']}': {r['rows']:,} rows")

    # ------------------------------------------------------------------
    # Step 6: Staple filter
    # ------------------------------------------------------------------
    print("\n--- Step 6: Staple filter ---")
    df["commodity_group"] = build_commodity_group_series(df["commodity"])
    n_non_staple = df["commodity_group"].isna().sum()
    df = df[df["commodity_group"].notna()].copy()
    after_step6 = len(df)
    print(f"  Non-staple rows dropped  : {n_non_staple:,}")
    print(f"  Rows after step 6        : {after_step6:,}")
    print(f"  Distinct staple commodities: {df['commodity'].nunique()}")
    print(f"  Groups: {df['commodity_group'].value_counts().to_dict()}")

    # ------------------------------------------------------------------
    # Step 7: Outlier removal per commodity
    # ------------------------------------------------------------------
    print("\n--- Step 7: Outlier removal ---")
    comm_medians = df.groupby("commodity")["modal_price"].median()
    df["_med"] = df["commodity"].map(comm_medians)
    df["_log_ratio"] = np.abs(np.log10(df["modal_price"] / df["_med"]))
    outlier_mask = df["_log_ratio"] > OUTLIER_LOG10_BAND
    n_outliers = outlier_mask.sum()

    # Report drops per commodity and per year
    outlier_by_comm = df[outlier_mask].groupby("commodity").size()
    outlier_by_year = df[outlier_mask].groupby(df.loc[outlier_mask, "arrival_date"].dt.year).size()
    print(f"  Total outlier rows dropped: {n_outliers:,}")
    print(f"  Outliers per commodity:")
    for comm, cnt in outlier_by_comm.items():
        print(f"    '{comm}': {cnt:,}")
    print(f"  Outliers per year:")
    for yr, cnt in outlier_by_year.items():
        print(f"    {yr}: {cnt:,}")

    df = df[~outlier_mask].copy()
    df = df.drop(columns=["_med", "_log_ratio"])
    after_step7 = len(df)
    print(f"  Rows after step 7        : {after_step7:,}")

    return df


def main() -> None:
    """Run price cleaning and write prices_clean.csv."""
    df = clean_prices()

    # ------------------------------------------------------------------
    # Select and order the exact 11 output columns (Columns order)
    # ------------------------------------------------------------------
    out_cols = [
        "district", "district_raw", "market", "commodity", "commodity_group",
        "variety", "grade", "arrival_date", "min_price", "max_price", "modal_price",
    ]
    df = df[out_cols].copy()

    # Format date as YYYY-MM-DD text
    df["arrival_date"] = df["arrival_date"].dt.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Checks Date range - Row counts by year, Columns order
    # ------------------------------------------------------------------
    print("\n--- Checks ---")

    dates = pd.to_datetime(df["arrival_date"])
    c31_min = dates.min().strftime("%Y-%m-%d")
    c31_max = dates.max().strftime("%Y-%m-%d")
    c31_pass = (c31_min >= MIN_DATE) and (c31_max <= MAX_DATE)
    print(f"  Date range [{c31_min}, {c31_max}]: {'PASS' if c31_pass else 'FAIL'}")

    c32_price_ok = (df["modal_price"] > 0).all()
    c32_nan_ok = df[["district", "commodity", "commodity_group", "arrival_date", "modal_price"]].notna().all().all()
    print(f"  modal_price > 0 everywhere: {'PASS' if c32_price_ok else 'FAIL'}")
    print(f"  No NaN in key columns: {'PASS' if c32_nan_ok else 'FAIL'}")

    from src.config import INTERIM
    nfhs = pd.read_csv(INTERIM / "nfhs_raw.csv")
    nfhs_names = set(nfhs["district"].str.strip())
    bad_districts = set(df["district"].unique()) - nfhs_names
    print(f"  All districts are canonical NFHS names: {'PASS' if not bad_districts else 'FAIL'} bad={bad_districts}")

    print(f"  Row counts by year:")
    year_counts = dates.dt.year.value_counts().sort_index()
    for yr, cnt in year_counts.items():
        print(f"         {yr}: {cnt:,}")

    c36_pass = list(df.columns) == out_cols
    print(f"  Exact 11 columns in correct order: {'PASS' if c36_pass else 'FAIL'}")
    print(f"       Columns: {list(df.columns)}")

    assert c31_pass, "Date range check failed"
    assert c32_price_ok and c32_nan_ok, "Price numeric check failed"
    assert not bad_districts, f"Canonical NFHS names check failed: {bad_districts}"
    assert c36_pass, "Columns order column order check failed"

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED / "prices_clean.csv.gz"
    df.to_csv(out_path, index=False, compression="gzip")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n  Written: {out_path}")
    print(f"  Rows   : {len(df):,}")
    print(f"  Size   : {size_mb:.1f} MB")

    # File size
    if size_mb > 50:
        print("  File size WARN: file > 50 MB - add prices_clean.csv to .gitignore")
    else:
        print(f"  File size PASS: {size_mb:.1f} MB < 50 MB - safe to commit")

    print("\n  All checks PASS.")
    print("  Done.")


if __name__ == "__main__":
    main()
