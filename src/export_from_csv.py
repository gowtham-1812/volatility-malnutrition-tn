"""
src/export_from_csv.py
======================
Exports raw data from local CSV files to data/interim/.

The raw price data lives in Data/archive/csv/<year>.csv (2018-2024).
The NFHS factsheet lives in data/external/NFHS_5_India_Districts_Factsheet_Data.csv.

Outputs:
  data/interim/prices_raw.csv  -- Tamil Nadu rows from all years (2018-2024)
  data/interim/nfhs_raw.csv    -- 32 Tamil Nadu district rows from NFHS-5

Note: exports ALL rows including 2024; date filtering happens in prices.py.
"""

import os
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Locate repository root and import config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (
    ROOT,
    INTERIM,
    EXTERNAL,
    EXPECTED_RAW_PRICE_ROWS,
    EXPECTED_NFHS_ROWS,
)

# ---------------------------------------------------------------------------
# Price CSV source directory (yearly archive files)
# ---------------------------------------------------------------------------
ARCHIVE_CSV_DIR = ROOT / "Data" / "archive" / "csv"

# Years to load - 2018 through 2024 inclusive (2024 rows stay in raw export
# but are excluded by MAX_DATE later; never deleted here).
PRICE_YEARS = list(range(2018, 2025))

# Column rename map: raw CSV headers -> canonical lowercase names
PRICE_COL_RENAME = {
    "State":          "state",
    "District":       "district",
    "Market":         "market",
    "Commodity":      "commodity",
    "Variety":        "variety",
    "Grade":          "grade",
    "Arrival_Date":   "arrival_date",
    "Min_Price":      "min_price",
    "Max_Price":      "max_price",
    "Modal_Price":    "modal_price",
    "Commodity_Code": "commodity_code",
}

# ---------------------------------------------------------------------------
# NFHS source file and column map
# Keys are the exact column names from the NFHS-5 factsheet CSV after stripping.
# ---------------------------------------------------------------------------
NFHS_SOURCE = ROOT / "data" / "external" / "NFHS_5_India_Districts_Factsheet_Data.csv"

# Exact column names (stripped) from the NFHS factsheet CSV.
# These were verified by reading the file header.
# Keys are the EXACT column names as they appear in the NFHS-5 factsheet CSV
# (after stripping whitespace). Verified by inspecting the actual file header.
# Note: 'severely wasted' uses footnote 19 (not 18 as in the DB-based pipeline);
#       BMI column includes the threshold value in its name.
# The anaemia column uses < (less-than sign, ASCII 0x3C) - check_ascii will
# flag the CSV itself only if it reads it; the column name below is ASCII-safe.
NFHS_COL_RENAME = {
    "District Names":
        "district",
    "State/UT":
        "state",
    "Number of Households surveyed":
        "households_surveyed",
    "Children under 5 years who are stunted (height-for-age)18 (%)":
        "stunting_pct",
    "Children under 5 years who are wasted (weight-for-height)18 (%)":
        "wasting_pct",
    # Footnote number is 19 in the CSV (not 18 as truncated in the old DB schema)
    "Children under 5 years who are severely wasted (weight-for-height)19 (%)":
        "severe_wasting_pct",
    "Children under 5 years who are underweight (weight-for-age)18 (%)":
        "underweight_pct",
    # The < character below is ASCII 0x3C (less-than), not a non-ASCII symbol
    "Children age 6-59 months who are anaemic (<11.0 g/dl)22 (%)":
        "anaemia_children_pct",
    "All women age 15-49 years who are anaemic22 (%)":
        "anaemia_women_pct",
    # Full column name includes BMI threshold; footnote is 21 in the CSV
    "Women (age 15-49 years) whose Body Mass Index (BMI) is below normal (BMI <18.5 kg/m2)21 (%)":
        "women_bmi_below_normal_pct",
}


def deduplicate_columns(df: pd.DataFrame, max_len: int = 63) -> pd.DataFrame:
    """Replicate PostgreSQL 63-char truncation to match column names in the factsheet.

    The NFHS CSV was originally loaded into PostgreSQL which silently truncates
    identifiers to 63 characters. We apply the same truncation so column lookups
    from the old DB-based scripts still work.
    """
    seen: dict = {}
    new_cols = []
    for col in df.columns:
        key = col[:max_len]
        if key in seen:
            seen[key] += 1
            suffix = f"_{seen[key]}"
            key = key[: max_len - len(suffix)] + suffix
        else:
            seen[key] = 1
        new_cols.append(key)
    df.columns = new_cols
    return df


def export_prices() -> int:
    """Read yearly CSVs, filter Tamil Nadu rows, write prices_raw.csv.

    Returns the total row count written.
    """
    print("=" * 60)
    print("EXPORTING PRICE DATA")
    print("=" * 60)

    INTERIM.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM / "prices_raw.csv"

    total_rows = 0
    first_file = True

    for year in PRICE_YEARS:
        csv_path = ARCHIVE_CSV_DIR / f"{year}.csv"
        assert csv_path.exists(), f"Missing price CSV: {csv_path}"

        print(f"\n  [{year}] Reading {csv_path.name} ...")
        df = pd.read_csv(csv_path, dtype=str)  # read as str to avoid parse issues
        raw_count = len(df)
        print(f"         Total rows in file  : {raw_count:,}")

        # Filter Tamil Nadu only
        df["State"] = df["State"].str.strip()
        tn = df[df["State"] == "Tamil Nadu"].copy()
        tn_count = len(tn)
        print(f"         Tamil Nadu rows      : {tn_count:,}")

        # Rename columns to canonical lowercase names
        tn = tn.rename(columns=PRICE_COL_RENAME)
        # Keep only the 11 expected columns in order
        expected_cols = list(PRICE_COL_RENAME.values())
        tn = tn[expected_cols]

        # Append to output CSV (write header only on first chunk)
        tn.to_csv(out_path, mode="w" if first_file else "a",
                  header=first_file, index=False)
        first_file = False
        total_rows += tn_count

    print(f"\n  Total Tamil Nadu rows written : {total_rows:,}")
    print(f"  Output file                  : {out_path}")

    # Verify row count
    assert total_rows == EXPECTED_RAW_PRICE_ROWS, (
        f"Row count mismatch: expected {EXPECTED_RAW_PRICE_ROWS}, got {total_rows}"
    )
    print(f"  ASSERT PASS: row count == {EXPECTED_RAW_PRICE_ROWS:,}")
    return total_rows


def export_nfhs() -> int:
    """Read NFHS factsheet CSV, extract Tamil Nadu rows, write nfhs_raw.csv.

    Returns the row count written.
    """
    print("\n" + "=" * 60)
    print("EXPORTING NFHS DATA")
    print("=" * 60)

    assert NFHS_SOURCE.exists(), f"Missing NFHS CSV: {NFHS_SOURCE}"
    print(f"  Reading {NFHS_SOURCE.name} ...")

    nfhs = pd.read_csv(NFHS_SOURCE, encoding="utf-8")
    raw_count = len(nfhs)
    print(f"  Raw rows (all India + header rows) : {raw_count}")
    print(f"  Columns (first 5): {list(nfhs.columns[:5])}")

    # Strip column names
    nfhs.columns = nfhs.columns.str.strip()

    # Apply 63-char deduplication (mirrors the old PostgreSQL pipeline)
    nfhs = deduplicate_columns(nfhs)

    # Drop repeated header rows (some NFHS CSVs have "District Names" repeated)
    nfhs = nfhs[nfhs["District Names"].str.strip() != "District Names"]
    after_dedup = len(nfhs)
    print(f"  After removing repeated header rows : {after_dedup}")

    # Check if the truncated column names we need actually exist
    # The NFHS factsheet has very long column names; after truncation some differ
    # from the original. We build a flexible match by checking if our key exists
    # exactly or if the CSV has the full (non-truncated) version.
    full_name_map = {}
    for original_key, target_col in NFHS_COL_RENAME.items():
        if original_key in nfhs.columns:
            full_name_map[original_key] = target_col
        else:
            # Try matching by 63-char prefix (for columns that were truncated)
            prefix = original_key[:63]
            matches = [c for c in nfhs.columns if c == prefix or c.startswith(prefix[:60])]
            if matches:
                full_name_map[matches[0]] = target_col
            else:
                print(f"  WARNING: Could not find column for '{target_col}' "
                      f"(looked for '{original_key[:63]}')")

    # Filter Tamil Nadu rows
    nfhs["State/UT"] = nfhs["State/UT"].str.strip()
    tn_nfhs = nfhs[nfhs["State/UT"] == "Tamil Nadu"].copy()
    tn_count = len(tn_nfhs)
    print(f"  Tamil Nadu rows                     : {tn_count}")

    # Rename columns using the resolved map
    tn_nfhs = tn_nfhs.rename(columns=full_name_map)

    # Keep only the 9 expected output columns
    out_cols = list(NFHS_COL_RENAME.values())
    missing_out = [c for c in out_cols if c not in tn_nfhs.columns]
    assert not missing_out, (
        f"Missing expected NFHS output columns after rename: {missing_out}"
    )
    tn_nfhs = tn_nfhs[out_cols].reset_index(drop=True)

    # Write to interim
    out_path = INTERIM / "nfhs_raw.csv"
    tn_nfhs.to_csv(out_path, index=False)
    print(f"  Output file                         : {out_path}")
    print(f"  Columns written                     : {list(tn_nfhs.columns)}")

    # Verify row count
    assert tn_count == EXPECTED_NFHS_ROWS, (
        f"NFHS row count mismatch: expected {EXPECTED_NFHS_ROWS}, got {tn_count}"
    )
    print(f"  ASSERT PASS: NFHS row count == {EXPECTED_NFHS_ROWS}")
    return tn_count


def main() -> None:
    """Run both exports and print a summary."""
    price_rows = export_prices()
    nfhs_rows = export_nfhs()

    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)
    print(f"  prices_raw.csv  : {price_rows:,} rows")
    print(f"  nfhs_raw.csv    : {nfhs_rows} rows")
    print("  Done.")


if __name__ == "__main__":
    main()
