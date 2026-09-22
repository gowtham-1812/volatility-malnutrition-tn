"""
src/districts.py
================
District name standardization for the Food Price Volatility pipeline.

Provides two public functions:
  normalize_name(s)                     -> str
  build_alias_table(price_names, nfhs_names) -> pd.DataFrame

The matching order (applied per raw price-side district string):
  1. exact      - normalized raw equals normalized NFHS name
  2. alias      - raw is in a DISTRICT_SPELLING_GROUPS group whose only NFHS
                  member provides the canonical name
  3. parent_merge - raw is a key of NEW_DISTRICT_PARENT; resolve parent through
                  rules 1 or 2 to get the canonical NFHS name
  4. unmatched  - none of the above (pipeline must stop and ask user)
"""

import re
import difflib
import pandas as pd

from src.config import (
    DISTRICT_SPELLING_GROUPS,
    NEW_DISTRICT_PARENT,
    INTERIM,
    EXTERNAL,
)


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def normalize_name(s: str) -> str:
    """Normalize a district name to a canonical lowercase form.

    Steps:
      - strip leading/trailing whitespace
      - lowercase
      - replace & with 'and'
      - collapse repeated whitespace to a single space
      - remove a leading 'the ' prefix
      - remove trailing periods
    """
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"\s+", " ", s)
    if s.startswith("the "):
        s = s[4:]
    s = s.rstrip(".")
    return s


# ---------------------------------------------------------------------------
# Alias table builder
# ---------------------------------------------------------------------------

def _build_spelling_lookup() -> dict:
    """Build a mapping: normalized spelling -> group index in DISTRICT_SPELLING_GROUPS."""
    lookup = {}
    for idx, group in enumerate(DISTRICT_SPELLING_GROUPS):
        for spelling in group:
            norm = normalize_name(spelling)
            lookup[norm] = idx
    return lookup


def build_alias_table(
    price_district_names: list,
    nfhs_district_names: list,
) -> pd.DataFrame:
    """Build a mapping from every distinct raw price-side district to its canonical NFHS name.

    Returns a DataFrame with columns:
      raw_name, normalized, canonical, rule, note

    Rules applied in order: exact, alias, parent_merge.
    Unmatched entries are included with rule='unmatched' and canonical=''.
    Callers must check for unmatched rows and stop if any exist.
    """
    # Normalize NFHS names (these are the canonical targets; never rename them)
    nfhs_norm_to_raw = {normalize_name(n): n.strip() for n in nfhs_district_names}
    nfhs_norm_set = set(nfhs_norm_to_raw.keys())

    # Pre-build spelling-group lookup
    spelling_lookup = _build_spelling_lookup()

    # Pre-build parent lookup: normalized child -> normalized parent
    parent_lookup = {normalize_name(k): normalize_name(v)
                     for k, v in NEW_DISTRICT_PARENT.items()}

    rows = []
    for raw in price_district_names:
        norm = normalize_name(raw)
        canonical = ""
        rule = "unmatched"
        note = ""

        # Rule 1: exact match
        if norm in nfhs_norm_set:
            canonical = nfhs_norm_to_raw[norm]
            rule = "exact"

        # Rule 2: alias (spelling group)
        elif norm in spelling_lookup:
            group_idx = spelling_lookup[norm]
            group = DISTRICT_SPELLING_GROUPS[group_idx]
            # Find how many members of this group appear in the NFHS set
            nfhs_hits = [normalize_name(sp) for sp in group
                         if normalize_name(sp) in nfhs_norm_set]
            if len(nfhs_hits) == 1:
                canonical = nfhs_norm_to_raw[nfhs_hits[0]]
                rule = "alias"
            elif len(nfhs_hits) == 0:
                rule = "unmatched"
                note = f"spelling group {group_idx} has no NFHS member"
            else:
                rule = "unmatched"
                note = f"spelling group {group_idx} matches multiple NFHS names: {nfhs_hits}"

        # Rule 3: parent merge
        elif norm in parent_lookup:
            parent_norm = parent_lookup[norm]
            # Resolve parent through rules 1 or 2
            if parent_norm in nfhs_norm_set:
                canonical = nfhs_norm_to_raw[parent_norm]
                rule = "parent_merge"
                note = "merged into parent to match NFHS-5 boundaries"
            elif parent_norm in spelling_lookup:
                group_idx = spelling_lookup[parent_norm]
                group = DISTRICT_SPELLING_GROUPS[group_idx]
                nfhs_hits = [normalize_name(sp) for sp in group
                             if normalize_name(sp) in nfhs_norm_set]
                if len(nfhs_hits) == 1:
                    canonical = nfhs_norm_to_raw[nfhs_hits[0]]
                    rule = "parent_merge"
                    note = "merged into parent to match NFHS-5 boundaries"
                else:
                    note = f"parent '{parent_norm}' resolved via alias but unclear: {nfhs_hits}"
            else:
                note = f"parent '{parent_norm}' not found in NFHS set"

        rows.append({
            "raw_name": raw,
            "normalized": norm,
            "canonical": canonical,
            "rule": rule,
            "note": note,
        })

    df = pd.DataFrame(rows, columns=["raw_name", "normalized", "canonical", "rule", "note"])
    return df


def difflib_suggestions(name: str, nfhs_names: list, n: int = 3) -> list:
    """Return up to n closest difflib matches for a district name (for reporting only).

    These suggestions are printed in reports only; never applied to data.
    """
    norm = normalize_name(name)
    nfhs_norms = [normalize_name(n) for n in nfhs_names]
    matches = difflib.get_close_matches(norm, nfhs_norms, n=n, cutoff=0.0)
    return matches


# ---------------------------------------------------------------------------
# Main: run district matching and produce output files
# ---------------------------------------------------------------------------

def main() -> None:
    """Run district name standardization and write output files."""
    import sys
    from src.config import PROCESSED, REPORTS

    print("=" * 60)
    print("DISTRICT NAME STANDARDIZATION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load NFHS canonical names
    # ------------------------------------------------------------------
    nfhs_path = INTERIM / "nfhs_raw.csv"
    assert nfhs_path.exists(), f"Missing: {nfhs_path}"
    nfhs_df = pd.read_csv(nfhs_path)
    nfhs_df["district"] = nfhs_df["district"].str.strip()
    nfhs_names = nfhs_df["district"].tolist()

    print(f"\n  NFHS distinct districts : {len(nfhs_names)}")
    assert len(nfhs_names) == 32, f"Expected 32 NFHS districts, got {len(nfhs_names)}"
    print("  ASSERT PASS: NFHS count == 32")
    print(f"  NFHS districts: {sorted(nfhs_names)}")

    # ------------------------------------------------------------------
    # Load distinct price-side district names from prices_raw.csv
    # ------------------------------------------------------------------
    prices_path = INTERIM / "prices_raw.csv"
    assert prices_path.exists(), f"Missing: {prices_path}"

    print(f"\n  Reading price district names from {prices_path.name} ...")
    # Read only the district column to avoid loading 1.5M rows fully
    price_df = pd.read_csv(prices_path, usecols=["district"])
    price_df["district"] = price_df["district"].str.strip()
    price_raw_names = sorted(price_df["district"].unique().tolist())
    print(f"  Distinct raw price district names : {len(price_raw_names)}")
    print(f"  Raw names: {price_raw_names}")

    # ------------------------------------------------------------------
    # Build alias table
    # ------------------------------------------------------------------
    print("\n  Building alias table ...")
    alias_df = build_alias_table(price_raw_names, nfhs_names)

    # ------------------------------------------------------------------
    # Report matching results
    # ------------------------------------------------------------------
    print("\n  Matching results:")
    for rule in ["exact", "alias", "parent_merge", "unmatched"]:
        subset = alias_df[alias_df["rule"] == rule]
        print(f"    {rule:15s}: {len(subset)} districts")
        for _, row in subset.iterrows():
            target = f"  -> {row['canonical']}" if row["canonical"] else "  -> UNMATCHED"
            print(f"      '{row['raw_name']}'{target}")

    # ------------------------------------------------------------------
    # Handle unmatched: print difflib suggestions then stop
    # ------------------------------------------------------------------
    unmatched = alias_df[alias_df["rule"] == "unmatched"]
    if len(unmatched) > 0:
        print("\n  ERROR: Unmatched price-side districts found!")
        for _, row in unmatched.iterrows():
            suggestions = difflib_suggestions(row["raw_name"], nfhs_names)
            print(f"    '{row['raw_name']}' -> top suggestions: {suggestions}")
        print("\n  STOP: Cannot proceed until all districts are matched.")
        print("  See Section 5.2 of project instructions for how to respond.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Check: every canonical value is one of the 32 NFHS names
    # ------------------------------------------------------------------
    nfhs_set = set(nfhs_names)
    bad_canonical = alias_df[~alias_df["canonical"].isin(nfhs_set)]
    assert len(bad_canonical) == 0, (
        f"Canonical values not in NFHS set: {bad_canonical['canonical'].tolist()}"
    )
    print("\n  ASSERT PASS: all canonical values are valid NFHS district names")

    # Check: no empty canonical
    assert alias_df["canonical"].notna().all() and (alias_df["canonical"] != "").all(), (
        "Some canonical values are empty or NaN"
    )
    print("  ASSERT PASS: no empty or NaN canonical values")

    # ------------------------------------------------------------------
    # Write district_alias.csv
    # ------------------------------------------------------------------
    PROCESSED.mkdir(parents=True, exist_ok=True)
    alias_out = PROCESSED / "district_alias.csv"
    alias_df.to_csv(alias_out, index=False)
    print(f"\n  Written: {alias_out} ({len(alias_df)} rows)")

    # ------------------------------------------------------------------
    # District match report: one row per NFHS district
    # ------------------------------------------------------------------
    # Count price rows per canonical district (using full price data)
    price_full = pd.read_csv(prices_path, usecols=["district"])
    price_full["district"] = price_full["district"].str.strip()
    # Map raw district -> canonical using alias table
    raw_to_canonical = dict(zip(alias_df["raw_name"], alias_df["canonical"]))
    price_full["canonical"] = price_full["district"].map(raw_to_canonical)
    price_row_counts = price_full.groupby("canonical").size().rename("price_rows_2018_2024")

    # Build one row per NFHS district
    match_rows = []
    for nfhs_name in sorted(nfhs_names):
        district_aliases = alias_df[alias_df["canonical"] == nfhs_name]
        raw_names_list = "; ".join(district_aliases["raw_name"].tolist())
        n_raw = len(district_aliases)
        n_price = int(price_row_counts.get(nfhs_name, 0))
        match_rows.append({
            "canonical": nfhs_name,
            "n_raw_names_mapped": n_raw,
            "raw_names": raw_names_list,
            "price_rows_2018_2024": n_price,
            "has_price_data": int(n_price > 0),
        })

    match_df = pd.DataFrame(match_rows)
    (REPORTS / "tables").mkdir(parents=True, exist_ok=True)
    match_out = REPORTS / "tables" / "district_match_report.csv"
    match_df.to_csv(match_out, index=False)
    print(f"  Written: {match_out} ({len(match_df)} rows)")

    # Report NFHS districts with no price data
    no_price = match_df[match_df["has_price_data"] == 0]["canonical"].tolist()
    print(f"\n  NFHS districts with NO price data: {no_price if no_price else 'none'}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DISTRICT MATCHING SUMMARY")
    print("=" * 60)
    print(f"  Raw price districts (before)  : {len(price_raw_names)}")
    print(f"  Canonical NFHS districts      : 32")
    print(f"  Exact matches                 : {len(alias_df[alias_df['rule'] == 'exact'])}")
    print(f"  Alias matches                 : {len(alias_df[alias_df['rule'] == 'alias'])}")
    print(f"  Parent merges                 : {len(alias_df[alias_df['rule'] == 'parent_merge'])}")
    print(f"  Unmatched                     : {len(alias_df[alias_df['rule'] == 'unmatched'])}")
    print(f"  Parent merges applied         : {alias_df[alias_df['rule'] == 'parent_merge']['raw_name'].tolist()}")
    print("  Done.")


if __name__ == "__main__":
    main()
