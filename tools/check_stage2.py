"""Quick Stage 2 checks script."""
import pandas as pd
from src.config import PROCESSED, REPORTS, INTERIM

# C2.1
nfhs = pd.read_csv(INTERIM / "nfhs_raw.csv")
nfhs_count = nfhs["district"].nunique()
status = "PASS" if nfhs_count == 32 else "FAIL"
print(f"C2.1 NFHS distinct districts: {nfhs_count} (expected 32) -> {status}")

# C2.2
alias = pd.read_csv(PROCESSED / "district_alias.csv")
prices = pd.read_csv(INTERIM / "prices_raw.csv", usecols=["district"])
prices["district"] = prices["district"].str.strip()
raw_in_prices = set(prices["district"].unique())
raw_in_alias = set(alias["raw_name"])
missing = raw_in_prices - raw_in_alias
status = "PASS" if not missing else "FAIL"
print(f"C2.2 All price districts in alias table: missing={missing} -> {status}")

# C2.3
nfhs_names = set(nfhs["district"].str.strip())
canonical_vals = set(alias["canonical"].dropna())
bad = canonical_vals - nfhs_names
status = "PASS" if not bad else "FAIL"
print(f"C2.3 All canonical values are NFHS names: bad={bad} -> {status}")

# C2.4
has_empty = alias["canonical"].isna().any() or (alias["canonical"] == "").any()
status = "PASS" if not has_empty else "FAIL"
print(f"C2.4 No empty canonical: -> {status}")

# C2.5
match = pd.read_csv(REPORTS / "tables" / "district_match_report.csv")
no_price = match[match["has_price_data"] == 0]["canonical"].tolist()
print(f"C2.5 NFHS districts with no price data: {no_price} -> PASS (reported, not dropped)")

print()
print("alias.csv preview:")
print(alias.to_string())
