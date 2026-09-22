"""
src/config.py
=============
All tunable constants for the Food Price Volatility pipeline.
Import this module; never hard-code numbers in other files.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Repository root and standard data directories
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

INTERIM   = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
EXTERNAL  = ROOT / "data" / "external"
REPORTS   = ROOT / "reports"

# ---------------------------------------------------------------------------
# Verbatim block from Section 3 of AGENT_INSTRUCTIONS.md
# ---------------------------------------------------------------------------

SEED = 42

# Date limits. Rows outside [MIN_DATE, MAX_DATE] are never used.
MIN_DATE = "2018-01-01"
MAX_DATE = "2023-12-31"

# Two price windows. "primary" matches the NFHS-5 fieldwork period (2019-2021).
# "extended" is a robustness check only.
WINDOWS = {
    "primary": {"prefix": "p_", "start": "2019-01-01", "end": "2021-12-31",
                "min_months": 12, "min_pairs": 8},
    "extended": {"prefix": "e_", "start": "2018-01-01", "end": "2023-12-31",
                 "min_months": 24, "min_pairs": 16},
}

MIN_DAYS_PER_MONTH = 1            # a month is valid if it has >= this many distinct dates
MIN_COMMODITIES_PER_DISTRICT = 3  # below this a district is flagged low_coverage
SPIKE_THRESHOLD_LOG = 0.22314355131420976   # ln(1.25): a >25 percent monthly jump
OUTLIER_LOG10_BAND = 1.0          # drop price if |log10(price / commodity median)| > 1

EXPECTED_RAW_PRICE_ROWS = 1506952
EXPECTED_NFHS_ROWS = 32



# Staple commodity matching: case-insensitive SUBSTRING match on the stripped
# commodity name. A name matching any EXCLUDE keyword is never a staple.
STAPLE_KEYWORDS = {
    "cereals": ["paddy", "rice", "maize", "ragi", "bajra", "jowar", "wheat"],
    "pulses": ["arhar", "bengal gram", "black gram", "green gram", "lentil", "cowpea"],
    "veg": ["onion", "tomato", "potato"],
    "other": ["coconut", "groundnut", "jaggery", "gur"],
}
STAPLE_EXCLUDE_KEYWORDS = ["oil", "seed", "sweet", "flower", "leaves", "ricebean"]

# Groups of spellings that refer to the SAME district. Values are lowercase,
# whitespace-collapsed, with a leading "the " removed. The canonical spelling is
# whichever spelling of the group appears in the NFHS table.
DISTRICT_SPELLING_GROUPS = [
    ["kancheepuram", "kanchipuram"],
    ["thiruvallur", "tiruvallur", "thiruvellore"],
    ["viluppuram", "villupuram", "vilupuram"],
    ["thoothukkudi", "thoothukudi", "tuticorin"],
    ["kanniyakumari", "kanyakumari", "nagercoil (kannyiakumari)"],
    ["tiruchirappalli", "tiruchirapalli", "tiruchchirappalli", "trichy", "thiruchirappalli"],
    ["tiruppur", "tirupur", "thirupur"],
    ["pudukkottai", "pudukottai"],
    ["nilgiris"],
    ["tiruvannamalai", "thiruvannamalai"],
    ["thiruvarur", "tiruvarur"],
    ["sivaganga", "sivagangai"],
    ["ramanathapuram", "ramnad"],
    ["thanjavur", "tanjore"],
    ["tirunelveli", "thirunelveli"],
]

# Districts created after the NFHS-5 boundaries were fixed. Their price rows are
# merged into the parent district so geography is consistent with NFHS-5.
# key = normalized new-district spelling, value = normalized parent spelling.
NEW_DISTRICT_PARENT = {
    "chengalpattu": "kancheepuram",
    "ranipet": "vellore",
    "tirupathur": "vellore",
    "tirupattur": "vellore",
    "thirupathur": "vellore",       # alternate spelling found in price data
    "kallakurichi": "viluppuram",
    "kallakuruchi": "viluppuram",   # alternate spelling found in price data
    "tenkasi": "tirunelveli",
    "mayiladuthurai": "nagapattinam",
}

# NFHS control indicators. Each concept needs a column whose name contains ALL
# keywords (case-insensitive). Required = must exist.
NFHS_CONTROL_RULES = {
    "women_literacy":          {"keywords": ["women", "literate"], "required": True},
    "improved_sanitation":     {"keywords": ["improved sanitation"], "required": True},
    "clean_cooking_fuel":      {"keywords": ["clean fuel for cooking"], "required": True},
    "improved_drinking_water": {"keywords": ["improved drinking-water"], "required": False},
    "health_insurance":        {"keywords": ["health insurance"], "required": False},
    "institutional_births":    {"keywords": ["institutional births"], "exclude_keywords": ["public"], "required": False},
    "adequate_diet":           {"keywords": ["adequate diet"], "exclude_keywords": ["breastfeed", "breastfed"], "required": False},
    "exclusive_breastfeeding": {"keywords": ["exclusively breastfed"], "required": False},
}

OUTCOMES = ["stunting_pct", "wasting_pct", "severe_wasting_pct", "underweight_pct",
            "anaemia_children_pct", "anaemia_women_pct", "women_bmi_below_normal_pct"]

# Analysis and ML settings
BOOTSTRAP_RESAMPLES = 2000
HOLDOUT_TEST_SIZE = 0.25
ML_TARGET_COLUMN = "stunting_pct"      # label = 1 if value > median of the 32 districts
ML_OUTER_SPLITS = 5
ML_OUTER_REPEATS = 10
ML_INNER_SPLITS = 3
PERMUTATION_N = 100
PERMUTATION_REPEATS = 3
