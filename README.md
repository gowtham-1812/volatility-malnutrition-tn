# Volatility in Food Prices and Nutritional Outcomes in Tamil Nadu

## Purpose
This project analyzes the relationship between staple food price volatility and
district-level malnutrition outcomes (stunting, wasting, underweight, and anaemia)
in Tamil Nadu, India. It combines 6 years of wholesale market price data with
NFHS-5 health survey data across all 32 Tamil Nadu districts, applies statistical
modeling and machine learning to test whether price instability predicts child
malnutrition risk.

**Research question:** Are districts with more volatile staple food prices
associated with higher child and maternal malnutrition?

**Key finding:** Random Forest achieves 66.6% accuracy (vs 44.4% dummy baseline)
but the permutation test (p = 0.069) indicates the result is not statistically
significant at 5%, likely due to the small sample (n=32 districts).

## Data Sources

- **Price Data**: Historical daily wholesale food prices from the Agricultural
  Marketing Information Network (Agmarknet, https://agmarknet.gov.in/).
  A Kaggle-mirror of the same data was used due to server outages during
  collection. Coverage: Tamil Nadu only, 2018-2024.
- **Health Data**: District-level malnutrition outcomes and socioeconomic
  controls from the National Family Health Survey 5 (NFHS-5) district
  factsheets (2019-2021), sourced from data.gov.in.

## How to Run

To quickly reproduce the final machine learning results and statistical findings without needing to download the raw 6GB data archive, you can run the streamlined single-file pipeline. This script reads directly from the pre-built SQLite database committed to this repository.

**Step 1 - Set up environment**
```bash
# Linux / macOS
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Step 2 - Run the pipeline**
```bash
python submission/cs2_pipeline.py
```
Expected runtime: ~2-3 minutes. This will print the final accuracy, F1 scores, and permutation test results to the terminal.

## File Map

```text
Project/
+-- data/
|   +-- external/      # NFHS-5 source CSV (committed, 598 KB)
|   +-- interim/       # Intermediate derived CSVs (git-ignored, large)
|   +-- processed/     # clean_data.csv (committed) + SQLite (git-ignored)
+-- src/
|   +-- config.py           # Global constants, thresholds, and file paths
|   +-- districts.py        # Step 1: maps 36 raw district names to 32 canonical
|   +-- prices.py           # Step 2: cleans prices, applies staple filter
|   +-- features.py         # Step 3: computes monthly medians + volatility metrics
|   +-- nfhs.py             # Step 4: extracts NFHS outcomes and controls
|   +-- assemble.py         # Step 5: merges all features into 32-row dataset
|   +-- build_sqlite.py     # Step 6: packages all tables into SQLite
|   +-- analysis.py         # Step 7: correlations, OLS regressions, figures
|   +-- ml.py               # Step 8: repeated nested CV + permutation test
|   +-- run_pipeline.py     # Master orchestrator (steps 1-8)
+-- submission/
|   +-- cs2_pipeline.py     # Single-file deliverable (reads from SQLite)
|   +-- interpretation.md   # Written interpretation of ML results
+-- tools/                  # Verification utilities
+-- tests/
|   +-- test_features.py    # Unit tests for feature computation
+-- reports/
    +-- figures/            # Generated matplotlib figures
    +-- tables/             # CSV tables (correlations, regression, descriptives)
    +-- ml/                 # ML cross-validation logs
```

## Key Config Parameters

All tunable constants live in [`src/config.py`](src/config.py). The most
important ones:

| Constant | Value | Meaning |
|----------|-------|---------|
| `MIN_DATE` / `MAX_DATE` | 2018-01-01 / 2023-12-31 | Analysis window |
| `WINDOWS["primary"]` | 2019-2021 | NFHS-5 fieldwork overlap period |
| `OUTLIER_LOG10_BAND` | 1.0 | Drop price if `|log10(p/median)| > 1` |
| `MIN_COMMODITIES_PER_DISTRICT` | 3 | Below this: flagged as low-coverage |
| `SPIKE_THRESHOLD_LOG` | ln(1.25) | >25% monthly jump = price spike |
| `ML_OUTER_SPLITS` / `ML_OUTER_REPEATS` | 5 / 10 | Nested CV configuration |
| `BOOTSTRAP_RESAMPLES` | 2000 | CI resamples for Spearman correlations |

## Limitations

- Only 32 districts, limiting statistical power significantly.
- Ecological (district-level) data cannot describe individual behavior or outcomes.
- NFHS-5 is a single cross-sectional snapshot (2019-2021); price and nutrition
  windows are adjacent but not concurrent.
- 21 of 32 districts are flagged low-coverage in the primary price window.
- Parent-district merging was applied for 5 post-NFHS new districts.
- Multiple testing corrections (FDR) applied; some associations may still be
  false positives given the small sample.
