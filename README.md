# Volatility in Food Prices and Nutritional Outcomes in Tamil Nadu

## Purpose
This project analyzes the relationship between staple food price volatility and district-level malnutrition outcomes (stunting, wasting, underweight, and anaemia) in Tamil Nadu, India. By evaluating monthly wholesale price data and merging it with demographic and health surveys, the analysis applies machine learning and statistical modeling to explore how economic stability relates to public health indicators.

## Data Sources
- **Price Data**: Data originated from the Agricultural Marketing Information Network ([https://agmarknet.gov.in/](https://agmarknet.gov.in/)). As the primary server was frequently down during data collection, a Kaggle-style scraped mirror of the same portal was utilized for historical monthly wholesale prices.
- **Health Data**: District-level nutritional outcomes and socioeconomic controls were sourced from the National Family Health Survey 5 (NFHS-5) factsheet resources available at [data.gov.in](https://data.gov.in).

## Installation
Requires **Python 3.10+**.

**For Bash (Linux/macOS):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**For Windows PowerShell:**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## How to Run

### (a) Single-File Submission (CS2 Deliverable)
The streamlined pipeline reconstructs the dataset from SQL and evaluates the final ML models.
```bash
python submission/cs2_pipeline.py
```

### (b) Full Pipeline
To execute the complete end-to-end data processing, feature engineering, analysis, and ML pipeline from scratch (skipping the SQLite export step by default):
```bash
python src/run_pipeline.py
```

## File Map
```text
Project/
+-- data/
|   +-- external/      # NFHS-5 source CSV
|   +-- interim/       # Intermediate derived CSVs (git-ignored)
|   +-- processed/     # Final analysis dataset and SQLite database
+-- src/
|   +-- config.py           # Global constants and file paths
|   +-- export_from_csv.py  # Reads raw CSVs, filters Tamil Nadu rows
|   +-- districts.py        # District name alias mapping
|   +-- prices.py           # Price cleaning and staple filter
|   +-- features.py         # Time-series volatility calculation
|   +-- nfhs.py             # NFHS-5 health survey parsing
|   +-- assemble.py         # Dataset compilation (32-row merge)
|   +-- build_sqlite.py     # Packages processed data into SQLite
|   +-- analysis.py         # Statistical correlations and figures
|   +-- ml.py               # Machine learning cross-validation
|   +-- run_pipeline.py     # Master orchestrator script
+-- submission/
|   +-- cs2_pipeline.py     # Single-file CS2 deliverable (reads from SQLite)
+-- tools/                  # Verification and testing scripts
+-- reports/
    +-- figures/            # Generated matplotlib charts
    +-- tables/             # Correlation, regression, descriptive CSVs
    +-- ml/                 # ML cross-validation logs
```

## Data Flow
1. **Raw Ingestion**: `export_from_csv.py` reads yearly CSVs from `Data/archive/csv/`,
   filters Tamil Nadu rows, and writes `data/interim/prices_raw.csv` (1.5M rows).
2. **Standardization**: `districts.py` maps 36 raw district spellings to 32 canonical
   NFHS names. `prices.py` cleans and staple-filters to `prices_clean.csv.gz`.
3. **Volatility Engine**: `features.py` aggregates to monthly medians and computes
   log-return volatility, CV, spike share, and range ratio per district.
4. **Outcome Processing**: `nfhs.py` extracts 7 malnutrition outcomes and 8 control
   indicators from the NFHS-5 factsheet.
5. **Assembly**: `assemble.py` merges all features and outcomes into
   `data/processed/clean_data.csv` (32 rows - one per district).
6. **SQL Build**: `build_sqlite.py` packages all processed tables into
   `foodprice_nutrition.sqlite` for the submission script.
7. **Analysis & ML**: `analysis.py` produces correlation tables and figures;
   `ml.py` runs repeated nested cross-validation.

## Limitations
- Only 32 districts are analyzed, limiting statistical power.
- The analysis relies on ecological (district-level) data, which cannot describe individual-level behavior or outcomes.
- NFHS-5 represents one cross-sectional snapshot in time.
- Price data before 2024 is sparse, resulting in 21 low-coverage districts in the primary window.
- Parent-district merging was required to match historical price data with newly formed districts.
- Multiple testing was performed, meaning some associations may appear significant by chance alone.

