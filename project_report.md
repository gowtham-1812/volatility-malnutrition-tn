# Project Overview & Detailed Methodology Report

## 1. Datasets Used & Final Features Selected
**Datasets Used**:
- **Price Data**: Historical daily wholesale food prices from the Agricultural Marketing Information Network (Agmarknet), utilized via a Kaggle mirror due to server outages.
- **Health Data**: Demographic and nutritional health outcomes (stunting, wasting, underweight, anaemia) sourced from the National Family Health Survey 5 (NFHS-5) factsheets from data.gov.in.

**Final Features Selected**:
The machine learning pipeline utilized two sets of features (`FS_full`), comprising:
1. **Control Features (`FS_controls`)**: Percentages of women's literacy, improved sanitation, clean cooking fuel, improved drinking water, health insurance coverage, institutional births, and adequate diet.
2. **Volatility Features (`FS_volatility`)**: `p_vol_logret_overall`, `p_vol_logret_cereals`, `p_vol_logret_pulses`, `p_vol_logret_veg`, `p_vol_logret_other`, `p_cv_overall` (Coefficient of Variation), `p_range_ratio_overall`, and `p_spike_share_overall`.

**Rationale for Feature Selection**:
The **volatility features** were engineered specifically to capture the dynamic economic instability of food access (e.g., standard deviation of log returns, spike share) over a 3-year primary window (2019-2021) prior to the NFHS survey. These are much stronger indicators of market turbulence than simple average prices. The **control features** were explicitly selected based on epidemiological literature to adjust for foundational socio-economic factors, ensuring we evaluate the true isolated effect of price volatility.

## 2. Tech Stack & Repository Structure
**Tech Stack**:
- **Language**: Python 3.10+
- **Data Manipulation**: `pandas`, `numpy`
- **Machine Learning**: `scikit-learn` (for pipelines, imputers, models, and nested cross-validation)
- **Statistical Analysis**: `statsmodels`, `scipy`
- **Database**: `sqlite3`
- **Visualization**: `matplotlib`, `seaborn`

**Repository Structure & Generated Files**:
The project is strictly organized to separate data ingestion from modeling logic:
- `data/`: Contains the original raw datasets (`interim/`) and final assembled outputs (`processed/`). 
  - *Generated files*: `clean_data.csv` (the final merged 32-row dataset used for ML), and `foodprice_nutrition.sqlite` (the relational database packaging of the same data). Note: Intermediate CSVs generated during feature engineering are isolated or tracked in separate archival folders (e.g., `CS1/`).
- `src/`: Core Python modules for execution (see Section 5 for module breakdown).
- `submission/`: Contains the streamlined CS2 single-file script (`cs2_pipeline.py`) that rebuilds the dataset entirely from SQL and runs the ML pipeline.
  - *Generated files*: `execution_output.txt` (a saved log of the pipeline's terminal output, providing proof of execution and final accuracy metrics).
- `reports/`: Automatically generated outputs. 
  - *Generated files*: The `figures/` subdirectory is populated with visual EDA outputs (e.g., `heatmap.png`, `ols_scatter.png`) every time the `src/analysis.py` script is run. These are used to manually interpret the statistical correlations.
- `tools/`: Utility scripts used for verification (e.g., checking if the SQLite dataset perfectly matches the final assembled CSV).

## 3a. Data Cleaning, Imputation, & Scaling
- **Cleaning**: Extensive cleaning was performed. For price data, daily modal prices were aggregated to robust monthly medians (requiring at least 15 days of data per month) to eliminate transient daily recording errors. Furthermore, legacy district names were mapped to their newly formed canonical boundaries (parent-district merging) so the price history perfectly aligned with the NFHS-5 factsheets.
- **Missing Values**: During the ML pipeline, any missing values (NaNs) caused by sparse commodity coverage in specific districts were handled using Scikit-Learn's `SimpleImputer(strategy="median")` to prevent data loss while maintaining distribution shapes.
- **Encoding & Scaling**: The final assembled dataset contained only numeric features (mostly percentages and continuous volatility metrics), eliminating the need for categorical encoding. We applied a `StandardScaler` inside the pipeline explicitly for the Logistic Regression model to ensure the gradient descent solver converged correctly on normalized scales.

## 3b. Exploratory Data Analysis (EDA)
Comprehensive EDA was conducted to validate statistical associations between volatility metrics and child stunting.
- **Statistical Tests**: We computed Spearman (non-parametric) and Pearson (parametric) correlation matrices, utilizing 2000 bootstrap resamples to construct 95% Confidence Intervals, and applied the Benjamini-Hochberg False Discovery Rate (FDR) correction to adjust for multiple testing.
- **Regression Diagnostics**: Ordinary Least Squares (OLS) models utilized HC3 robust standard errors to adjust for heteroskedasticity. We also ran Variance Inflation Factor (VIF) checks for multicollinearity and calculated Cook's distance, successfully isolating highly influential outlier districts (e.g., Nagapattinam, Kanniyakumari).
- **How to view Visual Outputs**: To view the generated correlation heatmaps, district outcome rankings, and OLS scatter plots, you can run the analysis script natively:
  ```bash
  python -m src.analysis
  ```
  The resulting visual charts will be automatically generated and saved in the `reports/figures/` directory.

## 4. Architecture: The Pipeline & Source Modules
The project is built as a modular pipeline to cleanly separate the complex logic of data ingestion, feature engineering, and statistical modeling. This separation of concerns makes the code easier to test, interpret, and modify compared to a single massive script.

**What the `src/` modules do:**
- `config.py`: Stores all global constants, file paths, and hyperparameters to ensure consistency across the project.
- `districts.py`: Contains dictionaries to map non-standard district names from different datasets into a single canonical naming convention (handling spelling variants and parent-district splits).
- `prices.py`: Ingests the raw wholesale price datasets, standardizes column names, filters for valid staple commodities, and exports a cleaned interim version.
- `features.py`: The core volatility engine. It groups daily prices into monthly medians, calculates mathematical variations over time (like standard deviations of log-returns), and outputs aggregate instability scores.
- `nfhs.py`: Uses Regex string parsing to extract structured numerical percentages from raw NFHS text factsheets.
- `assemble.py`: Acts as the final joiner, merging the processed NFHS outcomes and price volatility features perfectly on the canonical district names.
- `build_sqlite.py`: Packages the final merged DataFrames into a persistent relational SQLite database (`foodprice_nutrition.sqlite`) for reliable, strongly-typed downstream usage.
- `analysis.py`: Contains all the EDA logic (correlations, OLS regressions, diagnostic tests, and matplotlib chart generation).
- `ml.py`: Executes the final machine learning cross-validation and hold-out evaluations.
- `run_pipeline.py`: The master controller script that executes all the above modules in sequence.

## 5. Output Files Generated

| File | Location | Description |
|------|----------|-------------|
| `clean_data.csv` | `data/processed/` | Final 32-row analysis dataset (all features + outcomes) |
| `foodprice_nutrition.sqlite` | `data/processed/` | SQLite DB with 6 relational tables |
| `correlations_primary.csv` | `reports/tables/` | Spearman/Pearson correlations, bootstrap CIs, FDR q-values |
| `regression_results.csv` | `reports/tables/` | OLS regression coefficients per outcome |
| `descriptives.csv` | `reports/tables/` | Summary statistics for all numeric columns |
| `repeated_cv_summary.csv` | `reports/ml/` | Nested CV accuracy/F1 averaged across 50 outer folds |
| `feature_importance.csv` | `reports/ml/` | Random Forest permutation importances |
| `fig_corr_heatmap_primary.png` | `reports/figures/` | Spearman correlation heatmap |
| `fig_scatter_vol_vs_stunting.png` | `reports/figures/` | Key scatter: volatility vs stunting |
| `fig_ml_permutation_null.png` | `reports/figures/` | Permutation test null distribution |

## 6. Machine Learning Models Trained
We trained specific Scikit-Learn classifiers to predict whether a district's child stunting rate was above or below the state median.

**Training Methodology: Repeated Nested Cross-Validation**
Because the dataset is exceptionally small (32 districts), traditional Train/Test splits (e.g., 80/20) are extremely unstable. Depending on which random districts land in the test set, the accuracy could swing wildly. To counteract this, we utilized **Repeated Nested Cross-Validation (5-Fold, 10 Repeats)**. 
- **How it works**: The dataset is split into 5 chunks (folds). The model trains on 4 chunks and tests on the 1 remaining chunk, rotating until all chunks have been tested. This entire 5-fold process is then *repeated 10 times* with different random shuffles.
- **Why?**: This is considered a gold-standard approach in academia for small-n problems. It ensures that every single district is used for both training and testing multiple times, producing a highly robust, averaged performance metric that prevents "lucky" splits. Hyperparameter tuning (like tree depth) happens strictly in an inner loop, preventing data leakage.

**The Models (via Scikit-Learn)**:
1. `LogisticRegression`: Serves as a strong, interpretable linear baseline.
2. `RandomForestClassifier`: A robust tree-ensemble that handles non-linear interactions natively and resists overfitting through bagging (using default parameters with constrained depth).
3. `GradientBoostingClassifier`: A sequential tree-based ensemble method designed to minimize residual errors and pull maximum predictive power from small feature spaces.
4. `DummyClassifier(strategy="stratified")`: This is the **Dummy Baseline**. It does not learn from the features; it simply guesses the target variable randomly while respecting the true distribution of the classes in the dataset. It is strictly used as a baseline benchmark. If our trained ML models cannot significantly beat the Dummy Classifier, it means they are failing to find genuine predictive patterns in the data.

## 7. Model Interpretation & Accuracy Limits
The project's goal is to provide a data-driven approach to identifying vulnerable districts to help target market stabilization policies. We achieved this by treating child stunting as a binary classification problem (predicting High vs. Low vulnerability districts).

**Final Averaged Metrics on the Full Feature Set**:
- **Random Forest (Final Chosen Model)**:
  - Accuracy: 66.6%
  - F1 Score: 0.675
- **Gradient Boosting**: Accuracy: 65.6%
- **Logistic Regression**: Accuracy: 61.6%
- **Dummy Baseline**: Accuracy: 44.4%

**Why are the accuracies around 60-66%?**
While the Random Forest (66.6%) clearly outperforms random guessing (44.4%), the ceiling for accuracy is severely constrained by the **small sample size** of our dataset. We only have 32 districts in Tamil Nadu to train on. Extremely complex algorithms require thousands of rows to confidently learn patterns without memorizing noise (overfitting). Furthermore, predicting complex human health outcomes based purely on high-level macro-economic district averages is inherently noisy (an issue known in statistics as the *ecological fallacy*). 

**How could this be improved?**
To push accuracies into the 80%+ range, the dataset would need to transition from district-level ecological aggregates to household-level microdata. Tracking the exact food purchases and clinical malnutrition measurements of thousands of individual households over time would provide the granularity needed for high-confidence predictions. However, at the district policy level, a 66.6% model still offers mathematically grounded evidence to help governments identify which regions are most sensitive to price shocks.
