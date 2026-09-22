"""
src/build_sqlite.py
===================
Compiles processed CSV files and dataframes into a final SQLite database.
"""

import os
import sqlite3
import pandas as pd
from src.config import PROCESSED

def main():
    print("=" * 60)
    print("SQLITE BUILD")
    print("=" * 60)
    
    db_path = PROCESSED / "foodprice_nutrition.sqlite"
    
    if db_path.exists():
        db_path.unlink()
        
    conn = sqlite3.connect(db_path)
    
    # 1. district_alias
    df_alias = pd.read_csv(PROCESSED / "district_alias.csv")
    df_alias.to_sql("district_alias", conn, index=False, if_exists="replace")
    print(f"Loaded district_alias: {len(df_alias)} rows")
    
    # 2. prices_clean
    df_prices = pd.read_csv(PROCESSED / "prices_clean.csv.gz")
    df_prices.to_sql("prices_clean", conn, index=False, if_exists="replace")
    print(f"Loaded prices_clean: {len(df_prices)} rows")
    
    # 3. price_series_monthly
    df_monthly = pd.read_csv(PROCESSED / "price_series_monthly.csv")
    df_monthly.to_sql("price_series_monthly", conn, index=False, if_exists="replace")
    print(f"Loaded price_series_monthly: {len(df_monthly)} rows")
    
    # 4. volatility_series
    df_vol = pd.read_csv(PROCESSED / "volatility_series.csv")
    df_vol.to_sql("volatility_series", conn, index=False, if_exists="replace")
    print(f"Loaded volatility_series: {len(df_vol)} rows")
    
    # 5. nfhs_tn (outcomes + controls)
    from src.nfhs import load_outcomes, load_controls
    outcomes = load_outcomes()
    canonical = set(outcomes["district"])
    controls = load_controls(canonical)
    nfhs_tn = pd.merge(outcomes, controls, on="district", how="inner")
    nfhs_tn.to_sql("nfhs_tn", conn, index=False, if_exists="replace")
    print(f"Loaded nfhs_tn: {len(nfhs_tn)} rows")
    
    # 6. analysis_dataset
    df_analysis = pd.read_csv(PROCESSED / "clean_data.csv")
    df_analysis.to_sql("analysis_dataset", conn, index=False, if_exists="replace")
    print(f"Loaded analysis_dataset: {len(df_analysis)} rows")
    
    # Create indexes
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX idx_prices_dc ON prices_clean(district, commodity)")
    cursor.execute("CREATE INDEX idx_monthly_dc ON price_series_monthly(district, commodity)")
    conn.commit()
    
    # Verify count for check 5.4
    cursor.execute("SELECT COUNT(*) FROM analysis_dataset")
    analysis_count = cursor.fetchone()[0]
    print(f"\nSQL Test Query: SELECT COUNT(*) FROM analysis_dataset -> {analysis_count}")
    
    conn.close()
    
    size_mb = db_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved DB: {db_path.name}")
    print(f"Size: {size_mb:.1f} MB")
    
    if size_mb > 50:
        print("WARN: SQLite file > 50 MB, adding to .gitignore")
        gitignore_path = PROCESSED.parent.parent / ".gitignore"
        # Check if already in gitignore
        if gitignore_path.exists():
            with open(gitignore_path, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = ""
            
        if "foodprice_nutrition.sqlite" not in content:
            with open(gitignore_path, "a", encoding="utf-8") as f:
                f.write("\n# Auto-added by build_sqlite.py\nfoodprice_nutrition.sqlite\n")
    
if __name__ == "__main__":
    main()
