"""
src/features.py
===============
Temporal aggregation and volatility features for the Food Price Volatility pipeline.

Aggregates daily prices to monthly medians and computes volatility metrics per 
district-commodity across primary and extended windows.
"""

import sys
import numpy as np
import pandas as pd

from src.config import (
    INTERIM,
    PROCESSED,
    REPORTS,
    WINDOWS,
    MIN_DAYS_PER_MONTH,
    MIN_COMMODITIES_PER_DISTRICT,
    SPIKE_THRESHOLD_LOG,
)

def district_day_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (district, commodity, commodity_group, arrival_date), take the MEDIAN 
    of modal_price across all markets, varieties, and grades.
    """
    grouped = df.groupby(["district", "commodity", "commodity_group", "arrival_date"], dropna=False)
    dd = grouped["modal_price"].median().reset_index()
    return dd

def monthly_series(dd: pd.DataFrame) -> pd.DataFrame:
    """
    Roll daily prices to monthly: for each (district, commodity, commodity_group, year_month),
    take the MEDIAN of the district-day prices. Record n_days (distinct dates).
    Keep months with n_days >= MIN_DAYS_PER_MONTH.
    """
    dd = dd.copy()
    dd["year_month"] = pd.to_datetime(dd["arrival_date"]).dt.to_period("M")
    
    grouped = dd.groupby(["district", "commodity", "commodity_group", "year_month"], dropna=False)
    monthly = grouped.agg(
        monthly_price=("modal_price", "median"),
        n_days=("arrival_date", "nunique")
    ).reset_index()
    
    monthly = monthly[monthly["n_days"] >= MIN_DAYS_PER_MONTH].copy()
    return monthly

def series_metrics(prices: pd.Series, min_months: int, min_pairs: int) -> dict:
    """
    Computes volatility metrics on a monthly-indexed price series.
    prices: pandas Series indexed by Period("M"), sorted, no duplicates.
    """
    n_months = len(prices)
    
    if n_months < 2:
        n_pairs = 0
        log_returns = pd.Series(dtype=float)
    else:
        # Calculate log returns for consecutive months
        # We can check consecutive by comparing the diff of the index
        idx_diff = np.diff(prices.index.view('i8'))  # difference in months
        consecutive_mask = (idx_diff == 1)
        n_pairs = int(consecutive_mask.sum())
        
        # log_returns = ln(P_t / P_{t-1})
        # prices.values[1:] are P_t, prices.values[:-1] are P_{t-1}
        vals = prices.values
        all_r = np.log(vals[1:] / vals[:-1])
        log_returns = pd.Series(all_r[consecutive_mask])

    valid = (n_months >= min_months) and (n_pairs >= min_pairs)
    
    if not valid:
        return {
            "n_months": n_months,
            "n_pairs": n_pairs,
            "vol_logret_std": np.nan,
            "cv": np.nan,
            "range_ratio": np.nan,
            "spike_share": np.nan,
            "valid": False,
        }
        
    vol_logret_std = log_returns.std(ddof=1) if len(log_returns) > 1 else 0.0
    
    cv = prices.std(ddof=1) / prices.mean() if n_months > 1 else 0.0
    
    p95 = np.percentile(prices, 95)
    p5 = np.percentile(prices, 5)
    med = np.median(prices)
    range_ratio = (p95 - p5) / med if med > 0 else np.nan
    
    if len(log_returns) > 0:
        spike_share = (log_returns.abs() > SPIKE_THRESHOLD_LOG).mean()
    else:
        spike_share = 0.0
        
    return {
        "n_months": n_months,
        "n_pairs": n_pairs,
        "vol_logret_std": vol_logret_std,
        "cv": cv,
        "range_ratio": range_ratio,
        "spike_share": spike_share,
        "valid": True,
    }

def aggregate_to_district(vol_series_df: pd.DataFrame, window_prefix: str) -> pd.DataFrame:
    """
    Aggregate (district, commodity) metrics to district level using simple means.
    Outputs the features prefixed with window_prefix.
    """
    valid_df = vol_series_df[vol_series_df["valid"] == True].copy()
    
    # Calculate means per district and group
    grp_means = valid_df.groupby(["district", "commodity_group"])["vol_logret_std"].mean().unstack("commodity_group")
    
    # Calculate overall means
    dist_grp = valid_df.groupby("district")
    overall = dist_grp.agg(
        vol_logret_overall=("vol_logret_std", "mean"),
        cv_overall=("cv", "mean"),
        range_ratio_overall=("range_ratio", "mean"),
        spike_share_overall=("spike_share", "mean"),
        n_commodities_used=("commodity", "count")
    )
    
    # Merge grp_means and overall
    res = pd.concat([grp_means, overall], axis=1)
    
    # Fill missing columns if some groups don't exist
    for g in ["cereals", "pulses", "veg", "other"]:
        if g not in res.columns:
            res[g] = np.nan
            
    # Rename group columns
    res = res.rename(columns={
        "cereals": f"{window_prefix}vol_logret_cereals",
        "pulses": f"{window_prefix}vol_logret_pulses",
        "veg": f"{window_prefix}vol_logret_veg",
        "other": f"{window_prefix}vol_logret_other",
    })
    
    # Rename overall columns
    res = res.rename(columns={
        "vol_logret_overall": f"{window_prefix}vol_logret_overall",
        "cv_overall": f"{window_prefix}cv_overall",
        "range_ratio_overall": f"{window_prefix}range_ratio_overall",
        "spike_share_overall": f"{window_prefix}spike_share_overall",
        "n_commodities_used": f"{window_prefix}n_commodities_used",
    })
    
    res[f"{window_prefix}low_coverage"] = (res[f"{window_prefix}n_commodities_used"] < MIN_COMMODITIES_PER_DISTRICT).astype(int)
    
    return res

def main():
    print("=" * 60)
    print("TEMPORAL AGGREGATION & VOLATILITY FEATURES")
    print("=" * 60)
    
    # Load prices
    prices_path = PROCESSED / "prices_clean.csv.gz"
    print(f"Loading {prices_path.name}...")
    df = pd.read_csv(prices_path)
    
    # 1. District-day
    dd = district_day_prices(df)
    
    # 2. Monthly
    monthly = monthly_series(dd)
    
    # Save monthly (formatted as YYYY-MM)
    monthly_out = monthly.copy()
    monthly_out["year_month"] = monthly_out["year_month"].dt.strftime("%Y-%m")
    out_monthly = PROCESSED / "price_series_monthly.csv"
    monthly_out.to_csv(out_monthly, index=False)
    print(f"Saved monthly series: {out_monthly.name} ({len(monthly_out)} rows)")
    
    # 3. Process Windows
    all_vol_rows = []
    
    for win_name, win_cfg in WINDOWS.items():
        prefix = win_cfg["prefix"]
        start = pd.Period(win_cfg["start"], "M")
        end = pd.Period(win_cfg["end"], "M")
        min_months = win_cfg["min_months"]
        min_pairs = win_cfg["min_pairs"]
        
        win_monthly = monthly[(monthly["year_month"] >= start) & (monthly["year_month"] <= end)]
        
        for (dist, comm, grp), grp_df in win_monthly.groupby(["district", "commodity", "commodity_group"]):
            grp_df = grp_df.sort_values("year_month")
            prices_series = pd.Series(grp_df["monthly_price"].values, index=grp_df["year_month"].values)
            
            metrics = series_metrics(prices_series, min_months, min_pairs)
            
            row = {
                "window": win_name,
                "district": dist,
                "commodity": comm,
                "commodity_group": grp,
            }
            row.update(metrics)
            all_vol_rows.append(row)
            
    vol_series_df = pd.DataFrame(all_vol_rows)
    out_vol = PROCESSED / "volatility_series.csv"
    vol_series_df.to_csv(out_vol, index=False)
    print(f"Saved volatility series: {out_vol.name} ({len(vol_series_df)} rows)")
    
    # 4. District Aggregation
    nfhs_path = INTERIM / "nfhs_raw.csv"
    nfhs = pd.read_csv(nfhs_path)
    districts = sorted(nfhs["district"].str.strip().unique())
    district_df = pd.DataFrame({"district": districts}).set_index("district")
    
    for win_name, win_cfg in WINDOWS.items():
        prefix = win_cfg["prefix"]
        win_vol = vol_series_df[vol_series_df["window"] == win_name]
        agg = aggregate_to_district(win_vol, prefix)
        district_df = district_df.join(agg, how="left")
        
        # Fill NA for districts with zero valid commodities
        district_df[f"{prefix}n_commodities_used"] = district_df[f"{prefix}n_commodities_used"].fillna(0).astype(int)
        district_df[f"{prefix}low_coverage"] = (district_df[f"{prefix}n_commodities_used"] < MIN_COMMODITIES_PER_DISTRICT).astype(int)
        
    district_df = district_df.reset_index()
    
    # Order columns
    p_cols = [c for c in district_df.columns if c.startswith("p_")]
    e_cols = [c for c in district_df.columns if c.startswith("e_")]
    out_cols = ["district"] + p_cols + e_cols
    district_df = district_df[out_cols]
    
    out_dist = PROCESSED / "district_price_features.csv"
    district_df.to_csv(out_dist, index=False)
    print(f"Saved district features: {out_dist.name} ({len(district_df)} rows)")
    
    # Coverage sensitivity table for PRIMARY window
    primary_win = WINDOWS["primary"]
    p_start = pd.Period(primary_win["start"], "M")
    p_end = pd.Period(primary_win["end"], "M")
    p_monthly = monthly[(monthly["year_month"] >= p_start) & (monthly["year_month"] <= p_end)]
    
    sens_rows = []
    for m in [6, 9, 12]:
        p = m - 4
        n_valid_series = 0
        valid_dist_comm_counts = {d: 0 for d in districts}
        
        for (dist, comm), grp_df in p_monthly.groupby(["district", "commodity"]):
            grp_df = grp_df.sort_values("year_month")
            ps = pd.Series(grp_df["monthly_price"].values, index=grp_df["year_month"].values)
            met = series_metrics(ps, m, p)
            if met["valid"]:
                n_valid_series += 1
                valid_dist_comm_counts[dist] += 1
                
        n_dist_ge_3 = sum(1 for v in valid_dist_comm_counts.values() if v >= 3)
        n_dist_0 = sum(1 for v in valid_dist_comm_counts.values() if v == 0)
        
        sens_rows.append({
            "min_months": m,
            "min_pairs": p,
            "valid_series": n_valid_series,
            "districts_ge_3_comms": n_dist_ge_3,
            "districts_0_comms": n_dist_0
        })
        
    sens_df = pd.DataFrame(sens_rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "tables").mkdir(parents=True, exist_ok=True)
    out_sens = REPORTS / "tables" / "coverage_sensitivity.csv"
    sens_df.to_csv(out_sens, index=False)
    print(f"Saved sensitivity table: {out_sens.name}")
    
    # Checks & Reporting
    print("\n--- Checks ---")
    
    nfhs_path = INTERIM / "nfhs_raw.csv"
    nfhs = pd.read_csv(nfhs_path)
    nfhs_names = set(nfhs["district"].str.strip())
    
    districts_match = (set(district_df["district"]) == nfhs_names) and len(district_df) == 32
    print(f"  Exact 32 rows & matching NFHS canonical names: {'PASS' if districts_match else 'FAIL'}")
    
    # Validate series features
    valid_vol = vol_series_df[vol_series_df["valid"] == True]
    if len(valid_vol) > 0:
        vol_ok = (valid_vol["vol_logret_std"] >= 0).all()
        cv_ok = (valid_vol["cv"] >= 0).all()
        spike_ok = ((valid_vol["spike_share"] >= 0) & (valid_vol["spike_share"] <= 1)).all()
        finite_ok = np.isfinite(valid_vol[["vol_logret_std", "cv", "spike_share"]]).all().all()
        met_pass = vol_ok and cv_ok and spike_ok and finite_ok
    else:
        met_pass = True
        
    print(f"  Valid series metrics bounded & finite: {'PASS' if met_pass else 'FAIL'}")
    
    max_date = pd.to_datetime(monthly_out["year_month"]).max()
    date_pass = max_date <= pd.to_datetime("2023-12-31")
    print(f"  No months after 2023-12: {'PASS' if date_pass else 'FAIL'} (max is {max_date.strftime('%Y-%m')})")
    
    print("\nCoverage Report:")
    for win_name, win_cfg in WINDOWS.items():
        prefix = win_cfg["prefix"]
        low_cov_dist = district_df[district_df[f"{prefix}low_coverage"] == 1]["district"].tolist()
        ge_3_count = sum(district_df[f"{prefix}n_commodities_used"] >= 3)
        print(f"  [{win_name} window] Districts with >= 3 commodities: {ge_3_count}")
        print(f"  [{win_name} window] Low coverage districts: {low_cov_dist}")
        
    print("Done.")

if __name__ == "__main__":
    main()
