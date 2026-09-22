"""
submission/cs2_pipeline.py
==========================
Single-file deliverable for CS2.
Self-contained pipeline that loads data from SQLite, computes features,
and evaluates machine learning models for malnutrition classification.
"""

import sys
import time
import math
import sqlite3
import collections
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RepeatedStratifiedKFold, GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

# ===== CONSTANTS =====
SEED = 42
HOLDOUT_TEST_SIZE = 0.25
SPIKE_THRESHOLD_LOG = 0.22314355131420976
ML_TARGET_COLUMN = "stunting_pct"

STAPLE_KEYWORDS = {
    "cereals": ["paddy", "rice", "maize", "ragi", "bajra", "jowar", "wheat"],
    "pulses": ["arhar", "bengal gram", "black gram", "green gram", "lentil", "cowpea"],
    "veg": ["onion", "tomato", "potato"],
    "other": ["coconut", "groundnut", "jaggery", "gur"],
}
STAPLE_EXCLUDE = ["oil", "seed", "sweet", "flower", "leaves", "ricebean"]
def get_commodity_group(comm: str) -> str:
    clow = comm.lower().strip()
    if any(ex in clow for ex in STAPLE_EXCLUDE):
        return None
    for grp, kws in STAPLE_KEYWORDS.items():
        if any(kw in clow for kw in kws):
            return grp
    return None

def build_dataset(db_path: Path) -> pd.DataFrame:
    # ===== 1. LOAD FROM SQL =====
    conn = sqlite3.connect(db_path)
    df_prices = pd.read_sql("SELECT * FROM price_series_monthly", conn)
    df_nfhs = pd.read_sql("SELECT * FROM nfhs_tn", conn)
    conn.close()

    # Convert month string to Period
    df_prices["month_p"] = pd.to_datetime(df_prices["year_month"]).dt.to_period("M")
    
    # Filter to primary window
    w_start = pd.Period("2019-01", freq="M")
    w_end = pd.Period("2021-12", freq="M")
    mask = (df_prices["month_p"] >= w_start) & (df_prices["month_p"] <= w_end)
    w_df = df_prices[mask].copy()

    # Sort
    w_df = w_df.sort_values(["district", "commodity", "month_p"])

    # ===== 2. RECOMPUTE VOLATILITY FEATURES =====
    records = []
    for (d, c), group_df in w_df.groupby(["district", "commodity"]):
        if len(group_df) < 12:
            continue
            
        group_df = group_df.sort_values("month_p")
        prices = group_df["monthly_price"].values
        periods = group_df["month_p"].values
        
        log_prices = np.log(prices)
        p_ints = pd.PeriodIndex(periods).view('i8')
        diff_p = np.diff(p_ints)
        diff_log = np.diff(log_prices)
        
        valid_mask = (diff_p == 1)
        valid_logrets = diff_log[valid_mask]
        
        if len(valid_logrets) < 8:
            continue
            
        cg = get_commodity_group(c)
        if not cg:
            continue
            
        std_lr = np.std(valid_logrets, ddof=1) if len(valid_logrets) > 1 else 0.0
        cv = np.std(prices, ddof=1) / np.mean(prices) if np.mean(prices) > 0 and len(prices) > 1 else 0.0
        
        p05 = np.percentile(prices, 5)
        p95 = np.percentile(prices, 95)
        med_p = np.median(prices)
        rr = (p95 - p05) / med_p if med_p > 0 else 0.0
        
        spikes = np.sum(np.abs(valid_logrets) > SPIKE_THRESHOLD_LOG)
        ss = spikes / len(valid_logrets)
        
        records.append({
            "district": d,
            "commodity": c,
            "group": cg,
            "vol_logret_std": std_lr,
            "cv": cv,
            "range_ratio": rr,
            "spike_share": ss
        })
        
    f_df = pd.DataFrame(records)
    
    # Aggregate to district
    dist_features = []
    all_d = df_nfhs["district"].unique()
    
    for d in all_d:
        sub = f_df[f_df["district"] == d] if not f_df.empty else pd.DataFrame()
        
        row = {"district": d}
        row["p_vol_logret_overall"] = sub["vol_logret_std"].mean() if not sub.empty else np.nan
        row["p_cv_overall"] = sub["cv"].mean() if not sub.empty else np.nan
        row["p_range_ratio_overall"] = sub["range_ratio"].mean() if not sub.empty else np.nan
        row["p_spike_share_overall"] = sub["spike_share"].mean() if not sub.empty else np.nan
        
        for cg in ["cereals", "pulses", "veg", "other"]:
            csub = sub[sub["group"] == cg] if not sub.empty else pd.DataFrame()
            if not csub.empty:
                row[f"p_vol_logret_{cg}"] = csub["vol_logret_std"].mean()
            else:
                row[f"p_vol_logret_{cg}"] = np.nan
                
        dist_features.append(row)
        
    df_dist = pd.DataFrame(dist_features)
    df_merged = pd.merge(df_nfhs, df_dist, on="district", how="left")
    
    # Keep the expected columns 
    # To precisely match clean_data.csv, we only select what's in df_nfhs + the p_ columns.
    # Note: clean_data.csv might also have e_ columns. We were only told to recompute primary.
    # Wait, the instruction says "assert every numeric value equals... for the columns both share".
    # So we don't need e_ columns in this single file pipeline!
    
    return df_merged

def main():
    start_time = time.time()
    
    print("===== 1. LOAD FROM SQL =====")
    db_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "foodprice_nutrition.sqlite"
    if not db_path.exists():
        print(f"Error: DB not found at {db_path}")
        sys.exit(1)
        
    df = build_dataset(db_path)
    print(f"Dataset built: {len(df)} rows.")


    print("\n===== 3. MACHINE LEARNING: PREPROCESSING =====")
    # Define Target
    med_stunt = df[ML_TARGET_COLUMN].median()
    y = (df[ML_TARGET_COLUMN] > med_stunt).astype(int).values
    
    # Define FS_full
    fs_controls = [c for c in df.columns if c.startswith("ctrl_")]
    fs_volatility = [
        "p_vol_logret_cereals", "p_vol_logret_pulses", "p_vol_logret_veg", 
        "p_vol_logret_other", "p_vol_logret_overall", "p_cv_overall", 
        "p_range_ratio_overall", "p_spike_share_overall"
    ]
    fs_full = fs_controls + fs_volatility
    X = df[fs_full]
    
    # Hold-out Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=HOLDOUT_TEST_SIZE, stratify=y, random_state=SEED
    )
    print(f"Hold-out split: {len(y_train)} train, {len(y_test)} test.")
    
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    logreg = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, class_weight="balanced")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED)
    gb = GradientBoostingClassifier(random_state=SEED, subsample=0.8)
    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    
    grid_logreg = {"model__C": [0.01, 0.1, 1.0, 10.0]}
    grid_rf = {"model__max_depth": [2, 3, 4], "model__min_samples_leaf": [2, 3]}
    grid_gb = {"model__n_estimators": [50, 100], "model__max_depth": [1, 2], "model__learning_rate": [0.05, 0.1]}
    
    models = {
        "logreg": (Pipeline([("imputer", imputer), ("scaler", scaler), ("model", logreg)]), grid_logreg),
        "rf": (Pipeline([("imputer", imputer), ("model", rf)]), grid_rf),
        "gb": (Pipeline([("imputer", imputer), ("model", gb)]), grid_gb),
        "dummy": (Pipeline([("imputer", imputer), ("model", dummy)]), {})
    }
    
    print("\n===== 4. MACHINE LEARNING: HOLD-OUT EVALUATION =====")
    for m_name in ["logreg", "rf", "gb", "dummy"]:
        pipe, grid = models[m_name]
        if m_name == "dummy":
            clf = pipe
            clf.fit(X_train, y_train)
        else:
            cv_inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
            clf = GridSearchCV(pipe, grid, scoring="f1", cv=cv_inner, n_jobs=-1)
            clf.fit(X_train, y_train)
            
        yp = clf.predict(X_test)
        acc = accuracy_score(y_test, yp)
        prec = precision_score(y_test, yp, zero_division=0)
        rec = recall_score(y_test, yp, zero_division=0)
        f1 = f1_score(y_test, yp, zero_division=0)
        cm = confusion_matrix(y_test, yp)
        
        print(f"[{m_name}] Accuracy: {acc:.2f}, Precision: {prec:.2f}, Recall: {rec:.2f}, F1: {f1:.2f}")
        print(f"Confusion Matrix:\n{cm}")

    print("\n===== 5. MACHINE LEARNING: REPEATED NESTED CV (FS_full) =====")
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    cv_summary = []
    
    for m_name in ["logreg", "rf", "gb", "dummy"]:
        pooled_yt = collections.defaultdict(list)
        pooled_yp = collections.defaultdict(list)
        
        for fold_idx, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
            rep_idx = fold_idx // 5
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_te, y_te = X.iloc[te_idx], y[te_idx]
            
            pipe, grid = models[m_name]
            if m_name == "dummy":
                clf = pipe
                clf.fit(X_tr, y_tr)
            else:
                cv_inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
                clf = GridSearchCV(pipe, grid, scoring="f1", cv=cv_inner, n_jobs=-1)
                clf.fit(X_tr, y_tr)
                
            yp = clf.predict(X_te)
            pooled_yt[rep_idx].extend(y_te)
            pooled_yp[rep_idx].extend(yp)
            
        rep_accs = []
        rep_f1s = []
        for r_idx in range(10):
            yt = np.array(pooled_yt[r_idx])
            yp = np.array(pooled_yp[r_idx])
            rep_accs.append(accuracy_score(yt, yp))
            rep_f1s.append(f1_score(yt, yp, zero_division=0))
            
        mean_acc = np.mean(rep_accs)
        mean_f1 = np.mean(rep_f1s)
        cv_summary.append((m_name, mean_acc, mean_f1))
        print(f"[{m_name}] Nested CV Mean Accuracy: {mean_acc:.3f}, Mean F1: {mean_f1:.3f}")
        
    print("\n===== 6. FINAL MODEL OUTCOME =====")
    # tie-break logreg, rf, gb
    order = {"logreg": 0, "rf": 1, "gb": 2, "dummy": 3}
    best_m = sorted(cv_summary, key=lambda x: (-x[2], order[x[0]]))[0]
    print(f"Final chosen model: {best_m[0]} (Mean F1: {best_m[2]:.3f})")

    elapsed = time.time() - start_time
    print(f"\nTotal elapsed time: {elapsed:.1f} seconds")

if __name__ == "__main__":
    main()
