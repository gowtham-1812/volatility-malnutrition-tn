"""
src/analysis.py
===============
Analysis and visualization stage.
Computes descriptive stats, correlations, OLS regressions, and outputs plots.
"""

import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    PROCESSED,
    REPORTS,
    OUTCOMES,
    BOOTSTRAP_RESAMPLES,
    SEED,
)

def create_descriptives(df: pd.DataFrame):
    num_df = df.select_dtypes(include=[np.number])
    stats_list = []
    for col in num_df.columns:
        s = num_df[col].dropna()
        n = len(s)
        if n == 0:
            continue
        stats_list.append({
            "column": col,
            "n": n,
            "mean": s.mean(),
            "sd": s.std(ddof=1) if n > 1 else np.nan,
            "min": s.min(),
            "p25": s.quantile(0.25),
            "median": s.median(),
            "p75": s.quantile(0.75),
            "max": s.max()
        })
        
    desc_df = pd.DataFrame(stats_list)
    (REPORTS / "tables").mkdir(parents=True, exist_ok=True)
    desc_df.to_csv(REPORTS / "tables" / "descriptives.csv", index=False)

def compute_correlations(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    vol_vars = [
        f"{prefix}vol_logret_overall", f"{prefix}vol_logret_cereals", f"{prefix}vol_logret_pulses",
        f"{prefix}vol_logret_veg", f"{prefix}vol_logret_other", f"{prefix}cv_overall",
        f"{prefix}range_ratio_overall", f"{prefix}spike_share_overall"
    ]
    
    rng = np.random.default_rng(SEED)
    results = []
    
    for v in vol_vars:
        for out in OUTCOMES:
            subset = df[[v, out]].dropna()
            n = len(subset)
            if n < 3:
                continue
                
            xs = subset[v].values
            ys = subset[out].values
            
            rho, p_spearman = stats.spearmanr(xs, ys)
            r, p_pearson = stats.pearsonr(xs, ys)
            
            # Bootstrap CI for rho
            boot_rhos = []
            for _ in range(BOOTSTRAP_RESAMPLES):
                idx = rng.integers(0, n, size=n)
                x_boot = xs[idx]
                y_boot = ys[idx]
                
                # skip if constant
                if len(np.unique(x_boot)) == 1 or len(np.unique(y_boot)) == 1:
                    continue
                    
                rho_b, _ = stats.spearmanr(x_boot, y_boot)
                if not np.isnan(rho_b):
                    boot_rhos.append(rho_b)
                    
            if boot_rhos:
                ci_low = np.percentile(boot_rhos, 2.5)
                ci_high = np.percentile(boot_rhos, 97.5)
            else:
                ci_low = np.nan
                ci_high = np.nan
                
            results.append({
                "vol_var": v,
                "outcome": out,
                "n": n,
                "rho": rho,
                "p_value": p_spearman,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "r": r,
                "r_p_value": p_pearson
            })
            
    res_df = pd.DataFrame(results)
    
    if not res_df.empty:
        # FDR correction
        pvals = res_df["p_value"].values
        _, qvals, _, _ = multipletests(pvals, method="fdr_bh")
        res_df["q_value"] = qvals
        
    out_name = "correlations_primary.csv" if prefix == "p_" else "correlations_extended.csv"
    res_df.to_csv(REPORTS / "tables" / out_name, index=False)
    return res_df

def standardize(s):
    if len(s.dropna()) > 1:
        return (s - s.mean()) / s.std(ddof=1)
    return s

def run_regressions(df: pd.DataFrame):
    reg_outcomes = ["stunting_pct", "wasting_pct", "underweight_pct", "anaemia_children_pct"]
    
    results = []
    diagnostics = []
    
    for out in reg_outcomes:
        # Model A
        cols_A = [out, "p_vol_logret_overall"]
        sub_A = df[["district"] + cols_A].dropna()
        n_A = len(sub_A)
        if n_A > 0:
            y = sub_A[out]
            X = pd.DataFrame({"z_p_vol": standardize(sub_A["p_vol_logret_overall"])})
            X = sm.add_constant(X)
            mod = sm.OLS(y, X).fit(cov_type="HC3")
            
            for term in X.columns:
                results.append({
                    "model": "A", "outcome": out, "term": term,
                    "coef": mod.params[term], "se": mod.bse[term],
                    "ci_low": mod.conf_int().loc[term, 0],
                    "ci_high": mod.conf_int().loc[term, 1],
                    "p_value": mod.pvalues[term],
                    "r2": mod.rsquared, "adj_r2": mod.rsquared_adj, "n": n_A
                })
                
        # Model B
        cols_B = [out, "p_vol_logret_overall", "ctrl_women_literacy", "ctrl_improved_sanitation", "ctrl_clean_cooking_fuel"]
        sub_B = df[["district"] + cols_B].dropna()
        n_B = len(sub_B)
        if n_B > 0:
            y = sub_B[out]
            X = pd.DataFrame({
                "z_p_vol": standardize(sub_B["p_vol_logret_overall"]),
                "z_lit": standardize(sub_B["ctrl_women_literacy"]),
                "z_san": standardize(sub_B["ctrl_improved_sanitation"]),
                "z_fuel": standardize(sub_B["ctrl_clean_cooking_fuel"])
            })
            X_c = sm.add_constant(X)
            mod = sm.OLS(y, X_c).fit(cov_type="HC3")
            
            for term in X_c.columns:
                results.append({
                    "model": "B", "outcome": out, "term": term,
                    "coef": mod.params[term], "se": mod.bse[term],
                    "ci_low": mod.conf_int().loc[term, 0],
                    "ci_high": mod.conf_int().loc[term, 1],
                    "p_value": mod.pvalues[term],
                    "r2": mod.rsquared, "adj_r2": mod.rsquared_adj, "n": n_B
                })
                
            # VIF and Cooks
            vifs = {c: variance_inflation_factor(X_c.values, i) for i, c in enumerate(X_c.columns)}
            cooks, _ = mod.get_influence().cooks_distance
            max_cook = cooks.max()
            max_cook_idx = np.argmax(cooks)
            dist_max_cook = sub_B["district"].iloc[max_cook_idx]
            
            diagnostics.append({
                "model": "B", "outcome": out,
                "vif_z_p_vol": vifs.get("z_p_vol", np.nan),
                "vif_z_lit": vifs.get("z_lit", np.nan),
                "vif_z_san": vifs.get("z_san", np.nan),
                "vif_z_fuel": vifs.get("z_fuel", np.nan),
                "max_cooks_d": max_cook,
                "district_max_cooks_d": dist_max_cook,
                "influential": int(max_cook > 4 / n_B)
            })
            
        # Model B-ext
        cols_B_ext = [out, "e_vol_logret_overall", "ctrl_women_literacy", "ctrl_improved_sanitation", "ctrl_clean_cooking_fuel"]
        sub_B_ext = df[["district"] + cols_B_ext].dropna()
        n_B_ext = len(sub_B_ext)
        if n_B_ext > 0:
            y = sub_B_ext[out]
            X = pd.DataFrame({
                "z_e_vol": standardize(sub_B_ext["e_vol_logret_overall"]),
                "z_lit": standardize(sub_B_ext["ctrl_women_literacy"]),
                "z_san": standardize(sub_B_ext["ctrl_improved_sanitation"]),
                "z_fuel": standardize(sub_B_ext["ctrl_clean_cooking_fuel"])
            })
            X_c = sm.add_constant(X)
            mod = sm.OLS(y, X_c).fit(cov_type="HC3")
            
            for term in X_c.columns:
                results.append({
                    "model": "B-ext", "outcome": out, "term": term,
                    "coef": mod.params[term], "se": mod.bse[term],
                    "ci_low": mod.conf_int().loc[term, 0],
                    "ci_high": mod.conf_int().loc[term, 1],
                    "p_value": mod.pvalues[term],
                    "r2": mod.rsquared, "adj_r2": mod.rsquared_adj, "n": n_B_ext
                })
                
            vifs = {c: variance_inflation_factor(X_c.values, i) for i, c in enumerate(X_c.columns)}
            cooks, _ = mod.get_influence().cooks_distance
            max_cook = cooks.max()
            max_cook_idx = np.argmax(cooks)
            dist_max_cook = sub_B_ext["district"].iloc[max_cook_idx]
            
            diagnostics.append({
                "model": "B-ext", "outcome": out,
                "vif_z_e_vol": vifs.get("z_e_vol", np.nan),
                "vif_z_lit": vifs.get("z_lit", np.nan),
                "vif_z_san": vifs.get("z_san", np.nan),
                "vif_z_fuel": vifs.get("z_fuel", np.nan),
                "max_cooks_d": max_cook,
                "district_max_cooks_d": dist_max_cook,
                "influential": int(max_cook > 4 / n_B_ext)
            })

    res_df = pd.DataFrame(results)
    res_df.to_csv(REPORTS / "tables" / "regression_results.csv", index=False)
    
    diag_df = pd.DataFrame(diagnostics)
    diag_df.to_csv(REPORTS / "tables" / "regression_diagnostics.csv", index=False)

def make_figures(df: pd.DataFrame, corr_df: pd.DataFrame):
    fig_dir = REPORTS / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Heatmap
    vol_vars = [v for v in corr_df["vol_var"].unique()]
    outcomes = [o for o in corr_df["outcome"].unique()]
    
    heat = pd.DataFrame(index=vol_vars, columns=outcomes, dtype=float)
    for _, row in corr_df.iterrows():
        heat.loc[row["vol_var"], row["outcome"]] = row["rho"]
        
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    cax = ax.matshow(heat, cmap="RdBu_r", vmin=-1, vmax=1)
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(outcomes)))
    ax.set_xticklabels(outcomes, rotation=45, ha="left")
    ax.set_yticks(range(len(vol_vars)))
    ax.set_yticklabels(vol_vars)
    
    for i in range(len(vol_vars)):
        for j in range(len(outcomes)):
            val = heat.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if abs(val)<0.5 else "white")
                
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_corr_heatmap_primary.png")
    plt.close(fig)
    
    # 2. Scatter plots
    for out in ["stunting_pct", "wasting_pct", "anaemia_children_pct"]:
        sub = df[["district", "p_vol_logret_overall", out]].dropna()
        if sub.empty: continue
        
        x = sub["p_vol_logret_overall"]
        y = sub[out]
        rho, _ = stats.spearmanr(x, y)
        n = len(sub)
        
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        ax.scatter(x, y, alpha=0.7)
        
        # OLS line
        m, b = np.polyfit(x, y, 1)
        ax.plot(x, m*x + b, color="red", linestyle="--")
        
        for i, row in sub.iterrows():
            ax.text(row["p_vol_logret_overall"], row[out], row["district"], fontsize=6)
            
        ax.set_xlabel("p_vol_logret_overall")
        ax.set_ylabel(out)
        ax.set_title(f"{out} vs p_vol_logret_overall\nSpearman rho: {rho:.2f} (n={n})")
        fig.tight_layout()
        fig.savefig(fig_dir / f"fig_scatter_vol_vs_{out.replace('_pct', '')}.png")
        plt.close(fig)
        
    # 3. District Ranking
    sub = df[["district", "p_vol_logret_overall", "stunting_pct"]].dropna()
    if not sub.empty:
        sub = sub.sort_values("p_vol_logret_overall", ascending=True) # so highest is at top of barh
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8), dpi=150, sharey=True)
        
        y_pos = np.arange(len(sub))
        ax1.barh(y_pos, sub["p_vol_logret_overall"], align="center")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(sub["district"])
        ax1.set_xlabel("Volatility (p_vol_logret_overall)")
        
        ax2.barh(y_pos, sub["stunting_pct"], align="center", color="orange")
        ax2.set_xlabel("Stunting (%)")
        
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_district_ranking.png")
        plt.close(fig)
        
    # 4. Window robustness
    sub = df[["district", "p_vol_logret_overall", "e_vol_logret_overall"]].dropna()
    if not sub.empty:
        x = sub["p_vol_logret_overall"]
        y = sub["e_vol_logret_overall"]
        rho, _ = stats.spearmanr(x, y)
        n = len(sub)
        
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        ax.scatter(x, y, alpha=0.7)
        ax.set_xlabel("Primary Window Volatility")
        ax.set_ylabel("Extended Window Volatility")
        ax.set_title(f"Window Robustness\nSpearman rho: {rho:.2f} (n={n})")
        fig.tight_layout()
        fig.savefig(fig_dir / "fig_window_robustness.png")
        plt.close(fig)

def main():
    print("=" * 60)
    print("ANALYSIS & VISUALIZATION")
    print("=" * 60)
    
    df = pd.read_csv(PROCESSED / "clean_data.csv")
    
    create_descriptives(df)
    
    corr_p = compute_correlations(df, "p_")
    compute_correlations(df, "e_")
    
    run_regressions(df)
    
    make_figures(df, corr_p)
    
    print("Analysis complete.")

if __name__ == "__main__":
    main()
