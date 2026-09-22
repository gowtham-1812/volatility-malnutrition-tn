"""
src/ml.py
=========
Machine Learning Model stage.
Trains classifiers, runs nested cross-validation, ablation, permutation tests.
"""

import sys
import time
import collections
import numpy as np
import pandas as pd
import joblib
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
    confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    PROCESSED, REPORTS, OUTCOMES, SEED,
    HOLDOUT_TEST_SIZE, ML_TARGET_COLUMN, ML_INNER_SPLITS,
    ML_OUTER_SPLITS, ML_OUTER_REPEATS,
    PERMUTATION_N, PERMUTATION_REPEATS
)

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
ML_REPORTS_DIR = REPORTS / "ml"
FIG_DIR = REPORTS / "figures"

def check_forbidden(features_list):
    forbidden = set(OUTCOMES) | {"households_surveyed"}
    for f in features_list:
        assert f not in forbidden, f"Forbidden feature: {f}"
        assert not f.startswith("e_"), f"Forbidden feature (e_): {f}"
        assert not f.endswith("_n_commodities_used"), f"Forbidden feature: {f}"
        assert not f.endswith("_low_coverage"), f"Forbidden feature: {f}"

def get_models():
    # Pipeline components
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    
    # Models
    logreg = LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, class_weight="balanced")
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED)
    gb = GradientBoostingClassifier(random_state=SEED, subsample=0.8)
    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    
    # Grids
    grid_logreg = {"model__C": [0.01, 0.1, 1.0, 10.0]}
    grid_rf = {"model__max_depth": [2, 3, 4], "model__min_samples_leaf": [2, 3]}
    grid_gb = {"model__n_estimators": [50, 100], "model__max_depth": [1, 2], "model__learning_rate": [0.05, 0.1]}
    
    return {
        "logreg": (Pipeline([("imputer", imputer), ("scaler", scaler), ("model", logreg)]), grid_logreg),
        "rf": (Pipeline([("imputer", imputer), ("model", rf)]), grid_rf),
        "gb": (Pipeline([("imputer", imputer), ("model", gb)]), grid_gb),
        "dummy": (Pipeline([("imputer", imputer), ("model", dummy)]), {})
    }

def safe_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, zero_division=0)

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    ML_REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    FIG_DIR.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("MACHINE LEARNING PIPELINE")
    print("=" * 60)
    
    df = pd.read_csv(PROCESSED / "clean_data.csv")
    
    # 7.1 Target and features
    med_stunt = df[ML_TARGET_COLUMN].median()
    y = (df[ML_TARGET_COLUMN] > med_stunt).astype(int).values
    
    class_counts = pd.Series(y).value_counts().to_dict()
    print(f"Class counts (1 = high stunting): {class_counts}")
    
    fs_controls = [c for c in df.columns if c.startswith("ctrl_")]
    fs_volatility = [
        "p_vol_logret_cereals", "p_vol_logret_pulses", "p_vol_logret_veg", 
        "p_vol_logret_other", "p_vol_logret_overall", "p_cv_overall", 
        "p_range_ratio_overall", "p_spike_share_overall"
    ]
    fs_full = fs_controls + fs_volatility
    
    check_forbidden(fs_controls)
    check_forbidden(fs_volatility)
    check_forbidden(fs_full)
    
    feature_sets = {
        "FS_controls": fs_controls,
        "FS_volatility": fs_volatility,
        "FS_full": fs_full
    }
    
    # 7.3 View 1: Hold-out split
    X_full = df[fs_full]
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=HOLDOUT_TEST_SIZE, stratify=y, random_state=SEED
    )
    print(f"Hold-out sizes: Train={len(y_train)}, Test={len(y_test)}")
    
    models = get_models()
    holdout_results = []
    
    best_inner_f1 = -1.0
    holdout_chosen_model = None
    holdout_chosen_name = None
    holdout_chosen_cm = None
    
    # tie break order: logreg, rf, gb (dummy not considered for winner)
    model_order = {"logreg": 0, "rf": 1, "gb": 2, "dummy": 3}
    
    for m_name in ["logreg", "rf", "gb", "dummy"]:
        pipe, grid = models[m_name]
        
        if m_name == "dummy":
            clf = pipe
            clf.fit(X_train, y_train)
            cv_f1 = np.nan
            best_params = {}
        else:
            cv = StratifiedKFold(ML_INNER_SPLITS, shuffle=True, random_state=SEED)
            clf = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=-1)
            clf.fit(X_train, y_train)
            cv_f1 = clf.best_score_
            best_params = clf.best_params_
            
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = safe_f1(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        holdout_results.append({
            "model": m_name,
            "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "best_params": str(best_params)
        })
        
        if m_name != "dummy":
            # tie break condition
            if cv_f1 > best_inner_f1 or (cv_f1 == best_inner_f1 and model_order[m_name] < model_order.get(holdout_chosen_name, 99)):
                best_inner_f1 = cv_f1
                holdout_chosen_model = clf
                holdout_chosen_name = m_name
                holdout_chosen_cm = cm
                
    pd.DataFrame(holdout_results).to_csv(ML_REPORTS_DIR / "holdout_metrics.csv", index=False)
    
    # Save holdout CM fig
    disp = ConfusionMatrixDisplay(confusion_matrix=holdout_chosen_cm)
    disp.plot(cmap="Blues")
    plt.title(f"Hold-out Confusion Matrix ({holdout_chosen_name})")
    plt.savefig(FIG_DIR / "fig_ml_confusion_holdout.png")
    plt.close()
    
    # 7.4 View 2: Repeated Nested CV
    print("Running Repeated Nested CV...")
    start_time = time.time()
    
    outer_cv = RepeatedStratifiedKFold(n_splits=ML_OUTER_SPLITS, n_repeats=ML_OUTER_REPEATS, random_state=SEED)
    
    repeated_metrics = []
    selected_params = []
    
    for m_name in ["logreg", "rf", "gb", "dummy"]:
        fs_list = ["FS_full"] if m_name == "dummy" else ["FS_controls", "FS_volatility", "FS_full"]
        for fs_name in fs_list:
            X = df[feature_sets[fs_name]]
            
            # Pools for each repeat
            # Repeat index: fold_idx // ML_OUTER_SPLITS
            pooled_y_true = collections.defaultdict(list)
            pooled_y_pred = collections.defaultdict(list)
            pooled_y_prob = collections.defaultdict(list)
            
            for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
                rep_idx = fold_idx // ML_OUTER_SPLITS
                
                X_tr, y_tr = X.iloc[train_idx], y[train_idx]
                X_te, y_te = X.iloc[test_idx], y[test_idx]
                
                pipe, grid = models[m_name]
                
                if m_name == "dummy":
                    clf = pipe
                    clf.fit(X_tr, y_tr)
                    bp = {}
                else:
                    cv_inner = StratifiedKFold(ML_INNER_SPLITS, shuffle=True, random_state=SEED)
                    clf = GridSearchCV(pipe, grid, scoring="f1", cv=cv_inner, n_jobs=-1)
                    clf.fit(X_tr, y_tr)
                    bp = clf.best_params_
                    
                selected_params.append({
                    "model": m_name, "feature_set": fs_name,
                    "repeat": rep_idx, "fold": fold_idx % ML_OUTER_SPLITS,
                    "params": str(bp)
                })
                
                y_p = clf.predict(X_te)
                if m_name != "dummy" and hasattr(clf, "predict_proba"):
                    y_prob = clf.predict_proba(X_te)[:, 1]
                else:
                    y_prob = [np.nan] * len(y_te)
                    
                pooled_y_true[rep_idx].extend(y_te)
                pooled_y_pred[rep_idx].extend(y_p)
                pooled_y_prob[rep_idx].extend(y_prob)
                
            for rep_idx in range(ML_OUTER_REPEATS):
                yt = np.array(pooled_y_true[rep_idx])
                yp = np.array(pooled_y_pred[rep_idx])
                yprob = np.array(pooled_y_prob[rep_idx])
                
                acc = accuracy_score(yt, yp)
                prec = precision_score(yt, yp, zero_division=0)
                rec = recall_score(yt, yp, zero_division=0)
                f1 = safe_f1(yt, yp)
                
                if m_name != "dummy" and not np.isnan(yprob).all():
                    auc = roc_auc_score(yt, yprob)
                else:
                    auc = np.nan
                    
                repeated_metrics.append({
                    "model": m_name, "feature_set": fs_name, "repeat": rep_idx,
                    "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc
                })

    end_time = time.time()
    print(f"Nested CV elapsed time: {end_time - start_time:.1f} seconds")
    
    rep_df = pd.DataFrame(repeated_metrics)
    rep_df.to_csv(ML_REPORTS_DIR / "repeated_cv_metrics.csv", index=False)
    
    pd.DataFrame(selected_params).to_csv(ML_REPORTS_DIR / "cv_selected_params.csv", index=False)
    
    summ_df = rep_df.groupby(["model", "feature_set"]).agg({
        "accuracy": ["mean", "std"], "precision": ["mean", "std"],
        "recall": ["mean", "std"], "f1": ["mean", "std"], "roc_auc": ["mean", "std"]
    }).reset_index()
    summ_df.columns = ["model", "feature_set", "acc_mean", "acc_std", "prec_mean", "prec_std",
                       "rec_mean", "rec_std", "f1_mean", "f1_std", "auc_mean", "auc_std"]
    summ_df.to_csv(ML_REPORTS_DIR / "repeated_cv_summary.csv", index=False)
    
    # 7.5 Final Model
    f1_full = summ_df[summ_df["feature_set"] == "FS_full"].copy()
    f1_full["order"] = f1_full["model"].map(model_order)
    f1_full = f1_full.sort_values(["f1_mean", "order"], ascending=[False, True])
    final_model_name = f1_full.iloc[0]["model"]
    
    # Get mode of params for final model on FS_full
    p_df = pd.DataFrame(selected_params)
    final_params_df = p_df[(p_df["model"] == final_model_name) & (p_df["feature_set"] == "FS_full")]
    mode_params_str = final_params_df["params"].mode().iloc[0]
    import ast
    mode_params = ast.literal_eval(mode_params_str)
    
    pipe, _ = models[final_model_name]
    pipe.set_params(**mode_params)
    X_full_final = df[fs_full]
    pipe.fit(X_full_final, y)
    
    joblib.dump(pipe, MODELS_DIR / "final_model.joblib")
    print(f"Final model chosen: {final_model_name} with params {mode_params}")
    
    # 7.6 Ablation
    abl_rep = rep_df[rep_df["model"] == final_model_name].copy()
    fs_full_f1 = abl_rep[abl_rep["feature_set"] == "FS_full"].set_index("repeat")["f1"]
    fs_ctrl_f1 = abl_rep[abl_rep["feature_set"] == "FS_controls"].set_index("repeat")["f1"]
    
    diff_f1 = fs_full_f1 - fs_ctrl_f1
    mean_diff = diff_f1.mean()
    beat_count = (diff_f1 > 0).sum()
    
    abl_summ = summ_df[summ_df["model"] == final_model_name].copy()
    abl_summ.to_csv(ML_REPORTS_DIR / "ablation.csv", index=False)
    print(f"Ablation FS_full vs FS_controls F1 diff: mean={mean_diff:.4f}, beats={beat_count}/10")
    
    # 7.7 Permutation Test
    print("Running Permutation Test...")
    S_obs = summ_df[(summ_df["model"] == final_model_name) & (summ_df["feature_set"] == "FS_full")]["acc_mean"].values[0]
    
    null_accs = []
    # Note: Using RepeatedStratifiedKFold(ML_OUTER_SPLITS, PERMUTATION_REPEATS)
    perm_cv = RepeatedStratifiedKFold(n_splits=ML_OUTER_SPLITS, n_repeats=PERMUTATION_REPEATS, random_state=SEED)
    
    for i in range(PERMUTATION_N):
        rng = np.random.default_rng(SEED + 1000 + i)
        y_shuffled = y.copy()
        rng.shuffle(y_shuffled)
        
        pooled_y_true = collections.defaultdict(list)
        pooled_y_pred = collections.defaultdict(list)
        
        # Fit with fixed params (the final pipe)
        for fold_idx, (train_idx, test_idx) in enumerate(perm_cv.split(X_full_final, y_shuffled)):
            rep_idx = fold_idx // ML_OUTER_SPLITS
            X_tr, y_tr = X_full_final.iloc[train_idx], y_shuffled[train_idx]
            X_te, y_te = X_full_final.iloc[test_idx], y_shuffled[test_idx]
            
            p_clone, _ = models[final_model_name]
            p_clone.set_params(**mode_params)
            p_clone.fit(X_tr, y_tr)
            
            yp = p_clone.predict(X_te)
            pooled_y_true[rep_idx].extend(y_te)
            pooled_y_pred[rep_idx].extend(yp)
            
        rep_accs = []
        for rep_idx in range(PERMUTATION_REPEATS):
            yt = np.array(pooled_y_true[rep_idx])
            yp = np.array(pooled_y_pred[rep_idx])
            rep_accs.append(accuracy_score(yt, yp))
            
        null_accs.append(np.mean(rep_accs))
        
    null_accs = np.array(null_accs)
    p_perm = (1 + np.sum(null_accs >= S_obs)) / (1 + PERMUTATION_N)
    
    perm_res = pd.DataFrame([{
        "observed_acc": S_obs,
        "null_mean": np.mean(null_accs),
        "null_p95": np.percentile(null_accs, 95),
        "p_value": p_perm
    }])
    perm_res.to_csv(ML_REPORTS_DIR / "permutation_test.csv", index=False)
    pd.DataFrame({"null_acc": null_accs}).to_csv(ML_REPORTS_DIR / "permutation_null.csv", index=False)
    
    # 7.8 Feature Importance
    result = permutation_importance(pipe, X_full_final, y, scoring="accuracy", n_repeats=50, random_state=SEED)
    fi_df = pd.DataFrame({
        "feature": X_full_final.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values("importance_mean", ascending=False)
    fi_df.to_csv(ML_REPORTS_DIR / "feature_importance.csv", index=False)
    
    if final_model_name == "logreg":
        coefs = pipe.named_steps["model"].coef_[0]
        coef_df = pd.DataFrame({
            "feature": X_full_final.columns,
            "coef": coefs
        }).sort_values("coef", ascending=False)
        coef_df.to_csv(ML_REPORTS_DIR / "logreg_coefficients.csv", index=False)

    # 7.9 Figures
    
    # fig_ml_metrics_by_featureset.png
    plt.figure(figsize=(10, 6), dpi=150)
    models_plot = ["logreg", "rf", "gb"]
    sets_plot = ["FS_controls", "FS_volatility", "FS_full"]
    x = np.arange(len(models_plot))
    width = 0.25
    
    for i, fs in enumerate(sets_plot):
        f1s = []
        errs = []
        for m in models_plot:
            row = summ_df[(summ_df["model"]==m) & (summ_df["feature_set"]==fs)]
            if not row.empty:
                f1s.append(row["f1_mean"].values[0])
                errs.append(row["f1_std"].values[0])
            else:
                f1s.append(0)
                errs.append(0)
        plt.bar(x + (i - 1) * width, f1s, width, yerr=errs, label=fs, capsize=5)
        
    dummy_f1 = summ_df[(summ_df["model"]=="dummy") & (summ_df["feature_set"]=="FS_full")]["f1_mean"].values[0]
    plt.axhline(dummy_f1, color="red", linestyle="--", label="Dummy Baseline")
    plt.xticks(x, models_plot)
    plt.ylabel("F1 Score")
    plt.title("Mean F1 Score by Feature Set and Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_ml_metrics_by_featureset.png")
    plt.close()
    
    # fig_ml_permutation_null.png
    plt.figure(figsize=(6, 4), dpi=150)
    plt.hist(null_accs, bins=15, alpha=0.7, color="gray", label="Null Distribution")
    plt.axvline(S_obs, color="red", label=f"Observed ({S_obs:.3f})")
    plt.title(f"Permutation Test (p={p_perm:.4f})")
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig_ml_permutation_null.png")
    plt.close()
    
    print("Done.")

if __name__ == "__main__":
    main()
