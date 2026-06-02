###################################################################
#  TRACE — Transparent Regression with Adaptive Confidence
#  Estimation
#  cv_v2.py — 5-Fold Cross Validation
#
#  USAGE:
#    1. Place your dataset file in the data/ folder
#    2. Set FILE_CV, SHEET_CV and TARGET_COL below
#    3. Run: python cv_v2.py
#
#  Dataset must have:
#    - All numerical columns
#    - One target column (set TARGET_COL to its name)
#    - No missing values
###################################################################

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics         import r2_score, explained_variance_score
from sklearn.neighbors       import BallTree
from scipy.optimize          import minimize

# ══════════════════════════════════════════════════════════════════
#  SETTINGS — only change these three lines
# ══════════════════════════════════════════════════════════════════
FILE_CV    = "data/your_dataset.xlsx"   # path to your dataset file
SHEET_CV   = "Sheet1"                   # Excel sheet name (ignored for CSV)
TARGET_COL = "y"                        # name of the target column
# ══════════════════════════════════════════════════════════════════

# ── Fixed hyperparameters (do not change) ─────────────────────────
K_CV         = 5      # prediction neighbors
VAL_FRACTION = 0.20   # validation fraction for alpha optimization
K_OPT        = 50     # neighbors used during optimization
EPS          = 1e-6   # numerical stability constant
N_SPLITS     = 5      # cross-validation folds

# ── Load dataset ──────────────────────────────────────────────────
if FILE_CV.endswith(".csv"):
    df_raw = pd.read_csv(FILE_CV)
else:
    df_raw = pd.read_excel(FILE_CV, sheet_name=SHEET_CV)

# Rename target column to "y" for internal use
if TARGET_COL != "y":
    df_raw = df_raw.rename(columns={TARGET_COL: "y"})

df_full      = df_raw.copy()
feature_cols = [c for c in df_full.columns if c != "y"]

print(f"Features detected: {len(feature_cols)}")
print(f"Full dataset loaded: {df_full.shape}")

# ── Helper functions ──────────────────────────────────────────────

def _compute_global_stats(df_in):
    required = feature_cols + ["y"]
    df       = df_in[required].reset_index(drop=True)
    return df.mean(), df.std()

def _compute_diffs(df_vals, alphas, global_mean, fcols, eps=1e-6):
    out = np.zeros((len(df_vals), len(fcols)))
    for j, col in enumerate(fcols):
        a    = alphas[j]
        x    = df_vals[col].values
        mean = global_mean[col]
        if x.min() >= 0 and mean >= 0:
            xp        = x.clip(eps) ** a
            mp        = max(mean, eps) ** a
            out[:, j] = (xp - mp) / (mp + eps) * 100
        else:
            xp        = np.sign(x)    * np.abs(x).clip(eps) ** a
            mp        = np.sign(mean) * abs(mean + eps)      ** a
            out[:, j] = (xp - mp) / (abs(mp) + eps) * 100
    return out

def _build_match_space(diffs, raw_vals, raw_mean, raw_std, eps=1e-6):
    raw_norm = (raw_vals - raw_mean) / (raw_std + eps)
    return np.hstack([diffs, raw_norm])

def _compute_y_diffs(y_vals, mean_y, std_y, eps=1e-6):
    return (y_vals - mean_y) / (std_y + eps)

def _reconstruct_y(y_diffs_matched, mean_y, std_y):
    return mean_y + y_diffs_matched * std_y

def _compute_confidence(distances, y_diffs_matched):
    mean_dist   = distances.mean(axis=1)
    median_dist = np.median(mean_dist) + EPS
    conf_prox   = np.exp(-mean_dist / median_dist)
    y_std       = y_diffs_matched.std(axis=1)
    y_abs_mean  = np.abs(y_diffs_matched).mean(axis=1) + EPS
    conf_cons   = np.exp(-y_std / y_abs_mean)
    return (conf_prox + conf_cons) / 2

def _objective_cv(alphas, train_df, val_df, fcols,
                  global_mean, global_std, raw_mean, raw_std, k, eps):
    try:
        td_diffs = _compute_diffs(train_df, alphas, global_mean, fcols, eps)
        vd_diffs = _compute_diffs(val_df,   alphas, global_mean, fcols, eps)
        td = _build_match_space(
            td_diffs, train_df[fcols].values, raw_mean, raw_std, eps)
        vd = _build_match_space(
            vd_diffs, val_df[fcols].values,   raw_mean, raw_std, eps)
        tree   = BallTree(td, metric="manhattan")
        _, idx = tree.query(vd, k=k)
        mean_y = global_mean["y"]
        std_y  = max(global_std["y"], eps)
        y_tr   = train_df["y"].values
        yd     = _compute_y_diffs(y_tr, mean_y, std_y, eps)
        pred   = _reconstruct_y(yd[idx].mean(axis=1), mean_y, std_y)
        return np.sqrt(np.mean((pred - val_df["y"].values) ** 2))
    except Exception:
        return 1e10

def _compute_metrics(y_true, y_pred):
    errors = y_pred - y_true
    rmse   = np.sqrt(np.mean(errors ** 2))
    return {
        "MAE"          : np.mean(np.abs(errors)),
        "Median_AbsErr": np.median(np.abs(errors)),
        "RMSE"         : rmse,
        "MSE"          : np.mean(errors ** 2),
        "Max_Error"    : np.max(np.abs(errors)),
        "MBE"          : np.mean(errors),
        "MAPE"         : np.mean(
            np.abs(errors / np.where(y_true == 0, np.nan, y_true))) * 100,
        "SMAPE"        : np.mean(
            2 * np.abs(errors) / (np.abs(y_true) + np.abs(y_pred))) * 100,
        "NRMSE_range"  : rmse / (y_true.max() - y_true.min()) * 100,
        "NRMSE_mean"   : rmse / np.mean(y_true) * 100,
        "R2"           : r2_score(y_true, y_pred),
        "Explained_Var": explained_variance_score(y_true, y_pred),
    }

def _conf_group_metrics(actual_y, pred_y, mask):
    n = mask.sum()
    if n < 10:
        return None
    y_t    = actual_y[mask]
    y_p    = pred_y[mask]
    errors = y_p - y_t
    return {
        "n"    : int(n),
        "pct"  : n / len(actual_y) * 100,
        "r2"   : r2_score(y_t, y_p),
        "rmse" : np.sqrt(np.mean(errors ** 2)),
        "mae"  : np.mean(np.abs(errors)),
        "smape": np.mean(
            2 * np.abs(errors) / (np.abs(y_t) + np.abs(y_p) + 1e-6)) * 100,
    }

# ── 5-Fold CV loop ────────────────────────────────────────────────
kf              = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
fold_metrics    = []
fold_alphas     = []
fold_pred_times = []
fold_conf_stats = []
fold_conf_perf  = {
    "very_high": [], "high": [], "medium": [], "low": []}

t_train_start = time.perf_counter()

for fold, (train_idx, test_idx) in enumerate(kf.split(df_full)):
    print(f"\n── Fold {fold + 1}/{N_SPLITS} ──")

    train_fold = df_full.iloc[train_idx].reset_index(drop=True)
    test_fold  = df_full.iloc[test_idx].reset_index(drop=True)

    global_mean, global_std = _compute_global_stats(train_fold)
    mean_y   = global_mean["y"]
    std_y    = max(global_std["y"], EPS)
    raw_mean = train_fold[feature_cols].values.mean(axis=0)
    raw_std  = train_fold[feature_cols].values.std(axis=0)

    if fold == 0:
        print("  Feature formula selection:")
        for col in feature_cols:
            tag = ("original"
                   if (train_fold[col].min() >= 0 and global_mean[col] >= 0)
                   else "signed power")
            print(f"    {col:<30}: {tag}")

    n_val  = int(len(train_fold) * VAL_FRACTION)
    tr_sub = train_fold.iloc[:-n_val].copy()
    vl_sub = train_fold.iloc[-n_val:].copy()

    result = minimize(
        _objective_cv,
        x0      = [1.0] * len(feature_cols),
        args    = (tr_sub, vl_sub, feature_cols,
                   global_mean, global_std,
                   raw_mean, raw_std, K_OPT, EPS),
        method  = "Nelder-Mead",
        options = {"maxiter": 1000, "xatol": 0.001, "fatol": 0.01}
    )
    alphas_fold = result.x
    fold_alphas.append(alphas_fold)
    print(f"  Alphas: "
          f"{ {c: round(a, 3) for c, a in zip(feature_cols, alphas_fold)} }")

    train_diffs = _compute_diffs(
        train_fold, alphas_fold, global_mean, feature_cols, EPS)
    train_match = _build_match_space(
        train_diffs, train_fold[feature_cols].values,
        raw_mean, raw_std, EPS)
    y_raw   = train_fold["y"].values
    y_diffs = _compute_y_diffs(y_raw, mean_y, std_y, EPS)
    tree_fold = BallTree(train_match, metric="manhattan")

    X_test     = test_fold[feature_cols].reset_index(drop=True)
    actual_y   = test_fold["y"].values
    test_diffs = _compute_diffs(
        test_fold, alphas_fold, global_mean, feature_cols, EPS)
    test_match = _build_match_space(
        test_diffs, X_test.values, raw_mean, raw_std, EPS)

    t_pred_start    = time.perf_counter()
    dists, idx      = tree_fold.query(test_match, k=K_CV)
    matched_y_diffs = y_diffs[idx]
    pred_y          = _reconstruct_y(
        matched_y_diffs.mean(axis=1), mean_y, std_y)
    confidence      = _compute_confidence(dists, matched_y_diffs)
    fold_pred_times.append(time.perf_counter() - t_pred_start)

    abs_errors = np.abs(actual_y - pred_y)
    conf_corr  = np.corrcoef(confidence, abs_errors)[0, 1]

    conf_stats = {
        "mean"      : confidence.mean(),
        "very_high" : (confidence >= 0.75).mean() * 100,
        "high"      : ((confidence >= 0.50) &
                       (confidence < 0.75)).mean() * 100,
        "medium"    : ((confidence >= 0.25) &
                       (confidence < 0.50)).mean() * 100,
        "low"       : (confidence < 0.25).mean() * 100,
        "corr"      : conf_corr,
    }
    fold_conf_stats.append(conf_stats)

    groups = {
        "very_high": confidence >= 0.75,
        "high"     : (confidence >= 0.50) & (confidence < 0.75),
        "medium"   : (confidence >= 0.25) & (confidence < 0.50),
        "low"      : confidence < 0.25,
    }
    for gname, mask in groups.items():
        fold_conf_perf[gname].append(
            _conf_group_metrics(actual_y, pred_y, mask))

    metrics = _compute_metrics(actual_y, pred_y)
    fold_metrics.append(metrics)
    print(f"  R²={metrics['R2']:.4f} | RMSE={metrics['RMSE']:.4f}")
    print(f"  Confidence: mean={conf_stats['mean']:.3f} | "
          f"very_high={conf_stats['very_high']:.1f}% | "
          f"high={conf_stats['high']:.1f}% | "
          f"medium={conf_stats['medium']:.1f}% | "
          f"low={conf_stats['low']:.1f}%")
    print(f"  Conf-error correlation: {conf_corr:.3f} "
          f"({'good' if conf_corr < -0.2 else 'weak'})")

training_time_v2   = (time.perf_counter() - t_train_start) / N_SPLITS
prediction_time_v2 = np.mean(fold_pred_times)
df_cv_results      = pd.DataFrame(fold_metrics)

# ── Summary ───────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  CV RESULTS — TRACE")
print("=" * 55)
for col in df_cv_results.columns:
    print(f"  {col:15s}: "
          f"{df_cv_results[col].mean():.4f} "
          f"± {df_cv_results[col].std():.4f}")
print("=" * 55)
print(f"  Avg train time (s): {training_time_v2:.4f}")
print(f"  Avg pred time  (s): {prediction_time_v2:.4f}")
print("=" * 55)

# ── Confidence summary ────────────────────────────────────────────
print("\n── Confidence Summary (across all folds) ──")
print(f"  Mean confidence  : "
      f"{np.mean([f['mean'] for f in fold_conf_stats]):.3f}")
print(f"  Very high (>=0.75): "
      f"{np.mean([f['very_high'] for f in fold_conf_stats]):.1f}%")
print(f"  High      (>=0.50): "
      f"{np.mean([f['high'] for f in fold_conf_stats]):.1f}%")
print(f"  Medium    (>=0.25): "
      f"{np.mean([f['medium'] for f in fold_conf_stats]):.1f}%")
print(f"  Low       (<0.25) : "
      f"{np.mean([f['low'] for f in fold_conf_stats]):.1f}%")
corrs = [f['corr'] for f in fold_conf_stats]
print(f"  Conf-error r      : "
      f"{np.mean(corrs):.3f} ± {np.std(corrs):.3f}")

# ── Confidence-stratified performance ────────────────────────────
print("\n── Performance by Confidence Level ──")
print(f"  {'Group':12} {'Coverage%':>10} "
      f"{'R²':>8} {'RMSE':>10} {'MAE':>10}")
print("  " + "─" * 55)
for gname in ["very_high", "high", "medium", "low"]:
    valid = [f for f in fold_conf_perf[gname] if f is not None]
    if not valid:
        print(f"  {gname:12} — insufficient samples")
        continue
    pct   = np.mean([f["pct"]  for f in valid])
    r2m   = np.mean([f["r2"]   for f in valid])
    rmse  = np.mean([f["rmse"] for f in valid])
    mae   = np.mean([f["mae"]  for f in valid])
    print(f"  {gname:12} {pct:>9.1f}% "
          f"{r2m:>8.4f} {rmse:>10.4f} {mae:>10.4f}")
print(f"  {'Overall':12} {'100.0':>10}% "
      f"{df_cv_results['R2'].mean():>8.4f} "
      f"{df_cv_results['RMSE'].mean():>10.4f} "
      f"{df_cv_results['MAE'].mean():>10.4f}")

# ── Learned alphas ────────────────────────────────────────────────
alphas_mean = np.mean(fold_alphas, axis=0)
alphas_std  = np.std(fold_alphas,  axis=0)
print("\n── Learned sensitivity exponents (mean ± std across folds) ──")
for col, m, s in zip(feature_cols, alphas_mean, alphas_std):
    print(f"  {col}: alpha = {m:.4f} ± {s:.4f}")
