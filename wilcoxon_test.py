###################################################################
#  TRACE — Transparent Regression with Adaptive Confidence
#  Estimation
#  wilcoxon_test.py — Corrected Resampled t-test
#                     (Nadeau and Bengio, 2003)
#
#  IMPORTANT: This script must be run in the same session as
#  cv_v2.py and benchmark_models.py. It uses variables produced
#  by those scripts:
#    - df_full         (from cv_v2.py)
#    - fold_metrics    (from cv_v2.py)
#    - all_fold_metrics (from benchmark_models.py)
#
#  In Jupyter: run cv_v2.py and benchmark_models.py cells first,
#  then run this script.
#
#  Reference:
#    Nadeau C, Bengio Y (2003). Inference for the generalization
#    error. Machine Learning 52, 239-281.
###################################################################

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

# ── Dataset info — needed for correction factor ───────────────────
N_TOTAL  = len(df_full)         # total dataset size
N_SPLITS = 5                    # number of folds
N_TEST   = N_TOTAL // N_SPLITS  # approx test size per fold
N_TRAIN  = N_TOTAL - N_TEST     # approx train size per fold

# ── Extract per-fold metrics ──────────────────────────────────────
your_r2   = [f["R2"]   for f in fold_metrics]
your_rmse = [f["RMSE"] for f in fold_metrics]

baselines_r2   = {name: [f["R2"]   for f in folds]
                  for name, folds in all_fold_metrics.items()}
baselines_rmse = {name: [f["RMSE"] for f in folds]
                  for name, folds in all_fold_metrics.items()}

# ── Corrected resampled t-test ────────────────────────────────────
def corrected_t_test(a, b, n_train, n_test):
    """
    Nadeau and Bengio (2003) corrected resampled t-test.
    Accounts for dependence between cross-validation folds.

    Parameters
    ----------
    a, b    : per-fold metric values for two methods
    n_train : training set size per fold
    n_test  : test set size per fold
    """
    n      = len(a)
    diff   = np.array(a) - np.array(b)
    mean_d = diff.mean()

    if np.all(diff == 0):
        return np.nan, 1.0

    var_d         = diff.var(ddof=1)
    corrected_var = var_d * (1 / n + n_test / n_train)

    if corrected_var <= 0:
        return np.nan, 1.0

    t_stat = mean_d / np.sqrt(corrected_var)
    p_val  = 2 * t_dist.sf(abs(t_stat), df=n - 1)
    return t_stat, p_val

# ── Compare TRACE against all baselines ───────────────────────────
def compare_all(your_vals, baseline_dict, metric,
                higher_better, n_train, n_test):
    your_arr = np.array(your_vals)
    rows     = []

    for name, base_vals in baseline_dict.items():
        base_arr      = np.array(base_vals)
        t_stat, p_val = corrected_t_test(
            your_arr, base_arr, n_train, n_test)

        your_mean = your_arr.mean()
        base_mean = base_arr.mean()

        if higher_better:
            direction = "better" if your_mean > base_mean else "worse"
        else:
            direction = "better" if your_mean < base_mean else "worse"

        if np.isnan(p_val):
            sig = "—"
        elif p_val < 0.001:
            sig = "***"
        elif p_val < 0.01:
            sig = "**"
        elif p_val < 0.05:
            sig = "*"
        else:
            sig = "ns"

        rows.append({
            "Baseline"         : name,
            f"TRACE ({metric})" : round(your_mean, 4),
            f"Base ({metric})"  : round(base_mean, 4),
            "Direction"        : direction,
            "t-stat"           : round(t_stat, 3)
                                 if not np.isnan(t_stat) else "nan",
            "p-value"          : round(p_val, 4)
                                 if not np.isnan(p_val) else "nan",
            "Significance"     : sig,
        })

    return pd.DataFrame(rows)

# ── R² comparison ─────────────────────────────────────────────────
print("=" * 80)
print("  CORRECTED RESAMPLED t-TEST — R² (higher is better)")
print("  Nadeau and Bengio (2003)")
print("  * p<0.05   ** p<0.01   *** p<0.001   ns = not significant")
print("=" * 80)
df_r2 = compare_all(
    your_r2, baselines_r2, "R2",
    higher_better=True,
    n_train=N_TRAIN, n_test=N_TEST)
print(df_r2.to_string(index=False))

# ── RMSE comparison ───────────────────────────────────────────────
print("\n" + "=" * 80)
print("  CORRECTED RESAMPLED t-TEST — RMSE (lower is better)")
print("  Nadeau and Bengio (2003)")
print("  * p<0.05   ** p<0.01   *** p<0.001   ns = not significant")
print("=" * 80)
df_rmse = compare_all(
    your_rmse, baselines_rmse, "RMSE",
    higher_better=False,
    n_train=N_TRAIN, n_test=N_TEST)
print(df_rmse.to_string(index=False))

print(f"\n  Dataset  : n={N_TOTAL}, {N_SPLITS}-fold CV")
print(f"  n_train  : approx {N_TRAIN} per fold")
print(f"  n_test   : approx {N_TEST} per fold")
print(f"  Reference: Nadeau C, Bengio Y (2003). Inference for the")
print(f"             generalization error. Machine Learning 52, 239-281.")
