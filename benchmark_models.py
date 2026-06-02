###################################################################
#  TRACE — Transparent Regression with Adaptive Confidence
#  Estimation
#  benchmark_models.py — Baseline Models 5-Fold Cross Validation
#
#  USAGE:
#    1. Place your dataset file in the data/ folder
#    2. Set FILE, SHEET and TARGET below
#    3. Run: python benchmark_models.py
#
#  Models evaluated:
#    LinearRegression, DecisionTree, KNN, RandomForest,
#    XGBoost, CatBoost, MLP, ANN (PyTorch), EBM, TabNet
###################################################################

import time
import numpy as np
import pandas as pd
from sklearn.model_selection  import KFold
from sklearn.metrics          import r2_score, explained_variance_score
from sklearn.linear_model     import LinearRegression
from sklearn.tree             import DecisionTreeRegressor
from sklearn.neighbors        import KNeighborsRegressor
from sklearn.ensemble         import RandomForestRegressor
from sklearn.neural_network   import MLPRegressor
from sklearn.preprocessing    import StandardScaler
from sklearn.base             import BaseEstimator, RegressorMixin
from xgboost                  import XGBRegressor
from catboost                 import CatBoostRegressor
from interpret.glassbox       import ExplainableBoostingRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
import torch
import torch.nn as nn

# ══════════════════════════════════════════════════════════════════
#  SETTINGS — only change these three lines
# ══════════════════════════════════════════════════════════════════
FILE     = "data/your_dataset.xlsx"   # path to your dataset file
SHEET    = "Sheet1"                   # Excel sheet name (ignored for CSV)
TARGET   = "y"                        # name of the target column
# ══════════════════════════════════════════════════════════════════

# ── Fixed hyperparameters ─────────────────────────────────────────
N_SPLITS = 5

# ── Load dataset ──────────────────────────────────────────────────
if FILE.endswith(".csv"):
    df_full = pd.read_csv(FILE)
else:
    df_full = pd.read_excel(FILE, sheet_name=SHEET)

FEATURE_COLS = [c for c in df_full.columns if c != TARGET]
print(f"Features detected: {len(FEATURE_COLS)}")
print(f"Full dataset loaded: {df_full.shape}")

X_all = df_full[FEATURE_COLS].values
y_all = df_full[TARGET].values

# ── ANN (PyTorch) — sklearn-compatible wrapper ────────────────────
class ANNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden=(128, 64, 32),
                 epochs=200, lr=1e-3, random_state=42):
        self.hidden       = hidden
        self.epochs       = epochs
        self.lr           = lr
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        self.scaler_ = StandardScaler()
        X_s = torch.tensor(
            self.scaler_.fit_transform(X), dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        layers, in_dim = [], X_s.shape[1]
        for h in self.hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.model_ = nn.Sequential(*layers)
        opt  = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
        loss = nn.MSELoss()
        self.model_.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss(self.model_(X_s), y_t).backward()
            opt.step()
        return self

    def predict(self, X):
        self.model_.eval()
        X_s = torch.tensor(
            self.scaler_.transform(X), dtype=torch.float32)
        with torch.no_grad():
            return self.model_(X_s).squeeze(1).numpy()

# ── Metrics ───────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
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
            np.abs(errors /
                   np.where(y_true == 0, np.nan, y_true))) * 100,
        "SMAPE"        : np.mean(
            2 * np.abs(errors) /
            (np.abs(y_true) + np.abs(y_pred))) * 100,
        "NRMSE_range"  : rmse / (y_true.max() - y_true.min()) * 100,
        "NRMSE_mean"   : rmse / np.mean(y_true) * 100,
        "R2"           : r2_score(y_true, y_pred),
        "Explained_Var": explained_variance_score(y_true, y_pred),
    }

# ── Model definitions ─────────────────────────────────────────────
def get_models():
    return {
        "LinearRegression": LinearRegression(),
        "DecisionTree"    : DecisionTreeRegressor(random_state=42),
        "KNN_raw"         : KNeighborsRegressor(n_neighbors=7),
        "RandomForest"    : RandomForestRegressor(
            n_estimators=100, random_state=42),
        "XGBoost"         : XGBRegressor(
            n_estimators=100, random_state=42, verbosity=0),
        "CatBoost"        : CatBoostRegressor(
            n_estimators=100, random_state=42, verbose=0),
        "MLP"             : MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500, random_state=42),
        "ANN"             : ANNRegressor(
            hidden=(128, 64, 32), epochs=200, random_state=42),
        "EBM"             : ExplainableBoostingRegressor(
            random_state=42),
        "TabNet"          : TabNetRegressor(verbose=0, seed=42),
    }

# ── 5-Fold CV loop ────────────────────────────────────────────────
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

all_fold_metrics         = {}
all_fold_timing          = {}
trained_models_last_fold = {}

for fold, (train_idx, test_idx) in enumerate(kf.split(X_all)):
    print(f"\n{'=' * 50}")
    print(f"  Fold {fold + 1}/{N_SPLITS}")
    print(f"{'=' * 50}")

    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    X_test  = X_all[test_idx]
    y_test  = y_all[test_idx]

    models = get_models()

    for name, model in models.items():
        print(f"  Training {name}...")
        try:
            t_start = time.perf_counter()
            if name == "TabNet":
                model.fit(
                    X_train.astype(np.float32),
                    y_train.reshape(-1, 1).astype(np.float32))
            else:
                model.fit(X_train, y_train)
            train_time = time.perf_counter() - t_start

            t_start = time.perf_counter()
            if name == "TabNet":
                preds = model.predict(
                    X_test.astype(np.float32)).flatten()
            else:
                preds = model.predict(X_test)
            pred_time = time.perf_counter() - t_start

            metrics = compute_metrics(y_test, preds)

            if name not in all_fold_metrics:
                all_fold_metrics[name] = []
                all_fold_timing[name]  = {"train": [], "pred": []}

            all_fold_metrics[name].append(metrics)
            all_fold_timing[name]["train"].append(train_time)
            all_fold_timing[name]["pred"].append(pred_time)

            print(f"    R²={metrics['R2']:.4f} | "
                  f"RMSE={metrics['RMSE']:.4f} | "
                  f"train={train_time:.3f}s")

        except Exception as e:
            print(f"    FAILED: {e}")

    trained_models_last_fold = models

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("  BENCHMARK CV RESULTS — mean ± std across 5 folds")
print(f"{'=' * 70}")

benchmark_cv_results = {}

for name in all_fold_metrics:
    df_folds = pd.DataFrame(all_fold_metrics[name])
    summary  = {}
    for col in df_folds.columns:
        summary[f"{col}_mean"] = df_folds[col].mean()
        summary[f"{col}_std"]  = df_folds[col].std()
    summary["train_time_mean"] = np.mean(
        all_fold_timing[name]["train"])
    summary["pred_time_mean"]  = np.mean(
        all_fold_timing[name]["pred"])
    benchmark_cv_results[name] = summary

    print(f"\n  {name}")
    print(f"    R²    : "
          f"{df_folds['R2'].mean():.4f} "
          f"± {df_folds['R2'].std():.4f}")
    print(f"    RMSE  : "
          f"{df_folds['RMSE'].mean():.4f} "
          f"± {df_folds['RMSE'].std():.4f}")
    print(f"    SMAPE : "
          f"{df_folds['SMAPE'].mean():.2f}% "
          f"± {df_folds['SMAPE'].std():.2f}%")
    print(f"    Train : "
          f"{np.mean(all_fold_timing[name]['train']):.3f}s | "
          f"Pred: {np.mean(all_fold_timing[name]['pred']):.3f}s")

df_benchmark_cv = pd.DataFrame(benchmark_cv_results).T
print(f"\ndf_benchmark_cv shape: {df_benchmark_cv.shape}")
print(f"trained_models_last_fold keys: "
      f"{list(trained_models_last_fold.keys())}")
