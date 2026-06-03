###################################################################
# Figure 1 — TRACE prediction-level audit trail  (Nature MI vertical)
#
# Part of the TRACE project:
# https://github.com/mutazshafeey/trace-regression
#
# Panels in algorithm order:
#   a: Learned feature sensitivity   (alphas learned first)
#   b: Deviation profiles            (alpha transformation applied)
#   c: Matched training instances    (nearest neighbors retrieved)
#   d: Prediction reliability        (predict + confidence score)
#
# DEPENDENCY: Must be run after cv_v2.py in the same session.
# Requires the following variables from cv_v2.py:
#   kf, df_full, fold_alphas, feature_cols, EPS
#   _compute_global_stats, _compute_diffs, _build_match_space,
#   _compute_y_diffs, _reconstruct_y, _compute_confidence
#
# Output: audit_trace_Protein.pdf and audit_trace_Protein.png
###################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.colors import TwoSlopeNorm
from matplotlib.text import Text
from sklearn.neighbors import BallTree
import warnings
warnings.filterwarnings("ignore")

# ── Settings ──────────────────────────────────────────────────────
DATASET_NAME = "Protein"
TARGET_UNIT  = "\u00c5"
FOLD_IDX     = 0
K_SHOW       = 5
SAVE_PATH    = f"audit_trace_{DATASET_NAME}.pdf"

# ── Colors ────────────────────────────────────────────────────────
C_BLACK  = "#1A1A1A"
C_DARK   = "#2D3748"
C_GRAY   = "#64748B"
C_LGRAY  = "#CBD5E1"
C_VLIGHT = "#F8FAFC"
C_WHITE  = "#FFFFFF"
C_GOLD   = "#D97706"
C_BLUE   = "#2563EB"
C_GREEN  = "#059669"
C_RED    = "#DC2626"
C_AMBER  = "#92400E"
C_LBLUE  = "#DBEAFE"
C_PURPLE = "#7C3AED"

# ── Recompute fold data ───────────────────────────────────────────
splits = list(kf.split(df_full))
train_idx, test_idx = splits[FOLD_IDX]

train_fold = df_full.iloc[train_idx].reset_index(drop=True)
test_fold  = df_full.iloc[test_idx].reset_index(drop=True)

global_mean, global_std = _compute_global_stats(train_fold)
mean_y   = global_mean["y"]
std_y    = max(global_std["y"], EPS)
raw_mean = train_fold[feature_cols].values.mean(axis=0)
raw_std  = train_fold[feature_cols].values.std(axis=0)
# ── Use mean alphas across all folds ─────────────────────────────
# This ensures the sensitivity exponents shown in panel a are
# consistent with the cross-fold mean values in Supplementary
# Table S11, rather than reflecting a single fold.
alphas_f = np.mean(fold_alphas, axis=0)

train_diffs = _compute_diffs(
    train_fold, alphas_f, global_mean, feature_cols, EPS)
train_match = _build_match_space(
    train_diffs, train_fold[feature_cols].values, raw_mean, raw_std, EPS)
test_diffs  = _compute_diffs(
    test_fold, alphas_f, global_mean, feature_cols, EPS)
test_match  = _build_match_space(
    test_diffs, test_fold[feature_cols].values, raw_mean, raw_std, EPS)

tree = BallTree(train_match, metric="manhattan")
y_diffs_tr = _compute_y_diffs(
    train_fold["y"].values, mean_y, std_y, EPS)

# ── Query all test instances ──────────────────────────────────────
dists_all, idx_all = tree.query(test_match, k=K_SHOW)
matched_all    = y_diffs_tr[idx_all]
preds_all      = _reconstruct_y(matched_all.mean(axis=1), mean_y, std_y)
confs_all      = _compute_confidence(dists_all, matched_all)
errors_all     = np.abs(preds_all - test_fold["y"].values)
pct_errors_all = errors_all / (np.abs(test_fold["y"].values) + EPS) * 100

# Confidence decomposition matching _compute_confidence over full fold
mean_d_all            = dists_all.mean(axis=1)
fold_median_mean_dist = np.median(mean_d_all) + EPS
conf_prox_all         = np.exp(-mean_d_all / fold_median_mean_dist)
y_std_all             = matched_all.std(axis=1)
y_abs_m_all           = np.abs(matched_all).mean(axis=1) + EPS
conf_cons_all         = np.exp(-y_std_all / y_abs_m_all)

# Safety check: full-fold decomposition should match _compute_confidence
conf_recomputed_all = (conf_prox_all + conf_cons_all) / 2
if np.max(np.abs(conf_recomputed_all - confs_all)) > 1e-8:
    print(
        "Warning: full-fold confidence decomposition does not match "
        "_compute_confidence(). Check the confidence implementation."
    )

neighbor_std_all = np.array([
    train_fold.iloc[idx_all[i]]["y"].values.astype(float).std()
    for i in range(len(test_fold))
])
neighbor_std_p90 = np.percentile(neighbor_std_all, 90)

# ── Representative query selection ───────────────────────────────
# Select a query that is representative of typical TRACE behavior:
# moderate-to-high confidence, low error, stable neighbor agreement.
p05, p50 = np.percentile(errors_all, [5, 55])
candidates = np.where(
    (confs_all >= 0.55) & (confs_all <= 0.85) &
    (conf_prox_all >= 0.45) &
    (errors_all >= p05) & (errors_all <= p50) &
    (pct_errors_all <= 4.0) &
    (neighbor_std_all <= neighbor_std_p90)
)[0]

if len(candidates) == 0:
    candidates = np.where(
        (confs_all >= 0.50) & (conf_prox_all >= 0.40) &
        (errors_all <= p50) & (pct_errors_all <= 12.0)
    )[0]

if len(candidates) == 0:
    candidates = np.where(
        (confs_all >= 0.45) & (conf_prox_all >= 0.35)
    )[0]

if len(candidates) == 0:
    candidates = np.arange(len(test_fold))

alpha_spread_score = np.std(alphas_f)
err_norm = errors_all[candidates] / (np.max(errors_all[candidates]) + EPS)

scores = (
    0.40 * confs_all[candidates] +
    0.35 * conf_prox_all[candidates] +
    0.15 * conf_cons_all[candidates] +
    0.10 * alpha_spread_score -
    0.25 * err_norm
)

QUERY_IDX = int(candidates[np.argmax(scores)])

# ── Final selected query ──────────────────────────────────────────
dists_q, idx_q = tree.query(
    test_match[QUERY_IDX:QUERY_IDX + 1], k=K_SHOW)

query_row   = test_fold.iloc[QUERY_IDX]
actual_y    = float(query_row["y"])
matched_yd  = y_diffs_tr[idx_q[0]]
predicted_y = float(_reconstruct_y(matched_yd.mean(), mean_y, std_y))

# Use confidence values already computed from the full test fold
conf_prox  = float(conf_prox_all[QUERY_IDX])
conf_cons  = float(conf_cons_all[QUERY_IDX])
confidence = float(confs_all[QUERY_IDX])

abs_error = abs(predicted_y - actual_y)
pct_error = abs_error / (abs(actual_y) + EPS) * 100

if confidence >= 0.75:
    conf_label, conf_color = "Very high", C_GREEN
elif confidence >= 0.50:
    conf_label, conf_color = "High", C_BLUE
elif confidence >= 0.25:
    conf_label, conf_color = "Medium", C_GOLD
else:
    conf_label, conf_color = "Low", C_RED

neighbor_rows = [train_fold.iloc[i] for i in idx_q[0]]
neighbor_y    = [float(r["y"]) for r in neighbor_rows]
ny_arr        = np.array(neighbor_y)
ny_std        = ny_arr.std()

consensus = (
    "strong"   if ny_std < 0.05 * abs(mean_y) else
    "moderate" if ny_std < 0.15 * abs(mean_y) else
    "weak"
)

err_color = (
    C_GREEN if pct_error < 3 else
    C_GOLD  if pct_error < 8 else
    C_RED
)

def shorten(name, n=13):
    return name if len(name) <= n else name[:n - 1] + "."

feat_labels = [shorten(f) for f in feature_cols]
n_feat      = len(feature_cols)

# ── Adaptive font sizes ───────────────────────────────────────────
_fs_val = max(7, 16 - max(0, n_feat - 5))
_fs_hdr = max(7, 15 - max(0, n_feat - 5))
_fs_lbl = max(8, 16 - max(0, n_feat - 5))
_fs_tgt = max(9, 15 - max(0, n_feat - 5))
FS_NOTE = 16

# ── Deviation matrix for panel b ──────────────────────────────────
query_devs    = test_diffs[QUERY_IDX]
neighbor_devs = train_diffs[idx_q[0]]
dev_matrix    = np.vstack([query_devs, neighbor_devs])
row_labels_hm = ["Query"] + [f"M{i+1}" for i in range(K_SHOW)]

print(
    f"Representative query idx={QUERY_IDX} | "
    f"conf={confidence:.3f} ({conf_label}) | "
    f"proximity={conf_prox:.3f} | consistency={conf_cons:.3f} | "
    f"error={abs_error:.3f} ({pct_error:.1f}%)"
)

# ── Alpha legend colors ───────────────────────────────────────────
def alpha_color(a):
    if a > 1.2:   return C_GOLD
    elif a > 0.8: return C_BLUE
    elif a > 0.1: return C_GRAY
    else:         return C_RED

a_colors       = [alpha_color(a) for a in alphas_f]
present_colors = set(a_colors)

legend_candidates = [
    (C_GOLD, "Superlinear  \u03b1 > 1.2"),
    (C_BLUE, "Near-linear  0.8\u20131.2"),
    (C_GRAY, "Sublinear  0.1\u20130.8"),
    (C_RED,  "Suppressed  \u03b1 \u2248 0"),
]

legend_els = [
    Patch(facecolor=c, label=lbl)
    for c, lbl in legend_candidates
    if c in present_colors
]

# ══════════════════════════════════════════════════════════════════
# Figure layout — 4 panels, vertical (Nature MI format)
# ══════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(13.5, 26))
fig.patch.set_facecolor(C_WHITE)

gs = gridspec.GridSpec(
    4, 1, figure=fig,
    left=0.06, right=0.96,
    top=0.92, bottom=0.03,
    hspace=0.20,
    height_ratios=[1.8, 1.4, 2.2, 1.8]
)

ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[1, 0])
ax_C = fig.add_subplot(gs[2, 0])
ax_D = fig.add_subplot(gs[3, 0])

for ax in [ax_A, ax_B, ax_C, ax_D]:
    ax.set_facecolor(C_VLIGHT)
    for spine in ax.spines.values():
        spine.set_visible(False)

# ══════════════════════════════════════════════════════════════════
# Panel a — Learned feature sensitivity
# ══════════════════════════════════════════════════════════════════
ax_A.set_xlim(0, 1)
ax_A.set_ylim(0, 1)
ax_A.axis("off")

ax_A.text(
    0.02, 0.945, "a",
    ha="left", va="center",
    fontsize=21, fontweight="bold", color=C_BLACK
)

ax_A.text(
    0.5, 0.945, "Learned feature sensitivity",
    ha="center", va="center",
    fontsize=19, fontweight="bold", color=C_BLUE
)

ax_A.text(
    0.5, 0.895,
    "Learned feature sensitivity exponents shaping the deviation-space geometry",
    ha="center", va="center",
    fontsize=FS_NOTE, color=C_GRAY, style="italic"
)

ax_bar = ax_A.inset_axes([0.13, 0.24, 0.78, 0.50])
ax_bar.set_facecolor(C_VLIGHT)

y_pos  = np.arange(n_feat)
bars_a = ax_bar.barh(
    y_pos, alphas_f,
    color=a_colors,
    height=0.62,
    edgecolor=C_WHITE,
    linewidth=0.7
)

ax_bar.axvline(1.0, color=C_DARK, linestyle="--", linewidth=1.1, alpha=0.35)
ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(feat_labels, fontsize=17)
ax_bar.set_xlabel("Sensitivity exponent \u03b1", fontsize=18, labelpad=10)
ax_bar.spines["top"].set_visible(False)
ax_bar.spines["right"].set_visible(False)
ax_bar.spines["left"].set_color(C_LGRAY)
ax_bar.spines["bottom"].set_color(C_LGRAY)
ax_bar.tick_params(colors=C_GRAY, labelsize=11)

for bar, val in zip(bars_a, alphas_f):
    ax_bar.text(
        val + 0.02, bar.get_y() + bar.get_height() / 2,
        f"{val:.2f}", va="center", ha="left",
        fontsize=16, color=C_DARK
    )

ncol = min(len(legend_els), 4)
ax_A.legend(
    handles=legend_els, fontsize=15.5,
    loc="center", bbox_to_anchor=(0.5, 0.815),
    frameon=True, framealpha=0.96, edgecolor=C_LGRAY,
    ncol=ncol, handlelength=1.2, handletextpad=0.5, columnspacing=0.9
)

top2 = np.argsort(alphas_f)[::-1][:2]
bot1 = int(np.argmin(alphas_f))

ax_A.text(
    0.5, 0.085,
    f"{feat_labels[top2[0]]} (\u03b1={alphas_f[top2[0]]:.2f}) and "
    f"{feat_labels[top2[1]]} (\u03b1={alphas_f[top2[1]]:.2f}) dominated;  "
    f"{feat_labels[bot1]} had the lowest sensitivity "
    f"(\u03b1={alphas_f[bot1]:.2f}).",
    ha="center", va="center", fontsize=FS_NOTE,
    color="#1E3A8A", style="italic",
    bbox=dict(facecolor=C_VLIGHT, edgecolor="none", pad=1.5)
)

# ══════════════════════════════════════════════════════════════════
# Panel b — Deviation profiles
# ══════════════════════════════════════════════════════════════════
ax_B.set_xlim(0, 1)
ax_B.set_ylim(0, 1)
ax_B.axis("off")

ax_B.axhline(0.990, color=C_PURPLE, linewidth=4, alpha=0.85)

ax_B.text(
    0.02, 0.930, "b",
    ha="left", va="center",
    fontsize=21, fontweight="bold", color=C_BLACK
)

ax_B.text(
    0.5, 0.930, "Deviation profiles used for matching",
    ha="center", va="center",
    fontsize=19, fontweight="bold", color=C_PURPLE
)

ax_B.text(
    0.5, 0.840,
    "\u03b1-transformed percentage deviations from training mean "
    "\u2014 the geometry TRACE uses for nearest-neighbor retrieval",
    ha="center", va="center",
    fontsize=FS_NOTE, color=C_GRAY, style="italic"
)

ax_B.axhline(0.760, color=C_LGRAY, linewidth=0.9)

ax_hm = ax_B.inset_axes([0.08, 0.05, 0.78, 0.65])
ax_hm.set_facecolor(C_VLIGHT)

vmax = np.percentile(np.abs(dev_matrix), 95)
vmax = max(vmax, 1.0)

norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im   = ax_hm.imshow(dev_matrix, cmap="RdBu_r", norm=norm, aspect="auto")

ax_hm.set_xticks(range(n_feat))
ax_hm.set_xticklabels(feat_labels, fontsize=_fs_hdr, rotation=35, ha="right")
ax_hm.set_yticks(range(K_SHOW + 1))
ax_hm.set_yticklabels(row_labels_hm, fontsize=_fs_lbl)

for r in range(K_SHOW + 1):
    for c in range(n_feat):
        val       = dev_matrix[r, c]
        txt_color = "white" if abs(val) > vmax * 0.6 else C_DARK
        ax_hm.text(
            c, r, f"{val:.1f}",
            ha="center", va="center",
            fontsize=_fs_val, color=txt_color
        )

cbar_ax = ax_B.inset_axes([0.88, 0.10, 0.03, 0.55])
cbar    = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Deviation\nfrom mean (%)", fontsize=_fs_hdr, labelpad=4)
cbar.ax.tick_params(labelsize=_fs_hdr)
cbar.ax.axhline(0, color=C_DARK, linewidth=1.0, alpha=0.5)

# Highlight query row in deviation heatmap
for c in range(n_feat):
    ax_hm.add_patch(
        Rectangle(
            (c - 0.5, -0.5), 1, 1,
            fill=False, edgecolor=C_GOLD, linewidth=1.8, zorder=5
        )
    )

# ══════════════════════════════════════════════════════════════════
# Panel c — Matched training instances
# ══════════════════════════════════════════════════════════════════
ax_C.set_xlim(0, 1)
ax_C.set_ylim(0, 1)
ax_C.axis("off")

ax_C.axhline(0.990, color=C_GOLD, linewidth=4, alpha=0.85)

ax_C.text(
    0.02, 0.955, "c",
    ha="left", va="center",
    fontsize=21, fontweight="bold", color=C_BLACK
)

ax_C.text(
    0.5, 0.955, "Matched training instances",
    ha="center", va="center",
    fontsize=19, fontweight="bold", color=C_GOLD
)

ax_C.text(
    0.5, 0.910,
    f"{K_SHOW} nearest neighbors retrieved in deviation space "
    f"\u2014 raw values shown for interpretability",
    ha="center", va="center",
    fontsize=FS_NOTE, color=C_GRAY, style="italic"
)

ax_C.axhline(0.875, color=C_LGRAY, linewidth=0.9)

lbl_w  = 0.075
tgt_w  = 0.085
feat_w = (0.92 - lbl_w - tgt_w) / n_feat
x0     = 0.04 + lbl_w
row_h  = min(0.085, 0.60 / (K_SHOW + 2))
hdr_y  = 0.800

ax_C.add_patch(
    Rectangle(
        (0.03, hdr_y - row_h * 0.55), 0.94, row_h,
        facecolor=C_DARK, alpha=0.07, linewidth=0
    )
)

for j, name in enumerate(feat_labels):
    x = x0 + j * feat_w + feat_w / 2
    ax_C.text(
        x, hdr_y, name,
        ha="center", va="center",
        fontsize=_fs_hdr, fontweight="bold", color=C_DARK,
        rotation=35 if n_feat > 5 else 0
    )

ax_C.text(
    x0 + n_feat * feat_w + tgt_w / 2, hdr_y,
    TARGET_UNIT,
    ha="center", va="center",
    fontsize=_fs_tgt, fontweight="bold", color=C_GOLD
)

qy = hdr_y - row_h * 1.35

ax_C.add_patch(
    Rectangle(
        (0.03, qy - row_h * 0.52), 0.94, row_h,
        facecolor=C_GOLD, alpha=0.14, linewidth=0
    )
)

ax_C.text(
    0.03 + lbl_w * 0.5, qy, "Query",
    ha="center", va="center",
    fontsize=_fs_lbl, fontweight="bold", color=C_GOLD
)

for j, col in enumerate(feature_cols):
    x = x0 + j * feat_w + feat_w / 2
    ax_C.text(
        x, qy, f"{query_row[col]:.2f}",
        ha="center", va="center",
        fontsize=_fs_val, fontweight="bold", color=C_DARK
    )

ax_C.text(
    x0 + n_feat * feat_w + tgt_w / 2, qy,
    f"{actual_y:.3f}",
    ha="center", va="center",
    fontsize=_fs_tgt, fontweight="bold", color=C_GOLD
)

for i, (nrow, ny) in enumerate(zip(neighbor_rows, neighbor_y)):
    ry = qy - row_h * (i + 1.18)

    if i % 2 == 0:
        ax_C.add_patch(
            Rectangle(
                (0.03, ry - row_h * 0.52), 0.94, row_h,
                facecolor=C_LBLUE, alpha=0.35, linewidth=0
            )
        )

    ax_C.text(
        0.03 + lbl_w * 0.5, ry, f"M{i+1}",
        ha="center", va="center",
        fontsize=_fs_lbl, color=C_BLUE, fontweight="bold"
    )

    for j, col in enumerate(feature_cols):
        x = x0 + j * feat_w + feat_w / 2
        ax_C.text(
            x, ry, f"{nrow[col]:.2f}",
            ha="center", va="center",
            fontsize=_fs_val, color=C_DARK
        )

    ax_C.text(
        x0 + n_feat * feat_w + tgt_w / 2, ry,
        f"{ny:.3f}",
        ha="center", va="center",
        fontsize=_fs_tgt, fontweight="bold", color=C_BLUE
    )

ax_C.axhline(0.055, color=C_LGRAY, linewidth=0.9)

ax_C.text(
    0.5, 0.022,
    f"Neighbor target range: {ny_arr.min():.3f}"
    f"\u2013{ny_arr.max():.3f} {TARGET_UNIT};  "
    f"std = {ny_std:.3f};  {consensus} agreement",
    ha="center", va="center",
    fontsize=FS_NOTE, color=C_AMBER, style="italic"
)

# ══════════════════════════════════════════════════════════════════
# Panel d — Prediction reliability
# ══════════════════════════════════════════════════════════════════
ax_D.set_xlim(0, 1)
ax_D.set_ylim(0, 1)
ax_D.axis("off")

ax_D.axhline(0.985, color=C_GREEN, linewidth=4, alpha=0.85)

ax_D.text(
    0.02, 0.945, "d",
    ha="left", va="center",
    fontsize=23, fontweight="bold", color=C_BLACK
)

ax_D.text(
    0.5, 0.945, "Prediction reliability",
    ha="center", va="center",
    fontsize=21, fontweight="bold", color=C_GREEN
)

ax_D.text(
    0.5, 0.895,
    "Prediction from matched targets and confidence decomposition",
    ha="center", va="center",
    fontsize=FS_NOTE, color=C_GRAY, style="italic"
)

ax_D.axhline(0.860, color=C_LGRAY, linewidth=0.9)

left_panel  = ax_D.inset_axes([0.035, 0.18, 0.48, 0.58])
right_panel = ax_D.inset_axes([0.590, 0.20, 0.38, 0.50])

for subax in [left_panel, right_panel]:
    subax.set_facecolor(C_VLIGHT)
    for sp in subax.spines.values():
        sp.set_visible(False)

left_panel.set_xlim(0, 1)
left_panel.set_ylim(0, 1)
left_panel.axis("off")

left_panel.text(
    0.5, 0.965, "Prediction summary",
    ha="center", va="center",
    fontsize=18, fontweight="bold", color=C_DARK
)

cards = [
    ("Predicted",  f"{predicted_y:.3f} {TARGET_UNIT}",        C_BLUE,      0.76),
    ("Actual",     f"{actual_y:.3f} {TARGET_UNIT}",            C_DARK,      0.57),
    ("Error",      f"{abs_error:.3f} ({pct_error:.1f}%)",      err_color,   0.38),
    ("Confidence", f"{confidence:.3f} \u2014 {conf_label}",    conf_color,  0.19),
]

for label, val, color, yc in cards:
    left_panel.add_patch(
        FancyBboxPatch(
            (0.02, yc - 0.070), 0.96, 0.120,
            boxstyle="round,pad=0.010",
            facecolor=color, alpha=0.10,
            linewidth=0.9, edgecolor=color
        )
    )
    left_panel.text(
        0.08, yc, label,
        ha="left", va="center",
        fontsize=17, color=C_GRAY
    )
    left_panel.text(
        0.95, yc, val,
        ha="right", va="center",
        fontsize=18.5 if label == "Confidence" else 19,
        fontweight="bold", color=color
    )

right_panel.set_xlim(-0.75, 3.15)
right_panel.set_ylim(0, 1.22)
right_panel.set_title(
    "Confidence decomposition",
    fontsize=18, fontweight="bold", color=C_DARK, pad=10
)

c_vals = [conf_prox, conf_cons, confidence]
c_xpos = np.array([0.0, 1.25, 2.50])
c_clrs = [C_BLUE, C_GREEN, conf_color]

bars_c = right_panel.bar(
    c_xpos, c_vals,
    color=c_clrs, width=0.42,
    edgecolor=C_WHITE, linewidth=0.8
)

right_panel.set_xticks(c_xpos)
right_panel.set_xticklabels(
    ["Proximity", "Consistency", "Confidence"],
    fontsize=14.5
)
right_panel.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
right_panel.tick_params(axis="y", labelsize=10.5, colors=C_GRAY)
right_panel.tick_params(axis="x", labelsize=13.0, colors=C_DARK, pad=8)

right_panel.axhline(0.75, color=C_GOLD, linestyle="--", linewidth=1.2, alpha=0.8)
right_panel.axhline(0.50, color=C_GRAY, linestyle="--", linewidth=1.0, alpha=0.7)

right_panel.text(3.72, 0.85, "Very high", va="center", ha="right",
                 fontsize=13.5, color=C_GOLD)
right_panel.text(3.52, 0.60, "High",      va="center", ha="right",
                 fontsize=13.5, color=C_GRAY)

right_panel.spines["top"].set_visible(False)
right_panel.spines["right"].set_visible(False)
right_panel.spines["left"].set_color(C_LGRAY)
right_panel.spines["bottom"].set_color(C_LGRAY)

for bar, val in zip(bars_c, c_vals):
    right_panel.text(
        bar.get_x() + bar.get_width() / 2, val + 0.035,
        f"{val:.3f}",
        ha="center", va="bottom",
        fontsize=15.5, fontweight="bold", color=C_DARK
    )

prox_w = (
    "close"    if conf_prox >= 0.60 else
    "moderate" if conf_prox >= 0.40 else
    "distant"
)
cons_w = (
    "consistent"   if conf_cons >= 0.60 else
    "moderate"     if conf_cons >= 0.40 else
    "inconsistent"
)

ax_D.axhline(0.090, color=C_LGRAY, linewidth=0.9)

ax_D.text(
    0.5, 0.040,
    f"{prox_w.capitalize()} proximity and {cons_w} neighbor "
    f"agreement yield {conf_label.lower()} confidence "
    f"({confidence:.3f}).",
    ha="center", va="center",
    fontsize=FS_NOTE, color="#065F46", style="italic"
)

# ── Increase all font sizes by one point ──────────────────────────
for txt in fig.findobj(match=Text):
    txt.set_fontsize(txt.get_fontsize() + 1)

# ── Save ──────────────────────────────────────────────────────────
plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight", facecolor=C_WHITE)
plt.savefig(SAVE_PATH.replace(".pdf", ".png"), dpi=300,
            bbox_inches="tight", facecolor=C_WHITE)
plt.show()

print(f"Saved: {SAVE_PATH}")
print(f"Saved: {SAVE_PATH.replace('.pdf', '.png')}")
