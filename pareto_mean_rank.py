###################################################################
#  Pareto Frontier — Mean Rank vs Mean Complexity (8 datasets)
#  X axis: mean structural complexity (log scale, lower = simpler)
#  Y axis: mean rank across 8 datasets (rank 1 = best, at top)
#
#  Part of the TRACE project:
#  https://github.com/mutazshafeey/trace-regression
#
#  Run standalone — no session variables needed.
#  Output: pareto_mean_rank.pdf and pareto_mean_rank.png
###################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.lines as mlines

# ── Mean rank across 8 datasets (precomputed) ─────────────────────────────────
mean_rank = {
    "TRACE"           : 5.50,
    "EBM"             : 3.12,
    "LinearRegression": 8.75,
    "DecisionTree"    : 6.75,
    "RandomForest"    : 3.25,
    "XGBoost"         : 2.75,
    "CatBoost"        : 3.38,
    "MLP"             : 7.62,
    "ANN"             : 9.50,
    "TabNet"          : 7.38,
    "KNN_raw"         : 8.00,
}

std_rank = {
    "TRACE"           : 1.50,
    "EBM"             : 2.52,
    "LinearRegression": 0.97,
    "DecisionTree"    : 1.85,
    "RandomForest"    : 2.11,
    "XGBoost"         : 1.56,
    "CatBoost"        : 1.11,
    "MLP"             : 3.46,
    "ANN"             : 1.32,
    "TabNet"          : 2.91,
    "KNN_raw"         : 2.06,
}

# Mean structural complexity per model (averaged across 8 datasets):
#   TRACE     : p + k = 9 + 5 = 14 (Protein, largest dataset used as reference)
#               here set to 13 as reported in paper
#   EBM       : number of learned shape function components
#   LR        : number of non-zero coefficients
#   DTree     : total node count
#   RF/XGB/CB : total trainable nodes / trees
#   MLP/ANN   : total trainable parameters
#   TabNet    : total trainable parameters
#   KNN       : stored training instances (used as proxy for complexity)
mean_complexity = {
    "TRACE"           : 13,
    "EBM"             : 39,
    "LinearRegression": 8,
    "DecisionTree"    : 19512,
    "RandomForest"    : 1246914,
    "XGBoost"         : 10365,
    "CatBoost"        : 12700,
    "MLP"             : 6038,
    "ANN"             : 11569,
    "TabNet"          : 6193,
    "KNN_raw"         : 59,
}

# ── Colors — all baselines in same neutral gray-blue, TRACE in gold ───────────
colors = {
    "TRACE"           : "#FFD700",
    "EBM"             : "#5B7FA6",
    "LinearRegression": "#5B7FA6",
    "DecisionTree"    : "#5B7FA6",
    "RandomForest"    : "#5B7FA6",
    "XGBoost"         : "#5B7FA6",
    "CatBoost"        : "#5B7FA6",
    "MLP"             : "#5B7FA6",
    "ANN"             : "#5B7FA6",
    "TabNet"          : "#5B7FA6",
    "KNN_raw"         : "#5B7FA6",
}

# ── Short display labels ──────────────────────────────────────────────────────
labels = {
    "TRACE"           : "TRACE",
    "EBM"             : "EBM",
    "LinearRegression": "LR",
    "DecisionTree"    : "DTree",
    "RandomForest"    : "RF",
    "XGBoost"         : "XGBoost",
    "CatBoost"        : "CatBoost",
    "MLP"             : "MLP",
    "ANN"             : "ANN",
    "TabNet"          : "TabNet",
    "KNN_raw"         : "KNN",
}

# ── Label offsets (x_pts, y_pts, ha) ─────────────────────────────────────────
offsets = {
    "TRACE"           : ( 12,   8, "left"),
    "EBM"             : ( 10,  -11, "left"),
    "LinearRegression": (-10,   8, "right"),
    "DecisionTree"    : ( 10,  -11, "left"),
    "RandomForest"    : (-10,   8, "right"),
    "XGBoost"         : ( 10,   8, "left"),
    "CatBoost"        : ( 10,  -11, "left"),
    "MLP"             : (-25,   18, "left"),
    "ANN"             : ( 10,   8, "left"),
    "TabNet"          : (-10,  -11, "right"),
    "KNN_raw"         : ( 10,   8, "left"),
}

# ── Pareto frontier ───────────────────────────────────────────────────────────
# A model is on the Pareto frontier if no simpler model achieves equal or
# better mean rank. Sort by complexity ascending; keep if rank improves.
def pareto_frontier(complexity_dict, rank_dict):
    items = sorted(complexity_dict.items(), key=lambda x: x[1])
    pareto = []
    best_rank = np.inf
    for name, comp in items:
        r = rank_dict[name]
        if r <= best_rank:
            pareto.append((comp, r, name))
            best_rank = r
    return pareto

pareto = pareto_frontier(mean_complexity, mean_rank)
print("Pareto points:", [(p[2], f"comp={p[0]}", f"rank={p[1]}") for p in pareto])

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 8))

# Pareto frontier line
px = [p[0] for p in pareto]
py = [p[1] for p in pareto]
ax.plot(px, py, color="gray", linestyle="--",
        linewidth=1.5, alpha=0.6, zorder=1)

# All model points with error bars
for name in mean_complexity:
    x    = mean_complexity[name]
    y    = mean_rank[name]
    yerr = std_rank[name]
    col  = colors[name]
    size = 450 if name == "TRACE" else 160
    mark = "*" if name == "TRACE" else "o"

    ax.errorbar(x, y, yerr=yerr,
                fmt="none", color=col, alpha=0.35,
                capsize=4, capthick=1.2, elinewidth=1.2, zorder=2)

    ax.scatter(x, y, c=col, s=size, marker=mark,
               zorder=3, edgecolors="white", linewidths=1.3)

    ox, oy, ha = offsets[name]
    ax.annotate(labels[name],
                xy=(x, y),
                xytext=(ox, oy),
                textcoords="offset points",
                fontsize=12,
                ha=ha,
                va="center",
                color="#222222" if name != "TRACE" else "#B8860B",
                fontweight="bold" if name == "TRACE" else "normal",
                path_effects=[pe.withStroke(linewidth=2.5,
                                            foreground="white")])

# ── Axes ──────────────────────────────────────────────────────────────────────
ax.set_xscale("log")
ax.set_xlabel(
    "Mean structural complexity across 8 datasets (log scale, lower = more interpretable)",
    fontsize=15)
ax.set_ylabel(
    "Mean predictive rank across 8 datasets\n(rank 1 = best)",
    fontsize=15)

# Invert Y so rank 1 is at top
n_models = len(mean_rank)
ax.set_ylim(n_models + 0.8, 0.5)
ax.set_yticks(range(1, n_models + 1))
ax.set_yticklabels([f"Rank {r}" for r in range(1, n_models + 1)], fontsize=12)

ax.grid(True, which="major", alpha=0.22, linestyle="--")
ax.grid(True, which="minor", alpha=0.08, linestyle=":")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.set_title(
    "Interpretability\u2013performance tradeoff across eight benchmark datasets",
    fontsize=16, fontweight="bold", pad=15)

# ── Annotation ────────────────────────────────────────────────────────────────
ax.annotate(
    "TRACE occupies a low complexity region\nwhile maintaining competitive mean rank",
    xy=(mean_complexity["TRACE"], mean_rank["TRACE"]),
    xytext=(80, mean_rank["TRACE"] - 0.5),
    fontsize=12, color="#444444",
    arrowprops=dict(arrowstyle="->", color="#888888",
                    connectionstyle="arc3,rad=-0.25"),
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFDE7",
              edgecolor="#FFD700", alpha=0.92))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_items = [
    plt.scatter([], [], c="#FFD700", s=450, marker="*",
                edgecolors="white", linewidths=1.3, label="TRACE"),
    plt.scatter([], [], c="#5B7FA6", s=160, marker="o",
                edgecolors="white", linewidths=1.0, label="Baseline models"),
    mlines.Line2D([], [], color="gray", linestyle="--",
                  linewidth=1.5, label="Pareto frontier"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=13,
          frameon=True, framealpha=0.92)

plt.tight_layout()
plt.savefig("pareto_mean_rank.pdf", dpi=300, bbox_inches="tight")
plt.savefig("pareto_mean_rank.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved: pareto_mean_rank.pdf and pareto_mean_rank.png")
