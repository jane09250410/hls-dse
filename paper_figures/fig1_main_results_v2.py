"""Figure 1 (v2): Bambu main results — bar chart + distribution box plot.

Reads SAME source as Table II (compute_paper_tables.py):
  - results/master/bambu_main/run_summary.csv      (baselines; PA-DSE rows dropped)
  - results/rerun/bambu_pa_dse_perms/run_summary.csv (PA-DSE perms, 80 runs = 10 perm x 8 bench)

Panel (a): Overall mean SR (%) with std error bars.
Panel (b): Per-method SR distribution across all runs (boxplot + strip).

Output: paper_figures/out/fig1_bambu_main.{pdf,png}
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from plot_style import setup, COLORS, METHOD_ORDER, save_fig

setup()

# ============================================================
# DATA — identical to compute_paper_tables.py so figure and
# Table II are guaranteed to agree.
# ============================================================
ROOT = Path("/Users/zhangxinyu/Desktop/hls/results")

BASELINE_MAP = {
    "Random":             "Random",
    "Filtered_Random":    "Filtered_Random",
    "SimulatedAnnealing": "SA",
    "GeneticAlgorithm":   "GA",
    "GP-BO":              "GP-BO",
    "RF_Classifier":      "RF",
}

main = pd.read_csv(ROOT / "master/bambu_main/run_summary.csv")
perms = pd.read_csv(ROOT / "rerun/bambu_pa_dse_perms/run_summary.csv")

# Drop any PA-DSE rows from master so we use perms as the authoritative PA-DSE source.
main = main[~main["strategy"].str.contains("PA-DSE")].copy()
main["strategy"] = main["strategy"].map(BASELINE_MAP).fillna(main["strategy"])
perms = perms.copy()
perms["strategy"] = "PA-DSE"

df = pd.concat([main, perms], ignore_index=True)
df = df.rename(columns={"strategy": "method"})
print(f"Loaded {len(df)} runs")
print(df.groupby("method").size().to_string())

# Overall stats
overall = df.groupby("method")["sr_pct"].agg(["mean", "std", "count"]).reindex(METHOD_ORDER)
print("\nPer-method stats:")
print(overall.round(2))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3),
                                gridspec_kw={"width_ratios": [1.0, 1.1]})

# ============ Panel (a): bar chart ============
y_pos = np.arange(len(METHOD_ORDER))
means = overall["mean"].values
stds = overall["std"].values
bar_colors = [COLORS[m] for m in METHOD_ORDER]

ax1.barh(y_pos, means, xerr=stds, color=bar_colors,
         edgecolor="black", linewidth=0.5,
         error_kw={"elinewidth": 0.8, "capsize": 3, "ecolor": "#333"})

for i, (m, s) in enumerate(zip(means, stds)):
    color = "#C1272D" if METHOD_ORDER[i] == "PA-DSE" else "black"
    weight = "bold" if METHOD_ORDER[i] == "PA-DSE" else "normal"
    ax1.text(m + s + 2, i, f"{m:.1f}\u00B1{s:.1f}",
             va="center", ha="left", fontsize=8.5,
             color=color, fontweight=weight)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(METHOD_ORDER)
ax1.invert_yaxis()
ax1.set_xlabel("Success Rate (%)")
ax1.set_xlim(0, 115)
ax1.set_title("(a) Mean SR (across all runs)")
ax1.grid(axis="y", visible=False)

for label in ax1.get_yticklabels():
    if label.get_text() == "PA-DSE":
        label.set_color("#C1272D")
        label.set_fontweight("bold")

# ============ Panel (b): distribution boxplot ============
data = [df[df["method"] == m]["sr_pct"].values for m in METHOD_ORDER]

bp = ax2.boxplot(data, vert=True, widths=0.55,
                 patch_artist=True,
                 medianprops={"color": "black", "linewidth": 1.3},
                 whiskerprops={"color": "#555", "linewidth": 0.8},
                 capprops={"color": "#555", "linewidth": 0.8},
                 flierprops={"marker": "o", "markersize": 3,
                             "markerfacecolor": "#888", "markeredgecolor": "none",
                             "alpha": 0.6})

# Color each box
for patch, method in zip(bp["boxes"], METHOD_ORDER):
    patch.set_facecolor(COLORS[method])
    patch.set_edgecolor("black")
    patch.set_linewidth(0.5)
    patch.set_alpha(0.85)

# Overlay raw data points (strip plot) for transparency
rng = np.random.default_rng(42)
for i, (method, vals) in enumerate(zip(METHOD_ORDER, data)):
    jitter = rng.uniform(-0.15, 0.15, size=len(vals))
    ax2.scatter(np.full(len(vals), i + 1) + jitter, vals,
                s=8, color="black", alpha=0.35, zorder=3,
                linewidth=0)

ax2.set_xticks(np.arange(1, len(METHOD_ORDER) + 1))
ax2.set_xticklabels(METHOD_ORDER, rotation=25, ha="right")
ax2.set_ylabel("Success Rate (%)")
ax2.set_ylim(-5, 105)
ax2.set_title("(b) SR distribution across runs")
ax2.grid(axis="x", visible=False)

for label in ax2.get_xticklabels():
    if label.get_text() == "PA-DSE":
        label.set_color("#C1272D")
        label.set_fontweight("bold")

plt.tight_layout()
save_fig(fig, "fig1_bambu_main")
plt.show()