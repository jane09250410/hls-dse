"""Figure: Dynamatic Main Results v2.

Two-panel figure for Dynamatic evaluation:
  (a) Overall SR comparison: bar chart with error bars, 7 methods
  (b) Per-benchmark boxplot - 4 benchmarks

IMPORTANT: fir and histogram achieve SR=100% for ALL methods on Dynamatic,
so they carry no discriminative signal. The paper (Sec. IV.A) evaluates
Dynamatic on 4 benchmarks only: gcd, matching, binary_search, kernel_2mm.
We apply this filter here so all numbers agree with Table III.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/Users/zhangxinyu/Desktop/hls/results")
OUT  = Path("/Users/zhangxinyu/Desktop/hls/paper_figures/out")
OUT.mkdir(parents=True, exist_ok=True)

# Dynamatic 4-benchmark evaluation set (fir/histogram excluded)
DYNAMATIC_BENCHMARKS = ["gcd", "matching", "binary_search", "kernel_2mm"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# ==================== Load data ====================
dyn_main  = pd.read_csv(ROOT / "master/dynamatic_main/run_summary.csv")
dyn_perms = pd.read_csv(ROOT / "rerun/dynamatic_pa_dse_perms/run_summary.csv")

dyn_main_clean = dyn_main[~dyn_main["strategy"].str.contains("PA-DSE")].copy()
dyn_perms_clean = dyn_perms.copy()
dyn_perms_clean["strategy"] = "PA-DSE"

RENAME = {
    "Random":             "Random",
    "Filtered_Random":    "FilteredRandom",
    "SimulatedAnnealing": "SA",
    "GeneticAlgorithm":   "GA",
    "GP-BO":              "GP-BO",
    "RF_Classifier":      "RF",
}
dyn_main_clean["strategy"] = dyn_main_clean["strategy"].map(RENAME).fillna(dyn_main_clean["strategy"])

df = pd.concat([dyn_main_clean, dyn_perms_clean], ignore_index=True)

# Filter to 4 benchmarks
before = len(df)
df = df[df["benchmark"].isin(DYNAMATIC_BENCHMARKS)].copy()
print(f"[filter] Dynamatic 4 benchmarks: {before} -> {len(df)} rows")

# ==================== Method order + colors ====================
METHOD_ORDER = ["Random", "FilteredRandom", "SA", "GA", "GP-BO", "RF", "PA-DSE"]
BENCH_ORDER_PANEL_B = DYNAMATIC_BENCHMARKS

COLORS = {
    "Random":         "#8B8B8B",
    "FilteredRandom": "#A0A0A0",
    "SA":             "#4E79A7",
    "GA":             "#59A14F",
    "GP-BO":          "#F28E2B",
    "RF":             "#E15759",
    "PA-DSE":         "#C1272D",
}

# ==================== Create figure ====================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2),
                                gridspec_kw={"width_ratios": [1, 1.4]})

# Panel (a)
summary = df.groupby("strategy")["sr_pct"].agg(["mean", "std", "count"]).reset_index()
summary = summary.set_index("strategy").loc[METHOD_ORDER].reset_index()
print("\n=== Panel (a) values ===")
print(summary.round(2).to_string(index=False))

x = np.arange(len(METHOD_ORDER))
bars = axL.bar(
    x, summary["mean"],
    yerr=summary["std"],
    color=[COLORS[m] for m in METHOD_ORDER],
    edgecolor="black", linewidth=0.6,
    capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "#333"},
)

for i, method in enumerate(METHOD_ORDER):
    row = summary[summary["strategy"] == method].iloc[0]
    m = row["mean"]
    s = row["std"]
    y_pos = m + s + 2
    if method == "PA-DSE":
        axL.text(i, y_pos, f"{m:.1f}",
                 ha="center", fontsize=9, fontweight="bold",
                 color=COLORS["PA-DSE"])
    else:
        axL.text(i, y_pos, f"{m:.1f}",
                 ha="center", fontsize=8, color="#333")

axL.set_xticks(x)
axL.set_xticklabels(METHOD_ORDER, rotation=25, ha="right")
axL.set_ylabel("Success Rate (%)")
axL.set_ylim(0, 115)
axL.set_title("(a) Overall success rate (4 benchmarks)")
axL.grid(axis="y", alpha=0.3)

# Panel (b)
n_methods = len(METHOD_ORDER)
n_bench   = len(BENCH_ORDER_PANEL_B)
width = 0.11
group_width = width * n_methods * 1.1

positions = []
box_data = []
box_colors = []

for bi, bench in enumerate(BENCH_ORDER_PANEL_B):
    center = bi * (group_width + 0.20)
    for mi, method in enumerate(METHOD_ORDER):
        pos = center - group_width / 2 + (mi + 0.5) * width
        values = df[(df["strategy"] == method) & (df["benchmark"] == bench)]["sr_pct"].values
        if len(values) > 0:
            positions.append(pos)
            box_data.append(values)
            box_colors.append(COLORS[method])

bp = axR.boxplot(box_data, positions=positions, widths=width * 0.85,
                 patch_artist=True, showfliers=False,
                 medianprops={"color": "black", "linewidth": 1.0},
                 whiskerprops={"color": "#333", "linewidth": 0.7},
                 capprops={"color": "#333", "linewidth": 0.7})

for patch, color in zip(bp["boxes"], box_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
    patch.set_edgecolor("black")
    patch.set_linewidth(0.5)

group_centers = [bi * (group_width + 0.20) for bi in range(n_bench)]
axR.set_xticks(group_centers)
axR.set_xticklabels(BENCH_ORDER_PANEL_B, rotation=15, ha="right")
axR.set_ylabel("Success Rate (%)")
axR.set_ylim(0, 105)
axR.set_title("(b) Per-benchmark distribution (4 benchmarks)")
axR.grid(axis="y", alpha=0.3)

from matplotlib.patches import Patch
legend_handles = [Patch(facecolor=COLORS[m], edgecolor="black", label=m)
                  for m in METHOD_ORDER]
axR.legend(handles=legend_handles, loc="lower left", ncol=2,
           fontsize=8, framealpha=0.95)

plt.tight_layout()
plt.savefig(OUT / "fig_dynamatic_main.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_dynamatic_main.png", bbox_inches="tight", dpi=300)
print(f"\nSaved: {OUT}/fig_dynamatic_main.pdf")

plt.show()

# Print table
print("\n" + "=" * 80)
print("PAPER-READY TABLE (Dynamatic Main Results, 4 benchmarks)")
print("=" * 80)
print(f"{'Method':<15s} {'Mean':>8s} {'Std':>8s} {'n':>6s}")
print("-" * 45)
for m in METHOD_ORDER:
    sub = df[df["strategy"] == m]
    mean = sub["sr_pct"].mean()
    std  = sub["sr_pct"].std()
    n    = len(sub)
    print(f"{m:<15s} {mean:>7.2f}  {std:>7.2f} {n:>6d}")
