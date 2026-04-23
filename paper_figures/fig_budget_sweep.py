"""Figure: Budget Sweep.

Shows SR vs budget B for 3 PA-DSE configurations on 2 representative benchmarks:
  - vadd (Bambu)
  - gcd  (Dynamatic)

Budgets tested: 20, 40, 60, 80, 100, 120
Configurations: phago+Full, phago+RPE, phago+OFRS

Illustrates:
  - How SR scales with budget
  - Role reversal: RPE dominates on Bambu, OFRS dominates on Dynamatic
  - At B=60 (standard), phago+Full catches up in both

Input:  master/budget_sweep/run_summary.csv
Output: paper_figures/out/fig_budget_sweep.{pdf,png}
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/Users/zhangxinyu/Desktop/hls/results")
OUT  = Path("/Users/zhangxinyu/Desktop/hls/paper_figures/out")
OUT.mkdir(parents=True, exist_ok=True)

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

# ==================== Load ====================
df = pd.read_csv(ROOT / "master/budget_sweep/run_summary.csv")
print(f"Loaded {len(df)} runs")
print(df.groupby(["benchmark", "ablation_config", "budget"])["sr_pct"].agg(["mean", "std"]).round(2))

# ==================== Setup ====================
CONFIGS = ["phago+Full", "phago+RPE", "phago+OFRS"]
LABELS  = {
    "phago+Full": "SCF + RPE + OFRS (Full)",
    "phago+RPE":  "SCF + RPE only",
    "phago+OFRS": "SCF + OFRS only",
}
COLORS  = {
    "phago+Full": "#C1272D",   # red
    "phago+RPE":  "#F28E2B",   # orange
    "phago+OFRS": "#4E79A7",   # blue
}
MARKERS = {
    "phago+Full": "o",
    "phago+RPE":  "s",
    "phago+OFRS": "^",
}

BENCH_TO_TOOL = {"vadd": "Bambu", "gcd": "Dynamatic"}

# ==================== Figure ====================
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

for ax_idx, (bench, tool) in enumerate(BENCH_TO_TOOL.items()):
    ax = axes[ax_idx]
    sub = df[df["benchmark"] == bench]

    for cfg in CONFIGS:
        sub_cfg = sub[sub["ablation_config"] == cfg]
        grouped = sub_cfg.groupby("budget")["sr_pct"].agg(["mean", "std"]).reset_index()

        # Line + markers
        ax.plot(grouped["budget"], grouped["mean"],
                color=COLORS[cfg], marker=MARKERS[cfg],
                linewidth=1.6, markersize=7, markeredgecolor="black",
                markeredgewidth=0.5, label=LABELS[cfg])
        # Shaded error band
        ax.fill_between(grouped["budget"],
                         grouped["mean"] - grouped["std"],
                         grouped["mean"] + grouped["std"],
                         color=COLORS[cfg], alpha=0.15, linewidth=0)

    ax.set_xlabel("Budget $B$ (number of evaluations)")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title(f"({'ab'[ax_idx]}) {bench} ({tool})")
    ax.set_xticks([20, 40, 60, 80, 100, 120])
    ax.set_ylim(0, 105)
    ax.set_xlim(15, 125)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    # Mark B=60 as standard budget (vertical dashed line)
    ax.axvline(x=60, color="#888", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.text(60, 8, "B=60\n(default)", fontsize=7, color="#666",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.8))

plt.tight_layout()
plt.savefig(OUT / "fig_budget_sweep.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_budget_sweep.png", bbox_inches="tight", dpi=300)
print(f"\n✅ Saved: {OUT}/fig_budget_sweep.pdf")
plt.show()

# ==================== Paper table ====================
print("\n" + "=" * 70)
print("Budget Sweep — SR (%) vs budget B")
print("=" * 70)
table = df.pivot_table(index=["benchmark", "ablation_config"],
                        columns="budget", values="sr_pct",
                        aggfunc="mean").round(1)
print(table)
print("\nStd dev:")
table_std = df.pivot_table(index=["benchmark", "ablation_config"],
                            columns="budget", values="sr_pct",
                            aggfunc="std").round(2)
print(table_std)

# ==================== Key insight ====================
print("\n" + "=" * 70)
print("KEY INSIGHT (role reversal)")
print("=" * 70)
for bench, tool in BENCH_TO_TOOL.items():
    print(f"\n{bench} ({tool}):")
    for cfg in CONFIGS:
        sub = df[(df["benchmark"] == bench) & (df["ablation_config"] == cfg)]
        if len(sub) > 0:
            b20  = sub[sub["budget"] == 20]["sr_pct"].mean()
            b60  = sub[sub["budget"] == 60]["sr_pct"].mean()
            b120 = sub[sub["budget"] == 120]["sr_pct"].mean()
            print(f"  {cfg:<12s}  B=20: {b20:5.1f}%   B=60: {b60:5.1f}%   B=120: {b120:5.1f}%")
