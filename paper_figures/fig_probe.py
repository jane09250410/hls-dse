"""Figure: Probe Sensitivity.

Shows effect of p_probe on SR for two benchmarks:
  - vadd (Bambu)
  - gcd  (Dynamatic)

p_probe values tested: 0.0, 0.02, 0.05, 0.1, 0.2

Key insight:
  - p_probe = 0   : signature lock-in, possible false skip risk
  - p_probe = 0.05: sweet spot (default in paper)
  - p_probe = 0.2 : too much probe, wastes budget on already-skipped configs

Input:  master/probe/run_summary.csv
Output: paper_figures/out/fig_probe.{pdf,png}
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

df = pd.read_csv(ROOT / "master/probe/run_summary.csv")
print(f"Loaded {len(df)} runs")
print(df.groupby(["benchmark", "p_probe"])["sr_pct"].agg(["mean", "std"]).round(2))

BENCH_TO_TOOL = {"vadd": "Bambu", "gcd": "Dynamatic"}
COLORS = {"vadd": "#4E79A7", "gcd": "#E15759"}

fig, ax = plt.subplots(figsize=(6.5, 4.2))

for bench, tool in BENCH_TO_TOOL.items():
    sub = df[df["benchmark"] == bench]
    grouped = sub.groupby("p_probe")["sr_pct"].agg(["mean", "std"]).reset_index()

    ax.errorbar(grouped["p_probe"], grouped["mean"],
                yerr=grouped["std"],
                color=COLORS[bench], marker="o", markersize=7,
                linewidth=1.6, capsize=4, capthick=1.0,
                markeredgecolor="black", markeredgewidth=0.5,
                label=f"{bench} ({tool})")

# Mark default p=0.05
ax.axvline(x=0.05, color="#888", linestyle=":", linewidth=0.8, alpha=0.6)
ax.text(0.05, 60, "p=0.05\n(default)", fontsize=8, color="#666",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                  edgecolor="none", alpha=0.9))

ax.set_xlabel("Probe rate $p_{\\mathrm{probe}}$")
ax.set_ylabel("Success Rate (%)")
ax.set_title("Probe Sensitivity")
ax.set_xticks([0.0, 0.02, 0.05, 0.1, 0.2])
ax.set_xticklabels(["0.00", "0.02", "0.05", "0.10", "0.20"])
ax.set_ylim(50, 105)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", framealpha=0.95)

plt.tight_layout()
plt.savefig(OUT / "fig_probe.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_probe.png", bbox_inches="tight", dpi=300)
print(f"✅ Saved: {OUT}/fig_probe.pdf")
plt.show()

print("\nProbe sensitivity summary:")
print(df.pivot_table(index="benchmark", columns="p_probe",
                      values="sr_pct", aggfunc="mean").round(1))
