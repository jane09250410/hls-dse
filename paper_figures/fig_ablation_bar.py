"""Figure 4: Ablation bar chart for PA-DSE components.

Visualizes 8 ablation configs grouped by which component is active, with error bars.
Highlights the key findings: OFRS dominates, SCF is safety net, OFRS-skip breaks.

Input:
    ~/Desktop/hls/paper_figures/out/table_ablation_bambu.csv
    ~/Desktop/hls/paper_figures/out/table_ablation_dynamatic.csv

Output:
    ~/Desktop/hls/paper_figures/out/fig_ablation.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/Users/zhangxinyu/Desktop/hls/paper_figures/out")

# Load the precomputed ablation summary tables
bam = pd.read_csv(OUT / "table_ablation_bambu.csv").set_index("config")
dyn = pd.read_csv(OUT / "table_ablation_dynamatic.csv").set_index("config")

CONFIG_ORDER = [
    "no-filter", "SCF-only",
    "SCF+RPE", "SCF+OFRS", "SCF+DFRL",
    "DFRL-only",
    "SCF+RPE-reorder", "SCF+OFRS-skip",
]

# Color coding: highlight the "full" PA-DSE; flag the "unsafe" OFRS-skip
DEFAULT     = "#4E79A7"
FULL_PA_DSE = "#C1272D"  # red highlight
UNSAFE      = "#B71C1C"  # dark red for broken config
COLORS = {c: DEFAULT for c in CONFIG_ORDER}
COLORS["SCF+DFRL"]      = FULL_PA_DSE
COLORS["SCF+OFRS-skip"] = UNSAFE

# ==== Style ====
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
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

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)

def draw(ax, df, title):
    x = np.arange(len(CONFIG_ORDER))
    means = df.loc[CONFIG_ORDER, "sr_pct_mean"].values
    stds  = df.loc[CONFIG_ORDER, "sr_pct_std"].values
    cols  = [COLORS[c] for c in CONFIG_ORDER]
    bars = ax.bar(x, means, yerr=stds,
                  color=cols, edgecolor="black", linewidth=0.5,
                  capsize=3, error_kw={"elinewidth": 0.8, "ecolor": "#333"})

    # Annotate each bar with mean value above the error bar
    for i, (m, s, cfg) in enumerate(zip(means, stds, CONFIG_ORDER)):
        y = m + s + 2
        weight = "bold" if cfg in ("SCF+DFRL", "SCF+OFRS-skip") else "normal"
        color  = FULL_PA_DSE if cfg == "SCF+DFRL" else (UNSAFE if cfg == "SCF+OFRS-skip" else "#333")
        ax.text(i, y, f"{m:.1f}", ha="center", fontsize=8.5,
                color=color, fontweight=weight)

    ax.set_xticks(x)
    ax.set_xticklabels(CONFIG_ORDER, rotation=30, ha="right")
    ax.set_ylim(0, 110)
    ax.set_title(title, loc="left", pad=6)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

axL.set_ylabel("Success Rate (%)")
draw(axL, bam, "(a) Bambu (B=30, 24 runs/config)")
draw(axR, dyn, "(b) Dynamatic (B=30, 12 runs/config)")

# Legend
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=DEFAULT,     edgecolor="black", label="ablation variant"),
    Patch(facecolor=FULL_PA_DSE, edgecolor="black", label="PA-DSE full (SCF+DFRL)"),
    Patch(facecolor=UNSAFE,      edgecolor="black", label="evidence-hierarchy violation"),
]
fig.legend(handles=legend_handles, loc="upper center",
           bbox_to_anchor=(0.5, 1.02),
           ncol=3, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT / "fig_ablation.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_ablation.png", bbox_inches="tight", dpi=300)
print(f"✅ Saved: {OUT}/fig_ablation.pdf (and .png)")
plt.show()
