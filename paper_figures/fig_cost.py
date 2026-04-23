"""Figure 6: Cost comparison — wasted calls + TTFF (user-visible cost story).

Two-panel bar chart:
  (a) Wasted synthesis calls per method (Bambu + Dynamatic side by side)
  (b) Time-to-first-feasible (TTFF) per method

Highlights PA-DSE's cost advantage that SR alone doesn't reveal.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/Users/zhangxinyu/Desktop/hls/paper_figures/out")
bam = pd.read_csv(OUT / "table_main_bambu.csv").set_index("method")
dyn = pd.read_csv(OUT / "table_main_dynamatic.csv").set_index("method")

ORDER = ["Random", "FilteredRandom", "SA", "GA", "GP-BO", "RF", "PA-DSE"]
COLORS = {
    "Random":         "#B0B0B0",
    "FilteredRandom": "#808080",
    "SA":             "#F4A261",
    "GA":             "#E9C46A",
    "GP-BO":          "#2A9D8F",
    "RF":             "#264653",
    "PA-DSE":         "#C1272D",
}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times"],
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.2))

# ============ (a) Wasted calls ============
x = np.arange(len(ORDER))
w = 0.38
wa_bam = bam.loc[ORDER, "wasted_calls_mean"].values
wa_bam_s = bam.loc[ORDER, "wasted_calls_std"].values
wa_dyn = dyn.loc[ORDER, "wasted_calls_mean"].values
wa_dyn_s = dyn.loc[ORDER, "wasted_calls_std"].values

axL.bar(x - w/2, wa_bam, w, yerr=wa_bam_s,
        color=[COLORS[m] for m in ORDER], edgecolor="black", linewidth=0.5,
        label="Bambu (B=60)", capsize=2, error_kw={"elinewidth": 0.6})
axL.bar(x + w/2, wa_dyn, w, yerr=wa_dyn_s,
        color=[COLORS[m] for m in ORDER], edgecolor="black", linewidth=0.5,
        hatch="//", label="Dynamatic (B=30)", capsize=2, error_kw={"elinewidth": 0.6})

for i, m in enumerate(ORDER):
    fw = "bold" if m == "PA-DSE" else "normal"
    c  = "#C1272D" if m == "PA-DSE" else "#333"
    axL.text(i - w/2, wa_bam[i] + wa_bam_s[i] + 1, f"{wa_bam[i]:.1f}",
             ha="center", fontsize=8, color=c, fontweight=fw)

axL.set_xticks(x)
axL.set_xticklabels(ORDER, rotation=25, ha="right")
axL.set_ylabel("Wasted synthesis calls")
axL.set_title("(a) Wasted synthesis calls (lower is better)", loc="left")
axL.legend(loc="upper right", frameon=False)
axL.grid(axis="y", alpha=0.3)
axL.set_axisbelow(True)

# ============ (b) TTFF ============
ttff_bam = bam.loc[ORDER, "ttff_s_mean"].values
ttff_bam_s = bam.loc[ORDER, "ttff_s_std"].values
ttff_dyn = dyn.loc[ORDER, "ttff_s_mean"].values
ttff_dyn_s = dyn.loc[ORDER, "ttff_s_std"].values

# TTFF can't be negative — clip lower error at 0.
# Use asymmetric error bars: [min(std, mean), std].
def clip_err(mean, std):
    return np.stack([np.minimum(std, mean), std])

axR.bar(x - w/2, ttff_bam, w, yerr=clip_err(ttff_bam, ttff_bam_s),
        color=[COLORS[m] for m in ORDER], edgecolor="black", linewidth=0.5,
        label="Bambu (B=60)", capsize=2, error_kw={"elinewidth": 0.6})
axR.bar(x + w/2, ttff_dyn, w, yerr=clip_err(ttff_dyn, ttff_dyn_s),
        color=[COLORS[m] for m in ORDER], edgecolor="black", linewidth=0.5,
        hatch="//", label="Dynamatic (B=30)", capsize=2, error_kw={"elinewidth": 0.6})

for i, m in enumerate(ORDER):
    fw = "bold" if m == "PA-DSE" else "normal"
    c  = "#C1272D" if m == "PA-DSE" else "#333"
    # label Bambu value above Bambu bar (left), Dynamatic value above Dynamatic bar (right)
    axR.text(i - w/2, ttff_bam[i] + min(ttff_bam_s[i], ttff_bam[i]) + 2,
             f"{ttff_bam[i]:.1f}",
             ha="center", fontsize=8, color=c, fontweight=fw)
    axR.text(i + w/2, ttff_dyn[i] + min(ttff_dyn_s[i], ttff_dyn[i]) + 2,
             f"{ttff_dyn[i]:.1f}",
             ha="center", fontsize=8, color=c, fontweight=fw)

axR.set_xticks(x)
axR.set_xticklabels(ORDER, rotation=25, ha="right")
axR.set_ylabel("Time to first feasible (s)")
axR.set_ylim(bottom=0)                          # enforce physical floor
axR.set_title("(b) Time to first feasible (lower is better)", loc="left")
axR.legend(loc="upper right", frameon=False)
axR.grid(axis="y", alpha=0.3)
axR.set_axisbelow(True)

plt.tight_layout()
plt.savefig(OUT / "fig_cost.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_cost.png", bbox_inches="tight", dpi=300)
print(f"✅ Saved: {OUT}/fig_cost.pdf")
plt.show()
