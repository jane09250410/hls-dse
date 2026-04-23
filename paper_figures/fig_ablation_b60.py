"""Figure: Ablation Study @ B=60.

Shows contribution of each PA-DSE layer (SCF, RPE, OFRS) through
8 ablation configurations.

Layout: Two panels side by side
  (a) Bambu (8 benchmarks × 8 configs × 3 perms = 192 runs, freshly done)
  (b) Dynamatic (2 benchmarks × 8 configs × 3 perms = 48 runs, from old data)

Each panel: bar chart with 8 configs, ordered to show progressive layer addition.
Error bars = std across benchmarks × perms.

Key story:
  - no-filter   : baseline (no SCF, no DFRL)
  - phago-only  : SCF alone (huge jump from no-filter)
  - phago+RPE   : SCF + RPE skip
  - phago+OFRS  : SCF + OFRS rank
  - phago+Full  : SCF + RPE + OFRS (best)
  - DFRL-only   : DFRL without SCF (shows SCF importance)
  - phago+RPE-reorder : SCF + RPE as reorder (ablation variant)
  - phago+OFRS-skip   : SCF + OFRS as skip (extreme, may crash)
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
df = pd.read_csv(ROOT / "master/ablation/run_summary.csv")
print(f"Loaded {len(df)} runs")

# Separate Bambu / Dynamatic
BAMBU_BENCH = ['matmul', 'vadd', 'fir', 'histogram', 'atax', 'bicg', 'gemm', 'gesummv']
DYN_BENCH   = ['gcd', 'matching']

bam_df = df[df['benchmark'].isin(BAMBU_BENCH)].copy()
dyn_df = df[df['benchmark'].isin(DYN_BENCH)].copy()

print(f"\nBambu:     {len(bam_df)} runs across {bam_df['benchmark'].nunique()} benchmarks")
print(f"Dynamatic: {len(dyn_df)} runs across {dyn_df['benchmark'].nunique()} benchmarks")

# ==================== Order & colors ====================
# Order ablation configs to tell a progressive story
CONFIG_ORDER = [
    "no-filter",          # baseline
    "phago-only",         # +SCF
    "phago+RPE",          # +SCF +RPE-skip
    "phago+OFRS",         # +SCF +OFRS-rank
    "phago+Full",         # +SCF +RPE +OFRS (proposed)
    "DFRL-only",          # -SCF +RPE +OFRS (ablation)
    "phago+RPE-reorder",  # +SCF +RPE-reorder variant
    "phago+OFRS-skip",    # +SCF +OFRS-skip variant (extreme)
]

CONFIG_LABELS = {
    "no-filter":          "No filter",
    "phago-only":         "SCF only",
    "phago+RPE":          "SCF+RPE",
    "phago+OFRS":         "SCF+OFRS",
    "phago+Full":         "SCF+RPE+OFRS\n(PA-DSE)",
    "DFRL-only":          "RPE+OFRS\n(no SCF)",
    "phago+RPE-reorder":  "SCF+RPE\n(reorder)",
    "phago+OFRS-skip":    "SCF+OFRS\n(skip)",
}

# Color coding: highlight PA-DSE (Full), dim variants, red for bad (OFRS-skip)
CONFIG_COLORS = {
    "no-filter":          "#8B8B8B",   # gray (baseline)
    "phago-only":         "#A0A0A0",   # light gray
    "phago+RPE":          "#F28E2B",   # orange (RPE alone)
    "phago+OFRS":         "#4E79A7",   # blue (OFRS alone)
    "phago+Full":         "#C1272D",   # red — PA-DSE (main)
    "DFRL-only":          "#B07AA1",   # purple (no SCF)
    "phago+RPE-reorder":  "#76B7B2",   # teal (variant)
    "phago+OFRS-skip":    "#E15759",   # light red (extreme)
}

# ==================== Figure ====================
fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 4.8),
                                gridspec_kw={"width_ratios": [1.3, 1.3]})

def plot_ablation(ax, df_tool, tool_name, show_legend_hint=True):
    """Grouped bar chart for one tool."""
    stats = df_tool.groupby("ablation_config")["sr_pct"].agg(["mean", "std", "count"])
    # Order
    ordered_configs = [c for c in CONFIG_ORDER if c in stats.index]
    means = [stats.loc[c, "mean"] for c in ordered_configs]
    stds  = [stats.loc[c, "std"]  for c in ordered_configs]
    counts = [stats.loc[c, "count"] for c in ordered_configs]
    colors = [CONFIG_COLORS[c] for c in ordered_configs]

    x = np.arange(len(ordered_configs))
    bars = ax.bar(x, means, yerr=stds,
                  color=colors, edgecolor="black", linewidth=0.5,
                  capsize=4, error_kw={"elinewidth": 0.8, "ecolor": "#333"})

    # Value labels above bars (esp. highlighted for Full)
    for i, (c, m, s) in enumerate(zip(ordered_configs, means, stds)):
        color = "#C1272D" if c == "phago+Full" else "#333"
        weight = "bold" if c == "phago+Full" else "normal"
        ax.text(i, m + s + 2, f"{m:.1f}",
                ha="center", fontsize=8.5, color=color, fontweight=weight)

    ax.set_xticks(x)
    ax.set_xticklabels([CONFIG_LABELS[c] for c in ordered_configs],
                       rotation=25, ha="right", fontsize=8)

    # Highlight "PA-DSE" label in red
    for lbl, cfg in zip(ax.get_xticklabels(), ordered_configs):
        if cfg == "phago+Full":
            lbl.set_color("#C1272D")
            lbl.set_fontweight("bold")

    ax.set_ylabel("Success Rate (%)")
    ax.set_title(f"({['a','b'][ax is axR]}) {tool_name}")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    # Dataset info
    n_bench = df_tool["benchmark"].nunique()
    n_total = len(df_tool)
    ax.text(0.99, 0.97,
            f"{n_bench} benchmarks, {n_total} runs, B=60",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="#555", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#ccc", alpha=0.92))

plot_ablation(axL, bam_df, "Bambu")
plot_ablation(axR, dyn_df, "Dynamatic")

plt.tight_layout()
plt.savefig(OUT / "fig_ablation_b60.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_ablation_b60.png", bbox_inches="tight", dpi=300)
print(f"\n✅ Saved: {OUT}/fig_ablation_b60.pdf")
plt.show()

# ==================== Paper table ====================
print("\n" + "=" * 70)
print("BAMBU ABLATION @ B=60")
print("=" * 70)
print(bam_df.groupby("ablation_config")["sr_pct"].agg(
    ["count", "mean", "std"]).round(2).sort_values("mean", ascending=False))

print("\n" + "=" * 70)
print("DYNAMATIC ABLATION @ B=60")
print("=" * 70)
print(dyn_df.groupby("ablation_config")["sr_pct"].agg(
    ["count", "mean", "std"]).round(2).sort_values("mean", ascending=False))

# ==================== Key insights ====================
print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)
for tool_name, dfx in [("Bambu", bam_df), ("Dynamatic", dyn_df)]:
    stats = dfx.groupby("ablation_config")["sr_pct"].mean()
    no_filter = stats.get("no-filter", None)
    phago = stats.get("phago-only", None)
    full = stats.get("phago+Full", None)
    dfrl = stats.get("DFRL-only", None)
    if no_filter is not None and phago is not None:
        print(f"\n{tool_name}:")
        print(f"  SCF contribution:       {phago - no_filter:+.1f} pp (no-filter → phago-only)")
        if full is not None:
            print(f"  DFRL contribution:      {full - phago:+.1f} pp (phago-only → phago+Full)")
        if dfrl is not None:
            print(f"  SCF necessity:          {full - dfrl:+.1f} pp (DFRL-only → phago+Full)")
