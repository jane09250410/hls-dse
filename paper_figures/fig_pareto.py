"""
fig_pareto.py
=============
Pareto trade-off scatter plot: best_latency vs SR for each method on
Bambu (a) and Dynamatic (b). Highlights the production-feasible region
(SR >= 85%) where PA-DSE+QAT+QSD attains the Pareto frontier.

Directly responds to the reader's likely question: "Why are some
PA-DSE QoR numbers lower than Random/GA/SA?" Answer: those methods
trade 25-39pp of SR for marginal latency, leaving them outside any
production-feasible operating regime.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"
OUT  = Path(__file__).resolve().parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times"],
    "font.size": 10, "axes.labelsize": 10.5, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "figure.dpi": 150, "savefig.dpi": 300,
})

RENAME = {
    "Random": "Random",
    "Filtered_Random": "FilteredRandom",
    "SimulatedAnnealing": "SA",
    "GeneticAlgorithm": "GA",
    "GP-BO": "GP-BO",
    "RF_Classifier": "RF",
}
COLORS = {
    "Random":         "#8B8B8B",
    "FilteredRandom": "#A0A0A0",
    "SA":             "#4E79A7",
    "GA":             "#59A14F",
    "GP-BO":          "#F28E2B",
    "RF":             "#E15759",
    "PA-DSE":         "#C1272D",
    "PA-DSE+QAT+QSD": "#7B0E12",
}
MARKERS = {
    "Random":         "o",
    "FilteredRandom": "s",
    "SA":             "v",
    "GA":             "^",
    "GP-BO":          "D",
    "RF":             "P",
    "PA-DSE":         "*",
    "PA-DSE+QAT+QSD": "*",
}

DYNAMATIC_BENCHMARKS = ["gcd", "matching", "binary_search", "kernel_2mm"]
BAMBU_BENCHMARKS = ["matmul", "vadd", "fir", "histogram",
                    "atax", "bicg", "gemm", "gesummv"]


def load(main_path, perms_path, qat_qsd_path, bench_filter):
    main = pd.read_csv(ROOT / main_path)
    main = main[main["benchmark"].isin(bench_filter)].copy()
    main = main[~main["strategy"].str.contains("PA-DSE")].copy()
    main["strategy"] = main["strategy"].map(RENAME).fillna(main["strategy"])

    perms = pd.read_csv(ROOT / perms_path)
    perms = perms[perms["benchmark"].isin(bench_filter)].copy()
    perms["strategy"] = "PA-DSE"

    frames = [main, perms]
    if (ROOT / qat_qsd_path).exists():
        qat = pd.read_csv(ROOT / qat_qsd_path)
        qat = qat[qat["benchmark"].isin(bench_filter)].copy()
        if len(qat) > 0:
            qat["strategy"] = "PA-DSE+QAT+QSD"
            frames.append(qat)

    return pd.concat(frames, ignore_index=True)


def aggregate(df):
    out = (df.groupby("strategy")
             .agg(SR=("sr_pct", "mean"),
                  best_lat=("best_latency", "mean"))
             .reset_index())
    return out


def plot_panel(ax, df, title, threshold=85, lat_pad_frac=0.07):
    """Scatter SR (x) vs best_latency (y). Shade SR >= threshold region."""
    df = df.dropna(subset=["best_lat"])
    # Determine y-axis range
    lat_lo = df["best_lat"].min()
    lat_hi = df["best_lat"].max()
    pad = (lat_hi - lat_lo) * lat_pad_frac
    y_min = lat_lo - pad
    y_max = lat_hi + pad
    # Determine x-axis range
    x_min = max(0, df["SR"].min() - 5)
    x_max = 100

    # Production-feasible shaded region
    ax.axvspan(threshold, x_max, alpha=0.10, color="#1a9850",
               zorder=0, label=f"Production-feasible (SR ≥ {threshold}%)")
    ax.axvline(threshold, color="#1a9850", linestyle="--", alpha=0.5, lw=0.9, zorder=1)

    # Plot each method
    methods = ["Random", "FilteredRandom", "SA", "GA", "GP-BO", "RF",
               "PA-DSE", "PA-DSE+QAT+QSD"]
    for m in methods:
        row = df[df["strategy"] == m]
        if len(row) == 0:
            continue
        sr = row["SR"].iloc[0]
        bl = row["best_lat"].iloc[0]

        if m in ("PA-DSE", "PA-DSE+QAT+QSD"):
            sz = 320
            ec = "black"; ew = 1.0
        else:
            sz = 110
            ec = "black"; ew = 0.5

        ax.scatter(sr, bl, s=sz, c=COLORS[m], marker=MARKERS[m],
                   edgecolors=ec, linewidths=ew, label=m, zorder=5)

    # Annotate Random with text label nearby (no arrow, no positioning issues)
    rand = df[df["strategy"] == "Random"]
    if len(rand):
        rsr = rand["SR"].iloc[0]
        rbl = rand["best_lat"].iloc[0]
        # Place text just to the right of Random marker
        ax.text(rsr + 2.5, rbl,
                f"← Random\n(SR {rsr:.0f}%,\n not deployable)",
                fontsize=7.5, color="#555", va="center", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                         edgecolor="#bbb", alpha=0.85))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Inverted: lower latency = better, so put best at top
    ax.set_xlabel("Success Rate (%)")
    ax.set_ylabel("Best Latency (cycles, lower = better)")
    ax.set_title(title, loc="left", pad=6)
    ax.grid(True, alpha=0.3, linestyle="--", lw=0.5)
    return ax


fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4.6))

print("Loading Bambu data...")
bam_df = load(
    "master/bambu_main/run_summary.csv",
    "rerun/bambu_pa_dse_perms/run_summary.csv",
    "qse/bambu_main/run_summary.csv",
    BAMBU_BENCHMARKS,
)
bam_agg = aggregate(bam_df)
print(bam_agg.to_string(index=False))
plot_panel(axL, bam_agg, "(a) Bambu (B=60)", threshold=85)
axL.legend(loc="lower left", frameon=True, framealpha=0.95,
           ncol=2, fontsize=8, handletextpad=0.4, columnspacing=0.9)

print("\nLoading Dynamatic data...")
dyn_df = load(
    "master/dynamatic_main/run_summary.csv",
    "rerun/dynamatic_pa_dse_perms/run_summary.csv",
    "qse/dynamatic_main/run_summary.csv",
    DYNAMATIC_BENCHMARKS,
)
dyn_agg = aggregate(dyn_df)
print(dyn_agg.to_string(index=False))
plot_panel(axR, dyn_agg, "(b) Dynamatic (B=30)", threshold=85)
# Don't repeat legend on right panel

plt.tight_layout()
plt.savefig(OUT / "fig_pareto.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_pareto.png", bbox_inches="tight", dpi=300)
print(f"\n✅ Saved: {OUT}/fig_pareto.pdf")
