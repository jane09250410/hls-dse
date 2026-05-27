"""Figure 9: Per-benchmark SR heatmap — method × benchmark grid.

Shows PA-DSE wins or ties on every benchmark, revealing consistency.
Uses run_summary.csv (simpler, no eval_log needed).

Output: fig_perbench_heatmap.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "results"
OUT  = Path(__file__).resolve().parent / "out"

RENAME = {
    "Random": "Random",
    "Filtered_Random": "FilteredRandom",
    "SimulatedAnnealing": "SA",
    "GeneticAlgorithm": "GA",
    "GP-BO": "GP-BO",
    "RF_Classifier": "RF",
}
ORDER = ["Random", "FilteredRandom", "SA", "GA", "GP-BO", "RF", "PA-DSE", "PA-DSE+QAT+QSD"]


def load_main(main_path, perms_path, qat_qsd_path=None):
    main = pd.read_csv(ROOT / main_path)
    perms = pd.read_csv(ROOT / perms_path)
    main = main[~main["strategy"].str.contains("PA-DSE")].copy()
    main["strategy"] = main["strategy"].map(RENAME).fillna(main["strategy"])
    perms = perms.copy()
    perms["strategy"] = "PA-DSE"
    frames = [main, perms]
    if qat_qsd_path is not None and (ROOT / qat_qsd_path).exists():
        qat = pd.read_csv(ROOT / qat_qsd_path).copy()
        qat["strategy"] = "PA-DSE+QAT+QSD"
        frames.append(qat)
    return pd.concat(frames, ignore_index=True)


plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times"],
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
})

# Heatmap color: low red, mid yellow, high green
cmap = LinearSegmentedColormap.from_list(
    "sr", ["#d73027", "#fdae61", "#fee08b", "#d9ef8b", "#a6d96a", "#1a9850"])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 4.4),
                                gridspec_kw={"width_ratios": [1.0, 1.0]})


def plot_heatmap(ax, df, bench_order, title, show_ylabels=True):
    # Compute mean SR per (strategy, benchmark)
    piv = (df.groupby(["strategy", "benchmark"])["sr_pct"]
             .mean()
             .unstack("benchmark"))
    # Filter ORDER to only methods present in this df
    order_present = [m for m in ORDER if m in piv.index]
    piv = piv.loc[order_present, bench_order]

    im = ax.imshow(piv.values, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    # annotate cells
    for i in range(len(order_present)):
        for j in range(len(bench_order)):
            val = piv.values[i, j]
            if np.isnan(val):
                txt = "n/a"
                color = "#555"
            else:
                txt = f"{val:.1f}"
                # black text on lighter cells, white on darker
                color = "white" if val < 30 or val > 70 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(np.arange(len(bench_order)))
    ax.set_xticklabels(bench_order, rotation=20, ha="right")
    if show_ylabels:
        ax.set_yticks(np.arange(len(order_present)))
        ax.set_yticklabels(order_present)
    else:
        ax.set_yticks([])
    ax.set_title(title, loc="left", pad=6)
    # grid
    ax.set_xticks(np.arange(-0.5, len(bench_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(order_present), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1)
    ax.tick_params(which="minor", length=0)

    # Bold PA-DSE / PA-DSE+QAT+QSD row labels
    if show_ylabels:
        for lbl in ax.get_yticklabels():
            if lbl.get_text() == "PA-DSE":
                lbl.set_color("#C1272D"); lbl.set_fontweight("bold")
            elif lbl.get_text() == "PA-DSE+QAT+QSD":
                lbl.set_color("#7B0E12"); lbl.set_fontweight("bold")
    return im


print("Loading Bambu data...")
bam_df = load_main(
    "master/bambu_main/run_summary.csv",
    "rerun/bambu_pa_dse_perms/run_summary.csv",
    qat_qsd_path="qse/bambu_main/run_summary.csv",
)
BAMBU_BENCHES = ["matmul", "vadd", "fir", "histogram",
                 "atax", "bicg", "gemm", "gesummv"]
im1 = plot_heatmap(axL, bam_df, BAMBU_BENCHES, "(a) Bambu per-benchmark SR (%)",
                    show_ylabels=True)

print("Loading Dynamatic data...")
dyn_df = load_main(
    "master/dynamatic_main/run_summary.csv",
    "rerun/dynamatic_pa_dse_perms/run_summary.csv",
    qat_qsd_path="qse/dynamatic_main/run_summary.csv",
)
DYN_BENCHES = ["matmul", "vadd", "fir", "histogram",
               "atax", "bicg", "gemm", "gesummv"]
im2 = plot_heatmap(axR, dyn_df, DYN_BENCHES, "(b) Dynamatic per-benchmark SR (%)",
                    show_ylabels=False)

# Shared colorbar — independent axes to avoid squeezing subplots
fig.subplots_adjust(right=0.90)
cax = fig.add_axes([0.92, 0.18, 0.015, 0.65])  # [left, bottom, width, height]
cbar = fig.colorbar(im2, cax=cax)
cbar.set_label("Success Rate (%)")

plt.savefig(OUT / "fig_perbench_heatmap.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_perbench_heatmap.png", bbox_inches="tight", dpi=300)
print(f"✅ Saved: {OUT}/fig_perbench_heatmap.pdf")
plt.show()
