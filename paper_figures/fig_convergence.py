"""Figure 7: Convergence curves — cumulative feasibles over eval_step.

Reads eval_log.csv and computes cumulative successful count per
(method, benchmark, run). Then averages across benchmarks/runs and plots
one curve per method with shaded confidence band.

Shows HOW methods reach their final SR: PA-DSE is faster/more monotone.

Input:
    master/bambu_main/eval_log.csv
    rerun/bambu_pa_dse_perms/eval_log.csv
    master/dynamatic_main/eval_log.csv
    rerun/dynamatic_pa_dse_perms/eval_log.csv

Output:
    fig_convergence.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/Users/zhangxinyu/Desktop/hls/results")
OUT  = Path("/Users/zhangxinyu/Desktop/hls/paper_figures/out")

RENAME = {
    "Random": "Random",
    "Filtered_Random": "FilteredRandom",
    "SimulatedAnnealing": "SA",
    "GeneticAlgorithm": "GA",
    "GP-BO": "GP-BO",
    "RF_Classifier": "RF",
}
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


def load_and_process(main_path, perms_path, max_step):
    main = pd.read_csv(ROOT / main_path)
    perms = pd.read_csv(ROOT / perms_path)
    main = main[~main["strategy"].str.contains("PA-DSE")].copy()
    main["strategy"] = main["strategy"].map(RENAME).fillna(main["strategy"])
    perms = perms.copy()
    perms["strategy"] = "PA-DSE"
    df = pd.concat([main, perms], ignore_index=True)

    # For each (strategy, run_id), compute cumulative feasibles vs eval_step
    # eval_log has one row per synthesis call with 'success' boolean
    df["success"] = df["success"].astype(int) if "success" in df.columns else (df["outcome"] == "success").astype(int)

    # Group and compute cumsum per run
    curves = {}  # strategy -> list of per-run arrays
    for strat in ORDER:
        sub = df[df["strategy"] == strat]
        if len(sub) == 0:
            continue
        runs = []
        for (run_id, bench), g in sub.groupby(["run_id", "benchmark"]):
            g = g.sort_values("eval_step")
            cum = g["success"].cumsum().values
            if len(cum) < max_step:
                cum = np.concatenate([cum, np.full(max_step - len(cum), cum[-1] if len(cum) else 0)])
            runs.append(cum[:max_step])
        if runs:
            curves[strat] = np.stack(runs)
    return curves


plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times"],
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.2))


def plot_curves(ax, curves, budget, title):
    x = np.arange(1, budget + 1)
    for strat in ORDER:
        if strat not in curves:
            continue
        arr = curves[strat]
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        lw = 2.2 if strat == "PA-DSE" else 1.3
        alpha_fill = 0.25 if strat == "PA-DSE" else 0.12
        ax.plot(x, mean, color=COLORS[strat], linewidth=lw, label=strat)
        ax.fill_between(x, mean - std, mean + std, color=COLORS[strat], alpha=alpha_fill, lw=0)
    ax.set_xlabel("Evaluation step")
    ax.set_ylabel("Cumulative feasible configurations")
    ax.set_title(title, loc="left")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)


print("Loading Bambu eval logs (this may take ~10s)...")
bam_curves = load_and_process(
    "master/bambu_main/eval_log.csv",
    "rerun/bambu_pa_dse_perms/eval_log.csv",
    max_step=60,
)
plot_curves(axL, bam_curves, 60, "(a) Bambu convergence (B=60)")
axL.legend(loc="upper left", frameon=False, ncol=2, fontsize=8.5)

print("Loading Dynamatic eval logs...")
dyn_curves = load_and_process(
    "master/dynamatic_main/eval_log.csv",
    "rerun/dynamatic_pa_dse_perms/eval_log.csv",
    max_step=30,
)
plot_curves(axR, dyn_curves, 30, "(b) Dynamatic convergence (B=30, 4 benchmarks)")
axR.legend(loc="upper left", frameon=False, ncol=2, fontsize=8.5)

plt.tight_layout()
plt.savefig(OUT / "fig_convergence.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_convergence.png", bbox_inches="tight", dpi=300)
print(f"✅ Saved: {OUT}/fig_convergence.pdf")
plt.show()
