"""Figure 8 (v4): QoR coverage classification.

Each unique (area, latency) point is classified by who found it:
  • Shared       — gray — found by PA-DSE AND at least one baseline
  • PA-DSE only  — red  — only PA-DSE found it
  • Baseline only — blue — no PA-DSE run found it, but some baseline did

Global Pareto front overlaid in black. This is the HONEST version: it shows
that PA-DSE's coverage is (near-) identical to the union of all baselines,
while reaching all Pareto-optimal points.
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


def load_successes(main_path, perms_path):
    main = pd.read_csv(ROOT / main_path)
    perms = pd.read_csv(ROOT / perms_path)
    main = main[~main["strategy"].str.contains("PA-DSE")].copy()
    main["strategy"] = main["strategy"].map(RENAME).fillna(main["strategy"])
    perms = perms.copy()
    perms["strategy"] = "PA-DSE"
    df = pd.concat([main, perms], ignore_index=True)

    # 4-bench Dynamatic filter (paper §IV.A): exclude fir/histogram on Dynamatic
    if "dynamatic" in main_path.lower():
        from plot_style import DYNAMATIC_BENCHMARKS
        before = len(df)
        df = df[df["benchmark"].isin(DYNAMATIC_BENCHMARKS)].copy()
        print(f"  [filter] Dynamatic: {before} -> {len(df)} rows")

    if "success" in df.columns:
        df = df[df["success"] == True].copy()
    elif "outcome" in df.columns:
        df = df[df["outcome"] == "success"].copy()
    df = df.dropna(subset=["area", "latency"])
    df = df[(df["area"] > 0) & (df["latency"] > 0)].copy()
    return df


def classify_points(df):
    df = df.copy()
    df["a_r"] = df["area"].round(6)
    df["l_r"] = df["latency"].round(6)
    grouped = (df.groupby(["a_r", "l_r"])["strategy"]
                 .apply(set).reset_index())
    grouped["has_padse"] = grouped["strategy"].apply(lambda s: "PA-DSE" in s)
    grouped["has_base"]  = grouped["strategy"].apply(
        lambda s: any(m != "PA-DSE" for m in s))

    def cls(row):
        if row["has_padse"] and row["has_base"]: return "shared"
        if row["has_padse"]:                     return "padse_only"
        return "baseline_only"
    grouped["class"] = grouped.apply(cls, axis=1)
    return grouped.rename(columns={"a_r": "area", "l_r": "latency"})


def pareto_front(points):
    if len(points) == 0:
        return np.array([], dtype=int)
    idx = np.argsort(points[:, 0])
    sorted_pts = points[idx]
    pareto = [0]
    best_lat = sorted_pts[0, 1]
    for i in range(1, len(sorted_pts)):
        if sorted_pts[i, 1] < best_lat:
            pareto.append(i)
            best_lat = sorted_pts[i, 1]
    return idx[pareto]


plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times"],
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 150, "savefig.dpi": 300,
})

CLASS_STYLE = {
    "shared":        dict(color="#A8B0B3", marker="o", s=22, alpha=0.55,
                          edgecolor="white", linewidth=0.3),
    "padse_only":    dict(color="#C1272D", marker="o", s=38, alpha=0.90,
                          edgecolor="white", linewidth=0.4),
    "baseline_only": dict(color="#1F4E79", marker="s", s=32, alpha=0.85,
                          edgecolor="white", linewidth=0.4),
}
CLASS_LABELS = {
    "shared":        "Shared (PA-DSE and baselines)",
    "padse_only":    "PA-DSE only",
    "baseline_only": "Baselines only",
}
DRAW_ORDER = ["shared", "baseline_only", "padse_only"]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8))


def plot_qor(ax, classified, all_pts, title, tool_name):
    # Draw points
    for cls in DRAW_ORDER:
        sub = classified[classified["class"] == cls]
        style = dict(CLASS_STYLE[cls])
        style["label"] = f"{CLASS_LABELS[cls]} (n={len(sub)})"
        if len(sub) == 0:
            continue
        ax.scatter(sub["area"], sub["latency"], **style)

    # Global Pareto front
    pareto_count = 0
    pareto_on_padse = 0
    if len(all_pts) > 0:
        pf_idx = pareto_front(all_pts)
        pf = all_pts[pf_idx]
        pf = pf[np.argsort(pf[:, 0])]
        pareto_count = len(pf)

        # check Pareto points' class (is PA-DSE in their finder set?)
        pf_set = set(map(tuple, np.round(pf, 6)))
        for _, row in classified.iterrows():
            pt = (row["area"], row["latency"])
            if pt in pf_set and row["has_padse"]:
                pareto_on_padse += 1

        ax.plot(pf[:, 0], pf[:, 1], "--",
                color="black", linewidth=1.8, zorder=20,
                label=f"Pareto front (n={pareto_count})")
        ax.scatter(pf[:, 0], pf[:, 1],
                   s=150, marker="D",
                   facecolor="none", edgecolor="black",
                   linewidth=1.5, zorder=21)

    # Annotation box: Pareto coverage stat
    annot = (f"PA-DSE Pareto coverage:\n"
             f"{pareto_on_padse} / {pareto_count} vertices")
    ax.text(0.98, 0.03, annot,
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8.5, family="serif",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                      edgecolor="#C1272D", linewidth=1.0, alpha=0.95))

    ax.set_xlabel("Area")
    ax.set_ylabel("Latency")
    ax.set_title(title, loc="left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.set_axisbelow(True)


print("Loading Bambu eval logs...")
bam = load_successes(
    "master/bambu_main/eval_log.csv",
    "rerun/bambu_pa_dse_perms/eval_log.csv",
)
bam_cls = classify_points(bam)
print("Bambu classification:")
print(bam_cls["class"].value_counts().to_string())

all_bam = bam[["area", "latency"]].drop_duplicates().values
plot_qor(axL, bam_cls, all_bam, "(a) Bambu QoR coverage", "bambu")
axL.legend(loc="upper left", fontsize=8.5, frameon=True,
           framealpha=0.92, labelspacing=0.4)

print("\nLoading Dynamatic eval logs...")
dyn = load_successes(
    "master/dynamatic_main/eval_log.csv",
    "rerun/dynamatic_pa_dse_perms/eval_log.csv",
)
dyn_cls = classify_points(dyn)
print("Dynamatic classification:")
print(dyn_cls["class"].value_counts().to_string())

all_dyn = dyn[["area", "latency"]].drop_duplicates().values
plot_qor(axR, dyn_cls, all_dyn, "(b) Dynamatic QoR coverage", "dynamatic")
axR.legend(loc="upper left", fontsize=8.5, frameon=True,
           framealpha=0.92, labelspacing=0.4)

plt.tight_layout()
plt.savefig(OUT / "fig_qor.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_qor.png", bbox_inches="tight", dpi=300)
print(f"\n✅ Saved: {OUT}/fig_qor.pdf")
plt.show()
