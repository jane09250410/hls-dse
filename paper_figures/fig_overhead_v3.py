"""Figure: Overhead Analysis — v7.

Panel (a) line chart on log scale showing orders-of-magnitude gap
between algorithmic overhead and HLS synthesis cost.
Panel (b) stacked bar showing layer composition within algo overhead.

IMPORTANT: Dynamatic numbers use ONLY the 4 evaluation benchmarks
(gcd, matching, binary_search, kernel_2mm). fir/histogram are excluded
because they achieve SR=100% for all methods and are not part of the
paper's Dynamatic evaluation. Including them would underestimate
synthesis time and inflate the algo:synth ratio.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("/Users/zhangxinyu/Desktop/hls/results")
OUT  = Path("/Users/zhangxinyu/Desktop/hls/paper_figures/out")
OUT.mkdir(parents=True, exist_ok=True)

# Benchmark filters
BAMBU_BENCHMARKS = [
    "matmul", "vadd", "fir", "histogram",
    "atax", "bicg", "gemm", "gesummv",
]
DYNAMATIC_BENCHMARKS = [
    "gcd", "matching", "binary_search", "kernel_2mm",
]

# ===== Journal palette =====
C_BAMBU  = "#3C5488"
C_DYNAM  = "#E64B35"
C_SCF    = "#3C5488"
C_RPE    = "#F39B7F"
C_OFRS   = "#00A087"
C_AXIS   = "#222222"
C_GRID   = "#CCCCCC"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10.5,
    "axes.titleweight": "normal",
    "axes.edgecolor": C_AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": C_AXIS,
    "xtick.color": C_AXIS,
    "ytick.color": C_AXIS,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

# ===== Load data =====
bam = pd.read_csv(ROOT / "rerun/bambu_pa_dse_perms/run_summary.csv")
dyn = pd.read_csv(ROOT / "rerun/dynamatic_pa_dse_perms/run_summary.csv")

# Filter to evaluation benchmark sets
before_b, before_d = len(bam), len(dyn)
bam = bam[bam["benchmark"].isin(BAMBU_BENCHMARKS)].copy()
dyn = dyn[dyn["benchmark"].isin(DYNAMATIC_BENCHMARKS)].copy()
print(f"[filter] Bambu: {before_b} -> {len(bam)} rows")
print(f"[filter] Dynamatic: {before_d} -> {len(dyn)} rows "
      f"(fir/histogram excluded, SR=100% for all methods)")

def compute_per_iter(df):
    scf_pi  = (df["overhead_phago_ms"] / df["total_evals"]).mean()
    rpe_pi  = (df["overhead_rpe_ms"]   / df["total_evals"]).mean()
    ofrs_pi = (df["overhead_ofrs_ms"]  / df["total_evals"]).mean()
    algo_pi = scf_pi + rpe_pi + ofrs_pi
    synth_pi = (df["total_wall_clock_s"] * 1000
                - (df["overhead_phago_ms"] + df["overhead_rpe_ms"] + df["overhead_ofrs_ms"])
                ).mean() / df["total_evals"].mean()
    return {"SCF": scf_pi, "RPE": rpe_pi, "OFRS": ofrs_pi,
            "algo_total": algo_pi, "synthesis": synth_pi}

bam_t = compute_per_iter(bam)
dyn_t = compute_per_iter(dyn)

# ===== Figure =====
fig = plt.figure(figsize=(12.5, 4.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1], wspace=0.22,
                      left=0.06, right=0.93, top=0.88, bottom=0.14)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])

# ------------------------- Panel (a): LINE CHART -------------------------
categories = ["SCF", "RPE", "OFRS", "Algo.\ntotal", "HLS\nsynth."]
bam_values = [bam_t["SCF"], bam_t["RPE"], bam_t["OFRS"], bam_t["algo_total"], bam_t["synthesis"]]
dyn_values = [dyn_t["SCF"], dyn_t["RPE"], dyn_t["OFRS"], dyn_t["algo_total"], dyn_t["synthesis"]]

x = np.arange(len(categories))

axA.plot(x, bam_values, marker="o", markersize=6.5, linewidth=1.5,
         color=C_BAMBU, label="Bambu",
         markerfacecolor=C_BAMBU, markeredgecolor="white", markeredgewidth=1.1,
         zorder=3)
axA.plot(x, dyn_values, marker="s", markersize=6.5, linewidth=1.5,
         color=C_DYNAM, label="Dynamatic",
         markerfacecolor=C_DYNAM, markeredgecolor="white", markeredgewidth=1.1,
         zorder=3)

axA.set_yscale("log")
axA.set_ylim(0.005, 30000)
axA.set_xlim(-0.35, len(categories) - 0.65)
axA.set_xticks(x)
axA.set_xticklabels(categories)
axA.set_ylabel("Time per iteration (ms)", labelpad=6)
axA.set_title("(a) Algorithmic overhead vs. HLS synthesis", loc="left", pad=8)

axA.grid(axis="y", which="major", color=C_GRID, lw=0.5, alpha=0.7)
axA.set_axisbelow(True)
for s in ("top", "right"):
    axA.spines[s].set_visible(False)

for xi, bam_val, dyn_val in zip(x, bam_values, dyn_values):
    if bam_val >= dyn_val:
        bam_dy, dyn_dy = 10, -13
    else:
        bam_dy, dyn_dy = -13, 10
    bam_lbl = f"{bam_val:.2f}" if bam_val < 100 else f"{bam_val:.0f}"
    dyn_lbl = f"{dyn_val:.2f}" if dyn_val < 100 else f"{dyn_val:.0f}"
    axA.annotate(bam_lbl, xy=(xi, bam_val), xytext=(0, bam_dy),
                 textcoords="offset points", ha="center",
                 fontsize=7.5, color=C_BAMBU)
    axA.annotate(dyn_lbl, xy=(xi, dyn_val), xytext=(0, dyn_dy),
                 textcoords="offset points", ha="center",
                 fontsize=7.5, color=C_DYNAM)

axA.legend(loc="upper left", handlelength=1.8, handletextpad=0.6,
           borderaxespad=0.3)

# ------------------------- Panel (b) -------------------------
tools = ["Bambu", "Dynamatic"]
data  = {"Bambu": bam_t, "Dynamatic": dyn_t}

y = np.arange(len(tools))
bar_h = 0.45

for i, tool in enumerate(tools):
    t   = data[tool]
    tot = t["algo_total"]
    scf_p  = t["SCF"]  / tot * 100
    rpe_p  = t["RPE"]  / tot * 100
    ofrs_p = t["OFRS"] / tot * 100

    axB.barh(y[i], scf_p,  left=0,             height=bar_h,
             color=C_SCF,  edgecolor="white", linewidth=0.8)
    axB.barh(y[i], rpe_p,  left=scf_p,         height=bar_h,
             color=C_RPE,  edgecolor="white", linewidth=0.8)
    axB.barh(y[i], ofrs_p, left=scf_p + rpe_p, height=bar_h,
             color=C_OFRS, edgecolor="white", linewidth=0.8)

    ofrs_center = scf_p + rpe_p + ofrs_p / 2
    axB.text(ofrs_center, y[i], f"OFRS  {ofrs_p:.1f}%",
             ha="center", va="center", color="white",
             fontsize=9, fontweight="bold")

    if tool == "Bambu":
        y_text     = y[i] - bar_h / 2 - 0.32
        y_bar_edge = y[i] - bar_h / 2
        va = "bottom"
    else:
        y_text     = y[i] + bar_h / 2 + 0.32
        y_bar_edge = y[i] + bar_h / 2
        va = "top"

    axB.annotate(f"SCF  {scf_p:.1f}%",
                 xy=(scf_p / 2, y_bar_edge),
                 xytext=(-2, y_text),
                 ha="right", va=va, fontsize=8, color=C_SCF,
                 arrowprops=dict(arrowstyle="-", color=C_SCF, lw=0.6,
                                 shrinkA=0, shrinkB=0))
    axB.annotate(f"RPE  {rpe_p:.1f}%",
                 xy=(scf_p + rpe_p / 2, y_bar_edge),
                 xytext=(scf_p + rpe_p / 2 + 3, y_text),
                 ha="left", va=va, fontsize=8, color=C_RPE,
                 arrowprops=dict(arrowstyle="-", color=C_RPE, lw=0.6,
                                 shrinkA=0, shrinkB=0))

    axB.text(101.5, y[i], f"total {tot:.2f} ms/iter",
             ha="left", va="center", fontsize=8.5,
             color="#555", style="italic")

axB.set_yticks(y)
axB.set_yticklabels(tools)
axB.set_ylim(1.8, -0.8)
axB.set_xlim(0, 130)
axB.set_xticks([0, 20, 40, 60, 80, 100])
axB.set_xticklabels(["0", "20", "40", "60", "80", "100"])
axB.set_xlabel("Share of algorithmic overhead (%)", labelpad=6)
axB.set_title("(b) Layer composition within algorithmic overhead",
              loc="left", pad=8)

axB.grid(axis="x", which="major", color=C_GRID, lw=0.5, alpha=0.7)
axB.set_axisbelow(True)
for s in ("top", "right", "left"):
    axB.spines[s].set_visible(False)
axB.tick_params(axis="y", length=0, pad=4)

plt.savefig(OUT / "fig_overhead.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_overhead.png", bbox_inches="tight", dpi=300)
print(f"\nSaved: {OUT}/fig_overhead.pdf")
plt.show()

# ===== Summary =====
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Bambu     : PA-DSE overhead {bam_t['algo_total']:.2f}ms | synthesis {bam_t['synthesis']:.0f}ms | ratio 1:{bam_t['synthesis']/bam_t['algo_total']:.0f}")
print(f"Dynamatic : PA-DSE overhead {dyn_t['algo_total']:.2f}ms | synthesis {dyn_t['synthesis']:.0f}ms | ratio 1:{dyn_t['synthesis']/dyn_t['algo_total']:.0f}")
