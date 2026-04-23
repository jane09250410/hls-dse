"""Figure: Hyperparameter Sensitivity (θ, τ, n_min).

Three-panel figure showing SR response to each PA-DSE hyperparameter.

Parameters and their meanings:
  θ (theta):     Signature confidence threshold (default 0.8)
  τ (tau):       Minimum support for signature activation (default 2)
  n_min:         Minimum observations before stabilizing signature (default 5)

Shows the method is robust to reasonable perturbations of defaults.

Input:  master/sensitivity/run_summary.csv
Output: paper_figures/out/fig_sensitivity.{pdf,png}
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
df = pd.read_csv(ROOT / "master/sensitivity/run_summary.csv")
print(f"Loaded {len(df)} runs")
print(df.columns.tolist())
print(df.head(3))

# Sensitivity runs typically vary one of theta/tau/n_min at a time
# The default values are theta=0.8, tau=2, n_min=5
# A specific run is identified by which param differs from default.

DEFAULTS = {"theta": 0.8, "tau": 2, "n_min": 5}

def identify_param_swept(row):
    """Determine which parameter is being swept in this row."""
    if row["theta"] != DEFAULTS["theta"]:
        return "theta", row["theta"]
    if row["tau"] != DEFAULTS["tau"]:
        return "tau", row["tau"]
    if row["n_min"] != DEFAULTS["n_min"]:
        return "n_min", row["n_min"]
    # If all default, could be the base case (count toward any sweep)
    return None, None

# Categorize
df[["swept_param", "swept_val"]] = df.apply(
    lambda r: pd.Series(identify_param_swept(r)), axis=1)

# For the default case (swept_param=None), include it in ALL three sweeps
# at the default value.
default_rows = df[df["swept_param"].isna()]
print(f"\nDefault (all-default) rows: {len(default_rows)}")

# Build per-parameter dataframes
def build_param_df(param_name):
    p_rows = df[df["swept_param"] == param_name].copy()
    # Add default-value rows as the baseline
    baseline = default_rows.copy()
    baseline["swept_param"] = param_name
    baseline["swept_val"] = DEFAULTS[param_name]
    return pd.concat([p_rows, baseline], ignore_index=True)

theta_df = build_param_df("theta")
tau_df   = build_param_df("tau")
nmin_df  = build_param_df("n_min")

for name, dfx in [("theta", theta_df), ("tau", tau_df), ("n_min", nmin_df)]:
    print(f"\n=== {name} sweep ===")
    print(dfx.groupby(["benchmark", "swept_val"])["sr_pct"].agg(["mean", "std", "count"]).round(2))

# ==================== Figure ====================
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

BENCH_TO_TOOL = {"vadd": "Bambu", "gcd": "Dynamatic"}
COLORS  = {"vadd": "#4E79A7", "gcd": "#E15759"}
MARKERS = {"vadd": "o", "gcd": "s"}

sweeps = [
    ("theta", theta_df, "$\\theta$ (confidence threshold)", [0.7, 0.8, 0.9]),
    ("tau",   tau_df,   "$\\tau$ (min support)",             [2, 3, 4, 5]),
    ("n_min", nmin_df,  "$n_{\\min}$ (min observations)",    [3, 5, 8]),
]

for ax_idx, (pname, pdf, xlabel, xvals) in enumerate(sweeps):
    ax = axes[ax_idx]
    for bench, tool in BENCH_TO_TOOL.items():
        sub = pdf[pdf["benchmark"] == bench]
        grouped = sub.groupby("swept_val")["sr_pct"].agg(["mean", "std"]).reset_index()
        grouped = grouped.sort_values("swept_val")

        ax.errorbar(grouped["swept_val"], grouped["mean"],
                    yerr=grouped["std"],
                    color=COLORS[bench], marker=MARKERS[bench], markersize=7,
                    linewidth=1.6, capsize=4, capthick=1.0,
                    markeredgecolor="black", markeredgewidth=0.5,
                    label=f"{bench} ({tool})")

    # Mark default
    ax.axvline(x=DEFAULTS[pname], color="#888", linestyle=":",
               linewidth=0.8, alpha=0.6)
    ax.text(DEFAULTS[pname], 55,
            f"default\n{pname}={DEFAULTS[pname]}",
            fontsize=7.5, color="#666", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.9))

    ax.set_xlabel(xlabel)
    if ax_idx == 0:
        ax.set_ylabel("Success Rate (%)")
    ax.set_title(f"({'abc'[ax_idx]}) Sensitivity to {pname}")
    ax.set_xticks(xvals)
    ax.set_ylim(50, 105)
    ax.grid(True, alpha=0.3)
    if ax_idx == 0:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

plt.tight_layout()
plt.savefig(OUT / "fig_sensitivity.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig_sensitivity.png", bbox_inches="tight", dpi=300)
print(f"\n✅ Saved: {OUT}/fig_sensitivity.pdf")
plt.show()
