#!/usr/bin/env python3
"""compute_paper_tables.py — compute all summary tables needed for the paper.

Run in the paper_figures directory:
    python3 compute_paper_tables.py

Outputs CSV files into ~/Desktop/hls/paper_figures/out/:
    - table_main_bambu.csv         main comparison
    - table_main_dynamatic.csv     main comparison
    - table_ablation_bambu.csv     8-way ablation
    - table_ablation_dynamatic.csv 8-way ablation
    - table_overhead.csv           overhead breakdown

And prints a paper-ready markdown table.
"""

import pandas as pd
from pathlib import Path

ROOT = Path("/Users/zhangxinyu/Desktop/hls/results")
OUT  = Path("/Users/zhangxinyu/Desktop/hls/paper_figures/out")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# MAIN COMPARISON (Bambu + Dynamatic)
# ============================================================

BASELINE_MAP = {
    "Random":             "Random",
    "Filtered_Random":    "FilteredRandom",
    "SimulatedAnnealing": "SA",
    "GeneticAlgorithm":   "GA",
    "GP-BO":              "GP-BO",
    "RF_Classifier":      "RF",
}

METRICS_MAIN = ["sr_pct", "wasted_calls", "ttff_s", "uqor", "best_area", "best_latency"]
METHOD_ORDER = ["Random", "FilteredRandom", "SA", "GA", "GP-BO", "RF", "PA-DSE"]


def main_table(main_path, perms_path):
    """Combine main baselines + PA-DSE perms into a single summary."""
    main = pd.read_csv(ROOT / main_path)
    perms = pd.read_csv(ROOT / perms_path)

    # drop PA-DSE variants from main (use perms instead, more reps)
    main = main[~main["strategy"].str.contains("PA-DSE")].copy()
    main["strategy"] = main["strategy"].map(BASELINE_MAP).fillna(main["strategy"])

    perms = perms.copy()
    perms["strategy"] = "PA-DSE"

    df = pd.concat([main, perms], ignore_index=True)

    rows = []
    for m in METHOD_ORDER:
        sub = df[df["strategy"] == m]
        if len(sub) == 0: continue
        row = {"method": m, "n": len(sub)}
        for col in METRICS_MAIN:
            if col in sub.columns:
                # some metrics (ttff, best_*) may be NaN when SR=0; drop those
                vals = sub[col].dropna()
                if len(vals) > 0:
                    row[f"{col}_mean"] = vals.mean()
                    row[f"{col}_std"] = vals.std()
                else:
                    row[f"{col}_mean"] = float("nan")
                    row[f"{col}_std"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows), df


print("=" * 80)
print("MAIN RESULTS — BAMBU")
print("=" * 80)
bambu_tbl, bambu_all = main_table(
    "master/bambu_main/run_summary.csv",
    "rerun/bambu_pa_dse_perms/run_summary.csv",
)
bambu_tbl.to_csv(OUT / "table_main_bambu.csv", index=False)
print(bambu_tbl.round(2).to_string(index=False))

print()
print("=" * 80)
print("MAIN RESULTS — DYNAMATIC (4 benchmarks)")
print("=" * 80)
dyn_tbl, dyn_all = main_table(
    "master/dynamatic_main/run_summary.csv",
    "rerun/dynamatic_pa_dse_perms/run_summary.csv",
)
dyn_tbl.to_csv(OUT / "table_main_dynamatic.csv", index=False)
print(dyn_tbl.round(2).to_string(index=False))

# ============================================================
# ABLATION (Bambu + Dynamatic) @ B=60
# ============================================================

ABLATION_ORDER = [
    "no-filter", "SCF-only",
    "SCF+RPE", "SCF+OFRS",
    "SCF+DFRL",
    "DFRL-only",
    "SCF+RPE-reorder", "SCF+OFRS-skip",
]

def ablation_table(path):
    df = pd.read_csv(ROOT / path)

    rows = []
    for cfg in ABLATION_ORDER:
        sub = df[df["ablation_config"] == cfg]
        if len(sub) == 0: continue
        row = {"config": cfg, "n": len(sub)}
        for col in ["sr_pct", "wasted_calls", "ttff_s",
                    "signatures_learned", "total_skipped"]:
            if col in sub.columns:
                vals = sub[col].dropna()
                if len(vals) > 0:
                    row[f"{col}_mean"] = vals.mean()
                    row[f"{col}_std"] = vals.std()
                else:
                    row[f"{col}_mean"] = float("nan")
                    row[f"{col}_std"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


print()
print("=" * 80)
print("ABLATION — BAMBU (b30/ablation_bambu)")
print("=" * 80)
bam_abl = ablation_table("b30/ablation_bambu/run_summary.csv")
bam_abl.to_csv(OUT / "table_ablation_bambu.csv", index=False)
print(bam_abl.round(2).to_string(index=False))

print()
print("=" * 80)
print("ABLATION — DYNAMATIC (b30/ablation_dynamatic)")
print("=" * 80)
dyn_abl = ablation_table("b30/ablation_dynamatic/run_summary.csv")
dyn_abl.to_csv(OUT / "table_ablation_dynamatic.csv", index=False)
print(dyn_abl.round(2).to_string(index=False))

# ============================================================
# OVERHEAD (Bambu + Dynamatic)
# ============================================================

print()
print("=" * 80)
print("OVERHEAD BREAKDOWN (ms per iteration)")
print("=" * 80)

overhead_rows = []
for tool, path in [
    ("Bambu",     "rerun/bambu_pa_dse_perms/run_summary.csv"),
    ("Dynamatic", "rerun/dynamatic_pa_dse_perms/run_summary.csv"),
]:
    df = pd.read_csv(ROOT / path)
    scf  = (df["overhead_phago_ms"] / df["total_evals"]).mean()
    rpe  = (df["overhead_rpe_ms"]   / df["total_evals"]).mean()
    ofrs = (df["overhead_ofrs_ms"]  / df["total_evals"]).mean()
    total = scf + rpe + ofrs
    synth = ((df["total_wall_clock_s"] * 1000
              - (df["overhead_phago_ms"] + df["overhead_rpe_ms"] + df["overhead_ofrs_ms"]))
             / df["total_evals"]).mean()
    overhead_rows.append({
        "tool": tool,
        "scf_ms": scf, "rpe_ms": rpe, "ofrs_ms": ofrs,
        "algo_total_ms": total, "synth_ms": synth,
        "ratio": synth / total,
    })
ov_df = pd.DataFrame(overhead_rows)
ov_df.to_csv(OUT / "table_overhead.csv", index=False)
print(ov_df.round(3).to_string(index=False))

print()
print(f"\nAll tables saved to {OUT}/")
