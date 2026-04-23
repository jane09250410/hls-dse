#!/usr/bin/env python3
"""compute_paper_tables.py — compute all summary tables needed for the paper.

NOTE on Dynamatic benchmark filter:
    fir and histogram produce SR=100% for ALL methods on Dynamatic, so they
    carry no discriminative signal. The paper (§IV.A) evaluates Dynamatic on
    4 benchmarks: gcd, matching, binary_search, kernel_2mm. Any aggregation
    over Dynamatic MUST apply this filter or the reported numbers will
    diverge from Table III.
"""

import pandas as pd
from pathlib import Path

ROOT = Path("/Users/zhangxinyu/Desktop/hls/results")
OUT  = Path("/Users/zhangxinyu/Desktop/hls/paper_figures/out")
OUT.mkdir(parents=True, exist_ok=True)

BAMBU_BENCHMARKS = [
    "matmul", "vadd", "fir", "histogram",
    "atax", "bicg", "gemm", "gesummv",
]
DYNAMATIC_BENCHMARKS = [
    "gcd", "matching", "binary_search", "kernel_2mm",
]

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


def main_table(main_path, perms_path, bench_filter):
    main = pd.read_csv(ROOT / main_path)
    perms = pd.read_csv(ROOT / perms_path)

    before_main, before_perms = len(main), len(perms)
    main = main[main["benchmark"].isin(bench_filter)].copy()
    perms = perms[perms["benchmark"].isin(bench_filter)].copy()
    print(f"  [filter] main: {before_main} -> {len(main)} rows "
          f"({len(bench_filter)} benchmarks)")
    print(f"  [filter] perms: {before_perms} -> {len(perms)} rows")

    main = main[~main["strategy"].str.contains("PA-DSE")].copy()
    main["strategy"] = main["strategy"].map(BASELINE_MAP).fillna(main["strategy"])
    perms["strategy"] = "PA-DSE"

    df = pd.concat([main, perms], ignore_index=True)

    rows = []
    for m in METHOD_ORDER:
        sub = df[df["strategy"] == m]
        if len(sub) == 0: continue
        row = {"method": m, "n": len(sub)}
        for col in METRICS_MAIN:
            if col in sub.columns:
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
print("MAIN RESULTS - BAMBU (8 benchmarks)")
print("=" * 80)
bambu_tbl, bambu_all = main_table(
    "master/bambu_main/run_summary.csv",
    "rerun/bambu_pa_dse_perms/run_summary.csv",
    BAMBU_BENCHMARKS,
)
bambu_tbl.to_csv(OUT / "table_main_bambu.csv", index=False)
print(bambu_tbl.round(2).to_string(index=False))

print()
print("=" * 80)
print("MAIN RESULTS - DYNAMATIC (4 benchmarks; fir/histogram excluded, SR=100% for all)")
print("=" * 80)
dyn_tbl, dyn_all = main_table(
    "master/dynamatic_main/run_summary.csv",
    "rerun/dynamatic_pa_dse_perms/run_summary.csv",
    DYNAMATIC_BENCHMARKS,
)
dyn_tbl.to_csv(OUT / "table_main_dynamatic.csv", index=False)
print(dyn_tbl.round(2).to_string(index=False))

# ============================================================
# ABLATION
# ============================================================
ABLATION_ORDER = [
    "no-filter", "SCF-only",
    "SCF+RPE", "SCF+OFRS",
    "SCF+DFRL",
    "DFRL-only",
    "SCF+RPE-reorder", "SCF+OFRS-skip",
]

def ablation_table(path, bench_filter):
    df = pd.read_csv(ROOT / path)
    before = len(df)
    df = df[df["benchmark"].isin(bench_filter)].copy()
    print(f"  [filter] ablation: {before} -> {len(df)} rows")

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
print("ABLATION - BAMBU (8 benchmarks)")
print("=" * 80)
bam_abl = ablation_table("b30/ablation_bambu/run_summary.csv", BAMBU_BENCHMARKS)
bam_abl.to_csv(OUT / "table_ablation_bambu.csv", index=False)
print(bam_abl.round(2).to_string(index=False))

print()
print("=" * 80)
print("ABLATION - DYNAMATIC (4 benchmarks)")
print("=" * 80)
dyn_abl = ablation_table("b30/ablation_dynamatic/run_summary.csv", DYNAMATIC_BENCHMARKS)
dyn_abl.to_csv(OUT / "table_ablation_dynamatic.csv", index=False)
print(dyn_abl.round(2).to_string(index=False))

# ============================================================
# OVERHEAD
# ============================================================
print()
print("=" * 80)
print("OVERHEAD BREAKDOWN (ms per iteration)")
print("=" * 80)

overhead_rows = []
for tool, path, bench_filter in [
    ("Bambu",     "rerun/bambu_pa_dse_perms/run_summary.csv",     BAMBU_BENCHMARKS),
    ("Dynamatic", "rerun/dynamatic_pa_dse_perms/run_summary.csv", DYNAMATIC_BENCHMARKS),
]:
    df = pd.read_csv(ROOT / path)
    before = len(df)
    df = df[df["benchmark"].isin(bench_filter)].copy()
    print(f"  [filter] {tool}: {before} -> {len(df)} rows")

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
