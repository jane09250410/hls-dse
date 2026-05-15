#!/usr/bin/env python3
"""
rerun_ablation_n5.py
=====================
Re-run Dynamatic ablation with 5 permutations per config (up from 3).
Produces n=40 per config (8 benchmarks × 5 perms).

Usage:
    cd ~/hls-dse
    python3 rerun_ablation_n5.py
"""

import csv, os, sys, time, hashlib
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dynamatic_config_generator import generate_dynamatic_configs
from offline_sim.simulator import GroundTruthOracle, simulate_run
from methods.pa_dse_method import PADSEMethod

GT_CSV = "results/dynamatic_ground_truth/run_summary.csv"
DYN_PATH = os.path.expanduser("~/dynamatic/integration-test")

BENCH_MAP = {
    "matmul": "matrix", "vadd": "vadd", "fir": "fir",
    "histogram": "histogram", "atax": "atax", "bicg": "bicg",
    "gemm": "gemm", "gesummv": "gesummv",
}
BENCHMARKS = list(BENCH_MAP.keys())

ABLATIONS = [
    "no-filter", "SCF-only", "SCF+RPE", "SCF+OFRS",
    "SCF+DFRL", "DFRL-only", "SCF+RPE-reorder", "SCF+OFRS-skip",
]

BUDGET = 30
N_PERMS = 5  # increased from 3

SUMMARY_COLUMNS = [
    "run_id", "strategy", "benchmark", "tool", "budget", "seed",
    "queue_permutation_id", "ablation_config",
    "tau", "theta", "n_min", "p_probe",
    "total_evals", "successful_evals", "wasted_calls", "sr_pct",
    "best_qor_name", "best_qor_value", "best_area", "best_latency",
    "uqor", "ttff_s", "total_wall_clock_s",
    "total_skipped", "false_skips_pending", "false_skips_verified",
    "signatures_learned", "probes_triggered", "probes_succeeded",
    "overhead_phago_ms", "overhead_rpe_ms", "overhead_ofrs_ms",
]


def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)


def make_run_id(strategy, benchmark, budget, seed):
    h = hashlib.md5(f"{strategy}_{benchmark}_{budget}_{seed}_{time.time()}".encode()).hexdigest()[:7]
    return f"{strategy}_{benchmark}_B{budget}_s{seed}_{h}"


def get_source_path(bench):
    d = BENCH_MAP[bench]
    return os.path.join(DYN_PATH, d, f"{d}.c")


def result_to_row(result, run_id, perm, abl):
    return {
        "run_id": run_id,
        "strategy": result["strategy"],
        "benchmark": result["benchmark"],
        "tool": "dynamatic",
        "budget": result["budget"],
        "seed": None,
        "queue_permutation_id": perm,
        "ablation_config": abl,
        "tau": 2, "theta": 0.8, "n_min": 5, "p_probe": 0.05,
        "total_evals": result["total_evals"],
        "successful_evals": result["successful_evals"],
        "wasted_calls": result["wasted_calls"],
        "sr_pct": round(result["sr_pct"], 1),
        "best_qor_name": "component_count",
        "best_qor_value": result["best_area"],
        "best_area": result["best_area"],
        "best_latency": result["best_latency"],
        "uqor": result["uqor"],
        "ttff_s": round(result["ttff_synth_s"], 3) if result["ttff_synth_s"] else "",
        "total_wall_clock_s": round(result.get("total_wall_clock_s", 0), 3),
        "total_skipped": result["total_skipped"],
        "false_skips_pending": 0,
        "false_skips_verified": "",
        "signatures_learned": result["signatures_learned"],
        "probes_triggered": result["probes_triggered"],
        "probes_succeeded": result["probes_succeeded"],
        "overhead_phago_ms": 0,
        "overhead_rpe_ms": 0,
        "overhead_ofrs_ms": 0,
    }


def main():
    log("Loading oracle...")
    oracle = GroundTruthOracle(GT_CSV, tool="dynamatic")
    configs = generate_dynamatic_configs()
    log(f"Config space: {len(configs)}, Benchmarks: {len(BENCHMARKS)}")
    log(f"Ablation configs: {len(ABLATIONS)}, Perms: {N_PERMS}")
    log(f"Total runs: {len(ABLATIONS)} × {len(BENCHMARKS)} × {N_PERMS} = {len(ABLATIONS)*len(BENCHMARKS)*N_PERMS}")

    rows = []
    t0 = time.time()

    for bi, bench in enumerate(BENCHMARKS):
        src = get_source_path(bench)
        log(f"[{bi+1}/8] {bench}")

        for abl in ABLATIONS:
            for perm in range(N_PERMS):
                m = PADSEMethod(configs, bench, "dynamatic", BUDGET,
                                ablation_config=abl,
                                source_path=src,
                                queue_permutation_id=perm)
                result = simulate_run(m, oracle, bench)
                rid = make_run_id(f"PA-DSE_{abl}", bench, BUDGET, perm)
                rows.append(result_to_row(result, rid, perm, abl))

    out_path = "results/b30/ablation_dynamatic/run_summary.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Backup existing
    if os.path.exists(out_path):
        bak = out_path + ".bak_n3"
        if not os.path.exists(bak):
            os.rename(out_path, bak)
            log(f"Backed up old file to {bak}")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    log(f"Wrote {len(rows)} rows → {out_path}")
    log(f"Done in {time.time()-t0:.1f}s")

    # Print summary
    import pandas as pd
    df = pd.DataFrame(rows)
    NT = ['matmul', 'atax', 'bicg', 'gemm', 'gesummv']
    df_nt = df[df['benchmark'].isin(NT)]
    print(f"\n{'Config':25s}  {'SR':>10s}  {'Wasted':>6s}  n")
    print("-" * 55)
    for abl in ABLATIONS:
        sub = df_nt[df_nt['ablation_config'] == abl]
        print(f"{abl:25s}  {sub['sr_pct'].mean():5.1f}±{sub['sr_pct'].std():4.1f}  {sub['wasted_calls'].mean():6.1f}  {len(sub)}")


if __name__ == "__main__":
    main()
