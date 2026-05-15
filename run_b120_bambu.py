#!/usr/bin/env python3
"""
run_b120_bambu.py
==================
B=120 Bambu main comparison + SCF+OFRS ablation via offline simulation.
Requires: results/bambu_ground_truth/run_summary.csv (from collect_bambu_gt.py)

Methods: Random, FilteredRandom, SA, GA, GP-BO, RF, PA-DSE (SCF+DFRL), SCF+OFRS
8 benchmarks × 10 seeds/perms = 80 runs per method, 640 total
Runtime: ~10 minutes (offline simulation)

Usage:
    cd ~/hls-dse
    python3 run_b120_bambu.py

Output:
    results/b120/bambu_main/run_summary.csv
    results/b120/bambu_main/eval_log.csv
"""

import csv, os, sys, time, hashlib
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "scripts"))

from config_generator import generate_bambu_configs
from offline_sim.simulator import GroundTruthOracle, simulate_run
from methods.pa_dse_method import PADSEMethod
from methods.baseline_methods import RandomMethod, FilteredRandomMethod
from methods.advanced_baselines import (
    SimulatedAnnealingMethod, GeneticAlgorithmMethod,
    GPBayesOptMethod, RFClassifierMethod,
)

GT_CSV = "results/bambu_ground_truth/run_summary.csv"
BENCHMARKS = ["matmul", "vadd", "fir", "histogram", "atax", "bicg", "gemm", "gesummv"]
BUDGET = 120
N_SEEDS = 10

OUT_DIR = "results/b120/bambu_main"

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

EVAL_LOG_COLUMNS = [
    "run_id", "strategy", "benchmark", "eval_step",
    "config_id", "success", "area", "latency", "action",
]


def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)


def make_run_id(strategy, benchmark, seed):
    h = hashlib.md5(f"{strategy}_{benchmark}_{BUDGET}_{seed}_{time.time()}".encode()).hexdigest()[:7]
    return f"{strategy}_{benchmark}_B{BUDGET}_s{seed}_{h}"


def result_to_row(result, run_id, seed, ablation_config="N/A",
                  perm_id=None, tau=0, theta=0, n_min=0, p_probe=0):
    return {
        "run_id": run_id,
        "strategy": result["strategy"],
        "benchmark": result["benchmark"],
        "tool": "bambu",
        "budget": result["budget"],
        "seed": seed,
        "queue_permutation_id": perm_id,
        "ablation_config": ablation_config,
        "tau": tau, "theta": theta, "n_min": n_min, "p_probe": p_probe,
        "total_evals": result["total_evals"],
        "successful_evals": result["successful_evals"],
        "wasted_calls": result["wasted_calls"],
        "sr_pct": round(result["sr_pct"], 1),
        "best_qor_name": "area",
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


def eval_log_rows(result, run_id):
    rows = []
    for entry in result.get("eval_log", []):
        rows.append({
            "run_id": run_id,
            "strategy": result["strategy"],
            "benchmark": result["benchmark"],
            "eval_step": entry["step"],
            "config_id": entry["config_id"],
            "success": entry["success"],
            "area": entry["area"] if entry["area"] is not None else "",
            "latency": entry["latency"] if entry["latency"] is not None else "",
            "action": entry.get("action", "evaluate"),
        })
    return rows


def write_csv(path, rows, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    log(f"  Wrote {len(rows)} rows -> {path}")


def main():
    log("=" * 60)
    log(f"B=120 Bambu Main Comparison (offline simulation)")
    log("=" * 60)

    if not os.path.exists(GT_CSV):
        log(f"ERROR: {GT_CSV} not found. Run collect_bambu_gt.py first.")
        sys.exit(1)

    oracle = GroundTruthOracle(GT_CSV, tool="bambu")
    log(f"Oracle benchmarks: {oracle.benchmarks()}")

    configs = generate_bambu_configs(enable_pipeline=True)
    log(f"Config space: {len(configs)}")

    summary_rows = []
    eval_rows = []
    t0 = time.time()

    for bi, bench in enumerate(BENCHMARKS):
        src = os.path.abspath(f"benchmarks/{bench}/{bench}.c")
        log(f"[{bi+1}/8] {bench}")

        # === 6 baselines ===
        baseline_classes = [
            ("Random",             RandomMethod),
            ("Filtered_Random",    FilteredRandomMethod),
            ("SimulatedAnnealing", SimulatedAnnealingMethod),
            ("GeneticAlgorithm",   GeneticAlgorithmMethod),
            ("GP-BO",              GPBayesOptMethod),
            ("RF_Classifier",      RFClassifierMethod),
        ]

        for method_name, cls in baseline_classes:
            for seed in range(N_SEEDS):
                if cls == FilteredRandomMethod:
                    m = cls(configs, bench, "bambu", BUDGET,
                            seed=seed, source_path=src)
                else:
                    m = cls(configs, bench, "bambu", BUDGET, seed=seed)

                result = simulate_run(m, oracle, bench)
                rid = make_run_id(result["strategy"], bench, seed)
                summary_rows.append(result_to_row(result, rid, seed))
                eval_rows.extend(eval_log_rows(result, rid))

        # === PA-DSE (SCF+DFRL full) ===
        for perm in range(N_SEEDS):
            m = PADSEMethod(configs, bench, "bambu", BUDGET,
                            ablation_config="SCF+DFRL",
                            source_path=src,
                            queue_permutation_id=perm)
            result = simulate_run(m, oracle, bench)
            rid = make_run_id("PA-DSE_SCF+DFRL", bench, perm)
            summary_rows.append(result_to_row(
                result, rid, seed=None, ablation_config="SCF+DFRL",
                perm_id=perm, tau=2, theta=0.8, n_min=5, p_probe=0.05))
            eval_rows.extend(eval_log_rows(result, rid))

        # === SCF+OFRS (no RPE) — key ablation control ===
        for perm in range(N_SEEDS):
            m = PADSEMethod(configs, bench, "bambu", BUDGET,
                            ablation_config="SCF+OFRS",
                            source_path=src,
                            queue_permutation_id=perm)
            result = simulate_run(m, oracle, bench)
            rid = make_run_id("PA-DSE_SCF+OFRS", bench, perm)
            summary_rows.append(result_to_row(
                result, rid, seed=None, ablation_config="SCF+OFRS",
                perm_id=perm, tau=2, theta=0.8, n_min=5, p_probe=0.05))
            eval_rows.extend(eval_log_rows(result, rid))

    write_csv(os.path.join(OUT_DIR, "run_summary.csv"), summary_rows, SUMMARY_COLUMNS)
    write_csv(os.path.join(OUT_DIR, "eval_log.csv"), eval_rows, EVAL_LOG_COLUMNS)

    elapsed = time.time() - t0
    log(f"\nAll done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # === Summary ===
    import pandas as pd
    df = pd.DataFrame(summary_rows)

    RENAME = {'Random':'Random','Filtered_Random':'FilteredRandom',
              'SimulatedAnnealing':'SA','GeneticAlgorithm':'GA',
              'GP-BO':'GP-BO','RF_Classifier':'RF',
              'PA-DSE_SCF+DFRL':'PA-DSE','PA-DSE_SCF+OFRS':'SCF+OFRS'}
    df['method'] = df['strategy'].map(RENAME).fillna(df['strategy'])

    ORDER = ['Random','FilteredRandom','SA','GA','GP-BO','RF','SCF+OFRS','PA-DSE']

    print(f"\n{'Method':20s}  {'SR':>12s}  {'Wasted':>8s}  {'TTFF':>8s}  {'UQoR':>6s}  {'Sigs':>6s}  n")
    print("-" * 80)
    for m in ORDER:
        sub = df[df['method'] == m]
        if len(sub) == 0:
            continue
        sigs = sub['signatures_learned'].mean()
        print(f"{m:20s}  {sub['sr_pct'].mean():5.1f}±{sub['sr_pct'].std():4.1f}"
              f"  {sub['wasted_calls'].mean():8.1f}"
              f"  {sub['ttff_s'].mean() if sub['ttff_s'].dtype != object else 0:8.1f}"
              f"  {sub['uqor'].mean():6.1f}"
              f"  {sigs:6.1f}  {len(sub)}")

    # KEY COMPARISON: PA-DSE vs SCF+OFRS
    pa = df[df['method'] == 'PA-DSE']['sr_pct']
    ofrs = df[df['method'] == 'SCF+OFRS']['sr_pct']
    delta = pa.mean() - ofrs.mean()

    from scipy import stats
    # Paired test over per-benchmark means
    pa_bench = df[df['method']=='PA-DSE'].groupby('benchmark')['sr_pct'].mean()
    ofrs_bench = df[df['method']=='SCF+OFRS'].groupby('benchmark')['sr_pct'].mean()
    common = sorted(set(pa_bench.index) & set(ofrs_bench.index))
    t_stat, p_val = stats.ttest_rel([pa_bench[b] for b in common],
                                     [ofrs_bench[b] for b in common])

    print(f"\n*** KEY RESULT ***")
    print(f"  PA-DSE (SCF+DFRL): {pa.mean():.1f}% ± {pa.std():.1f}")
    print(f"  SCF+OFRS:          {ofrs.mean():.1f}% ± {ofrs.std():.1f}")
    print(f"  Delta:             {delta:+.1f} pp")
    print(f"  Paired t-test:     t={t_stat:.3f}, p={p_val:.4f}")
    if abs(delta) > 0.5 and p_val < 0.05:
        print(f"  --> RPE CONTRIBUTES at B=120 (significant)")
    elif abs(delta) > 0.5:
        print(f"  --> RPE contributes but not significant (p={p_val:.3f})")
    else:
        print(f"  --> RPE absorbed by OFRS even at B=120")

    # RPE activation stats
    pa_runs = df[df['method'] == 'PA-DSE']
    print(f"\n  RPE activation: sigs_learned = {pa_runs['signatures_learned'].mean():.2f} ± {pa_runs['signatures_learned'].std():.2f}")
    print(f"  Probes triggered: {pa_runs['probes_triggered'].mean():.1f}")
    print(f"  Total skipped: {pa_runs['total_skipped'].mean():.1f}")


if __name__ == "__main__":
    main()
