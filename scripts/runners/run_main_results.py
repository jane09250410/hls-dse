#!/usr/bin/env python3
"""
run_main_results.py — E1: Overall Performance.

Usage:
    python3 scripts/runners/run_main_results.py --phase 1     # primary budget
    python3 scripts/runners/run_main_results.py --phase 2     # budget sweep + appendix baselines
"""

import argparse, os, subprocess, sys, time, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config_generator import generate_bambu_configs, config_to_bambu_cmd
from logging.experiment_logger import ExperimentLogger
from runners.run_single import run_single
from methods.baseline_methods import (
    RandomMethod, FilteredRandomMethod, GridMethod, LHSMethod, FailureMemoMethod)
from methods.pa_dse_method import PADSEMethod

# ── Frozen config ───────────────────────────────────────────────

BAMBU = {
    "matmul":    {"src": "benchmarks/matmul/matmul.c",       "top": "matmul"},
    "vadd":      {"src": "benchmarks/vadd/vadd.c",           "top": "vadd"},
    "fir":       {"src": "benchmarks/fir/fir.c",             "top": "fir"},
    "histogram": {"src": "benchmarks/histogram/histogram.c", "top": "histogram"},
}
BAMBU_PRIMARY = [60]
BAMBU_SWEEP   = [20, 40, 60, 80]
N_SEEDS = 20


def make_bambu_synth(src, top, results_base):
    """Create a synthesis function for Bambu."""
    def synthesize(config):
        cmd = config_to_bambu_cmd(config, os.path.abspath(src), top)
        work_dir = os.path.join(results_base, f"cfg_{config['id']}")
        os.makedirs(work_dir, exist_ok=True)
        t0 = time.time()
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=work_dir,
                capture_output=True, text=True, timeout=300,
            )
            output = result.stdout + "\n" + result.stderr
            elapsed = time.time() - t0
            success = result.returncode == 0 and "Total area" in output
        except subprocess.TimeoutExpired:
            output = "TIMEOUT"
            elapsed = time.time() - t0
            success = False
        except Exception as e:
            output = f"ERROR: {e}"
            elapsed = time.time() - t0
            success = False
        return output, elapsed, success
    return synthesize


def run_bambu_phase(benchmarks, budgets, logger, include_appendix):
    """Run all Bambu experiments for given budgets."""
    configs = generate_bambu_configs(enable_pipeline=True)

    for bname, binfo in benchmarks.items():
        src, top = binfo["src"], binfo["top"]

        for B in budgets:
            base = f"results/experiments/bambu/{bname}/B{B}"
            synth = make_bambu_synth(src, top, base)

            # ── Main baselines (always) ─────────────────────────
            for seed in range(N_SEEDS):
                print(f"  Random / {bname} / B={B} / seed={seed}")
                m = RandomMethod(configs, bname, "bambu", B, seed=seed)
                run_single(m, synth, logger, tool="bambu",
                           ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)

            for seed in range(N_SEEDS):
                print(f"  Filtered_Random / {bname} / B={B} / seed={seed}")
                m = FilteredRandomMethod(configs, bname, "bambu", B,
                                         seed=seed, source_path=src)
                run_single(m, synth, logger, tool="bambu",
                           ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)

            print(f"  PA-DSE_L1 / {bname} / B={B}")
            m = PADSEMethod(configs, bname, "bambu", B,
                            ablation_config="phago+Full",
                            dynamic_mode="intersection", source_path=src)
            run_single(m, synth, logger, tool="bambu",
                       ablation_config="phago+Full(L1)")

            print(f"  PA-DSE_Full / {bname} / B={B}")
            m = PADSEMethod(configs, bname, "bambu", B,
                            ablation_config="phago+Full", source_path=src)
            run_single(m, synth, logger, tool="bambu")

            # ── Appendix baselines (phase 2 only) ──────────────
            if include_appendix:
                print(f"  Grid / {bname} / B={B}")
                m = GridMethod(configs, bname, "bambu", B)
                run_single(m, synth, logger, tool="bambu",
                           ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)

                for seed in range(N_SEEDS):
                    print(f"  LHS / {bname} / B={B} / seed={seed}")
                    m = LHSMethod(configs, bname, "bambu", B, seed=seed)
                    run_single(m, synth, logger, tool="bambu",
                               ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)

                print(f"  Failure_Memo / {bname} / B={B}")
                m = FailureMemoMethod(configs, bname, "bambu", B)
                run_single(m, synth, logger, tool="bambu",
                           ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)


def main():
    parser = argparse.ArgumentParser(description="E1: Main results")
    parser.add_argument("--phase", choices=["1", "2"], required=True)
    args = parser.parse_args()

    logger = ExperimentLogger()

    if args.phase == "1":
        print("=== Phase 1: Primary budgets (B=60) — main baselines only ===")
        run_bambu_phase(BAMBU, BAMBU_PRIMARY, logger, include_appendix=False)
    else:
        print("=== Phase 2: Budget sweep (B=20,40,60,80) + appendix baselines ===")
        run_bambu_phase(BAMBU, BAMBU_SWEEP, logger, include_appendix=True)

    print(f"Done. Logs → {logger.output_dir}")


if __name__ == "__main__":
    main()
