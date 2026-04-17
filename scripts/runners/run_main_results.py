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
from exp_logging.experiment_logger import ExperimentLogger
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

DYNAMATIC_PATH = os.path.expanduser("~/dynamatic")
DYNAMATIC = {
    "gcd":           {"src": f"{DYNAMATIC_PATH}/integration-test/gcd/gcd.c",                     "top": "gcd"},
    "matching":      {"src": f"{DYNAMATIC_PATH}/integration-test/matching/matching.c",             "top": "matching"},
    "binary_search": {"src": f"{DYNAMATIC_PATH}/integration-test/binary_search/binary_search.c",   "top": "binary_search"},
    "fir":           {"src": f"{DYNAMATIC_PATH}/integration-test/fir/fir.c",                       "top": "fir"},
    "histogram":     {"src": f"{DYNAMATIC_PATH}/integration-test/histogram/histogram.c",           "top": "histogram"},
}
DYNAMATIC_PRIMARY = [30]
DYNAMATIC_SWEEP   = [20, 30, 40, 60]

N_SEEDS = 20


def make_dynamatic_synth(src, top, results_base):
    """Create a synthesis function for Dynamatic.

    Each config gets its own temp directory with a copy of the source file,
    so that Dynamatic's output directory (out/) does not conflict across
    sequential evaluations.
    """
    from run_dynamatic_single import run_dynamatic_single
    import shutil, tempfile

    def synthesize(config):
        # Create isolated work directory per config
        work_dir = os.path.join(results_base, f"cfg_{config['id']}")
        os.makedirs(work_dir, exist_ok=True)

        # Copy source file to work directory
        src_abs = os.path.abspath(src)
        src_copy = os.path.join(work_dir, os.path.basename(src_abs))
        src_dir_path = os.path.dirname(src_abs)
        for f in os.listdir(src_dir_path):
            full = os.path.join(src_dir_path, f)
            if os.path.isfile(full):
                shutil.copy2(full, work_dir)

        success, metrics, raw_output = run_dynamatic_single(
            config=config,
            src_file=src_copy,
            top_func=top,
            timeout=120,
        )
        encoded = raw_output
        if success:
            nc = metrics.get("num_components", 0)
            nb = metrics.get("num_buffers", 0)
            nho = metrics.get("num_handshake_ops", 0)
            encoded += f"\n[METRICS] components={nc} buffers={nb} handshake_ops={nho}"
        elapsed = metrics.get("compile_time_s", 0.0)
        return encoded, elapsed, success
    return synthesize


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
            success = result.returncode == 0 and "Total estimated area" in output
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


def run_dynamatic_phase(benchmarks, budgets, logger, include_appendix):
    """Run all Dynamatic experiments for given budgets."""
    from dynamatic_config_generator import generate_dynamatic_configs
    configs = generate_dynamatic_configs()

    for bname, binfo in benchmarks.items():
        src, top = binfo["src"], binfo["top"]

        for B in budgets:
            base = f"results/experiments/dynamatic/{bname}/B{B}"
            synth = make_dynamatic_synth(src, top, base)

            # ── Main baselines (always) ─────────────────────────
            for seed in range(N_SEEDS):
                print(f"  Random / {bname} / B={B} / seed={seed}", flush=True)
                m = RandomMethod(configs, bname, "dynamatic", B, seed=seed)
                run_single(m, synth, logger, tool="dynamatic",
                           ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)

            for seed in range(N_SEEDS):
                print(f"  Filtered_Random / {bname} / B={B} / seed={seed}", flush=True)
                m = FilteredRandomMethod(configs, bname, "dynamatic", B,
                                         seed=seed, source_path=src)
                run_single(m, synth, logger, tool="dynamatic",
                           ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)

            print(f"  PA-DSE_L1 / {bname} / B={B}", flush=True)
            m = PADSEMethod(configs, bname, "dynamatic", B,
                            ablation_config="phago+Full",
                            dynamic_mode="intersection", source_path=src)
            run_single(m, synth, logger, tool="dynamatic",
                       ablation_config="phago+Full(L1)")

            print(f"  PA-DSE_Full / {bname} / B={B}", flush=True)
            m = PADSEMethod(configs, bname, "dynamatic", B,
                            ablation_config="phago+Full", source_path=src)
            run_single(m, synth, logger, tool="dynamatic")

            # ── Appendix baselines (phase 2 only) ──────────────
            if include_appendix:
                print(f"  Grid / {bname} / B={B}", flush=True)
                m = GridMethod(configs, bname, "dynamatic", B)
                run_single(m, synth, logger, tool="dynamatic",
                           ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)

                for seed in range(N_SEEDS):
                    print(f"  LHS / {bname} / B={B} / seed={seed}", flush=True)
                    m = LHSMethod(configs, bname, "dynamatic", B, seed=seed)
                    run_single(m, synth, logger, tool="dynamatic",
                               ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)

                print(f"  Failure_Memo / {bname} / B={B}", flush=True)
                m = FailureMemoMethod(configs, bname, "dynamatic", B)
                run_single(m, synth, logger, tool="dynamatic",
                           ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)


def main():
    parser = argparse.ArgumentParser(description="E1: Main results")
    parser.add_argument("--phase", choices=["1", "2"], required=True)
    parser.add_argument("--tool", choices=["bambu", "dynamatic", "both"], default="both")
    args = parser.parse_args()

    logger = ExperimentLogger()

    if args.phase == "1":
        if args.tool in ("bambu", "both"):
            print("=== Phase 1: Bambu (B=60) ===", flush=True)
            run_bambu_phase(BAMBU, BAMBU_PRIMARY, logger, include_appendix=False)
        if args.tool in ("dynamatic", "both"):
            print("=== Phase 1: Dynamatic (B=30) ===", flush=True)
            run_dynamatic_phase(DYNAMATIC, DYNAMATIC_PRIMARY, logger, include_appendix=False)
    else:
        if args.tool in ("bambu", "both"):
            print("=== Phase 2: Bambu sweep ===", flush=True)
            run_bambu_phase(BAMBU, BAMBU_SWEEP, logger, include_appendix=True)
        if args.tool in ("dynamatic", "both"):
            print("=== Phase 2: Dynamatic sweep ===", flush=True)
            run_dynamatic_phase(DYNAMATIC, DYNAMATIC_SWEEP, logger, include_appendix=True)

    print(f"Done. Logs → {logger.output_dir}")


if __name__ == "__main__":
    main()
