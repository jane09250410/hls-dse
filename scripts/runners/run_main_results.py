#!/usr/bin/env python3
"""
run_main_results.py — Main results experiment driver (Tables II / III).

Updated 2026-05-06:
  - Full benchmark coverage (8 Bambu, 4 Dynamatic) matching paper §IV-A.
  - PA-DSE now runs N_PERMS queue permutations (default 10) per benchmark.
  - --variant flag: vanilla | cc | both
        vanilla = β_cov=0   (paper baseline)
        cc      = β_cov=0.2 (paper PA-DSE-CC)
        both    = run both
  - --beta-cov / --n-cov override defaults if you want to sweep.

Usage:
    python3 scripts/runners/run_main_results.py --tool bambu --variant cc
    python3 scripts/runners/run_main_results.py --tool dynamatic --variant both
    python3 scripts/runners/run_main_results.py --tool both --variant cc --include-appendix
"""

import argparse, os, subprocess, sys, time, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config_generator import generate_bambu_configs, config_to_bambu_cmd
from exp_logging.experiment_logger import ExperimentLogger
from runners.run_single import run_single
from methods.baseline_methods import (
    RandomMethod, FilteredRandomMethod, GridMethod, LHSMethod, FailureMemoMethod)
from methods.advanced_baselines import (
    SimulatedAnnealingMethod, GeneticAlgorithmMethod,
    GPBayesOptMethod, RFClassifierMethod)
from methods.pa_dse_method import PADSEMethod

# ── Frozen config (paper §IV-A) ────────────────────────────────

BAMBU = {
    "matmul":    {"src": "benchmarks/matmul/matmul.c",       "top": "matmul"},
    "vadd":      {"src": "benchmarks/vadd/vadd.c",           "top": "vadd"},
    "fir":       {"src": "benchmarks/fir/fir.c",             "top": "fir"},
    "histogram": {"src": "benchmarks/histogram/histogram.c", "top": "histogram"},
    "atax":      {"src": "benchmarks/atax/atax.c",           "top": "atax"},
    "bicg":      {"src": "benchmarks/bicg/bicg.c",           "top": "bicg"},
    "gemm":      {"src": "benchmarks/gemm/gemm.c",           "top": "gemm"},
    "gesummv":   {"src": "benchmarks/gesummv/gesummv.c",     "top": "gesummv"},
}
BAMBU_PRIMARY = [60]
BAMBU_SWEEP   = [20, 40, 60, 80]

DYNAMATIC_PATH = os.path.expanduser("~/dynamatic")
DYNAMATIC = {
    "gcd":           {"src": f"{DYNAMATIC_PATH}/integration-test/gcd/gcd.c",                     "top": "gcd"},
    "matching":      {"src": f"{DYNAMATIC_PATH}/integration-test/matching/matching.c",            "top": "matching"},
    "binary_search": {"src": f"{DYNAMATIC_PATH}/integration-test/binary_search/binary_search.c",  "top": "binary_search"},
    "kernel_2mm":    {"src": f"{DYNAMATIC_PATH}/integration-test/kernel_2mm/kernel_2mm.c",        "top": "kernel_2mm"},
}
DYNAMATIC_PRIMARY = [30]
DYNAMATIC_SWEEP   = [20, 30, 40, 60]

N_SEEDS = 10  # baseline seeds (paper used 10)
N_PERMS = 10  # PA-DSE queue permutations (paper used 10)

# Default CC hyperparams (frozen, do not tune per benchmark)
CC_BETA_DEFAULT = 0.2
CC_NCOV_DEFAULT = 2


# ── Synthesis wrappers ─────────────────────────────────────────

def make_bambu_synth(src, top, results_base):
    """Bambu synthesis function — returns (output, elapsed, success)."""
    def synthesize(config):
        cmd = config_to_bambu_cmd(config, src, top)
        work_dir = os.path.join(results_base, f"cfg_{config['id']}")
        os.makedirs(work_dir, exist_ok=True)
        t0 = time.time()
        try:
            res = subprocess.run(cmd, cwd=work_dir, capture_output=True,
                                 text=True, timeout=120)
            elapsed = time.time() - t0
            output = (res.stdout or "") + (res.stderr or "")
            success = (res.returncode == 0
                       and "Total estimated area" in output)
            return output, elapsed, success
        except subprocess.TimeoutExpired:
            return "TIMEOUT", time.time() - t0, False
    return synthesize


def make_dynamatic_synth(src, top, results_base):
    """Dynamatic synthesis function — isolated tmp dir per config."""
    from run_dynamatic_single import run_dynamatic_single
    import shutil

    def synthesize(config):
        work_dir = os.path.join(results_base, f"cfg_{config['id']}")
        os.makedirs(work_dir, exist_ok=True)
        local_src = os.path.join(work_dir, os.path.basename(src))
        if not os.path.exists(local_src):
            shutil.copy(src, local_src)
        return run_dynamatic_single(config, local_src, top, work_dir)
    return synthesize


# ── PA-DSE variants ────────────────────────────────────────────

def padse_variants(args):
    """Yield (variant_label, beta_cov, n_cov) for each PA-DSE variant to run."""
    if args.variant == "vanilla":
        yield ("vanilla", 0.0, args.n_cov)
    elif args.variant == "cc":
        yield ("cc", args.beta_cov, args.n_cov)
    elif args.variant == "both":
        yield ("vanilla", 0.0, args.n_cov)
        yield ("cc", args.beta_cov, args.n_cov)


# ── Bambu phase ────────────────────────────────────────────────

def run_bambu_phase(benchmarks, budgets, logger, args):
    configs = generate_bambu_configs(enable_pipeline=True)

    for bname, binfo in benchmarks.items():
        src, top = binfo["src"], binfo["top"]

        for B in budgets:
            base = f"results/experiments/bambu/{bname}/B{B}"
            synth = make_bambu_synth(src, top, base)

            # ── Stochastic baselines ───────────────────────────
            for seed in range(N_SEEDS):
                print(f"  Random / {bname} / B={B} / seed={seed}", flush=True)
                m = RandomMethod(configs, bname, "bambu", B, seed=seed)
                run_single(m, synth, logger, tool="bambu",
                           ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)

            for seed in range(N_SEEDS):
                print(f"  Filtered_Random / {bname} / B={B} / seed={seed}", flush=True)
                m = FilteredRandomMethod(configs, bname, "bambu", B,
                                         seed=seed, source_path=src)
                run_single(m, synth, logger, tool="bambu",
                           ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)

            if args.include_advanced_baselines:
                for seed in range(N_SEEDS):
                    print(f"  SA / {bname} / B={B} / seed={seed}", flush=True)
                    m = SimulatedAnnealingMethod(configs, bname, "bambu", B, seed=seed)
                    run_single(m, synth, logger, tool="bambu",
                               ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)

                    print(f"  GA / {bname} / B={B} / seed={seed}", flush=True)
                    m = GeneticAlgorithmMethod(configs, bname, "bambu", B, seed=seed)
                    run_single(m, synth, logger, tool="bambu",
                               ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)

                    print(f"  GP-BO / {bname} / B={B} / seed={seed}", flush=True)
                    m = GPBayesOptMethod(configs, bname, "bambu", B, seed=seed)
                    run_single(m, synth, logger, tool="bambu",
                               ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)

                    print(f"  RF / {bname} / B={B} / seed={seed}", flush=True)
                    m = RFClassifierMethod(configs, bname, "bambu", B, seed=seed)
                    run_single(m, synth, logger, tool="bambu",
                               ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)

            # ── PA-DSE (every variant × every queue permutation) ──
            for variant, beta_cov, n_cov in padse_variants(args):
                for pid in range(N_PERMS):
                    print(f"  PA-DSE_Full[{variant},β={beta_cov}] / {bname} / B={B} / perm={pid}", flush=True)
                    m = PADSEMethod(configs, bname, "bambu", B,
                                    ablation_config="phago+Full",
                                    source_path=src,
                                    queue_permutation_id=pid,
                                    beta_cov=beta_cov, n_cov=n_cov)
                    run_single(m, synth, logger, tool="bambu",
                               queue_permutation_id=pid)

            # ── Appendix ────────────────────────────────────────
            if args.include_appendix:
                print(f"  Grid / {bname} / B={B}")
                m = GridMethod(configs, bname, "bambu", B)
                run_single(m, synth, logger, tool="bambu",
                           ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)


# ── Dynamatic phase ────────────────────────────────────────────

def run_dynamatic_phase(benchmarks, budgets, logger, args):
    from dynamatic_config_generator import generate_dynamatic_configs
    configs = generate_dynamatic_configs()

    for bname, binfo in benchmarks.items():
        src, top = binfo["src"], binfo["top"]

        for B in budgets:
            base = f"results/experiments/dynamatic/{bname}/B{B}"
            synth = make_dynamatic_synth(src, top, base)

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

            if args.include_advanced_baselines:
                for seed in range(N_SEEDS):
                    print(f"  SA / {bname} / B={B} / seed={seed}", flush=True)
                    m = SimulatedAnnealingMethod(configs, bname, "dynamatic", B, seed=seed)
                    run_single(m, synth, logger, tool="dynamatic",
                               ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)

                    print(f"  GA / {bname} / B={B} / seed={seed}", flush=True)
                    m = GeneticAlgorithmMethod(configs, bname, "dynamatic", B, seed=seed)
                    run_single(m, synth, logger, tool="dynamatic",
                               ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)

                    print(f"  GP-BO / {bname} / B={B} / seed={seed}", flush=True)
                    m = GPBayesOptMethod(configs, bname, "dynamatic", B, seed=seed)
                    run_single(m, synth, logger, tool="dynamatic",
                               ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)

                    print(f"  RF / {bname} / B={B} / seed={seed}", flush=True)
                    m = RFClassifierMethod(configs, bname, "dynamatic", B, seed=seed)
                    run_single(m, synth, logger, tool="dynamatic",
                               ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)

            for variant, beta_cov, n_cov in padse_variants(args):
                for pid in range(N_PERMS):
                    print(f"  PA-DSE_Full[{variant},β={beta_cov}] / {bname} / B={B} / perm={pid}", flush=True)
                    m = PADSEMethod(configs, bname, "dynamatic", B,
                                    ablation_config="phago+Full",
                                    source_path=src,
                                    queue_permutation_id=pid,
                                    beta_cov=beta_cov, n_cov=n_cov)
                    run_single(m, synth, logger, tool="dynamatic",
                               queue_permutation_id=pid)

            if args.include_appendix:
                print(f"  Grid / {bname} / B={B}", flush=True)
                m = GridMethod(configs, bname, "dynamatic", B)
                run_single(m, synth, logger, tool="dynamatic",
                           ablation_config="N/A", tau=0, theta=0, n_min=0, p_probe=0)


# ── CLI ────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tool", choices=["bambu", "dynamatic", "both"], default="both")
    p.add_argument("--variant", choices=["vanilla", "cc", "both"], default="cc",
                   help="PA-DSE variant: vanilla=β=0; cc=β=0.2; both=run both")
    p.add_argument("--beta-cov", type=float, default=CC_BETA_DEFAULT)
    p.add_argument("--n-cov", type=int, default=CC_NCOV_DEFAULT)
    p.add_argument("--phase", type=int, default=1, help="1=primary B, 2=B sweep")
    p.add_argument("--include-advanced-baselines", action="store_true",
                   default=True, help="Run SA/GA/GP-BO/RF (default on)")
    p.add_argument("--no-advanced-baselines", dest="include_advanced_baselines",
                   action="store_false")
    p.add_argument("--include-appendix", action="store_true",
                   help="Run Grid offline reference")
    p.add_argument("--output-dir", default=None,
                   help="Override default logger output directory")
    args = p.parse_args()

    log_kwargs = {}
    if args.output_dir:
        log_kwargs["output_dir"] = args.output_dir
    logger = ExperimentLogger(**log_kwargs)

    bam_budgets = BAMBU_PRIMARY if args.phase == 1 else BAMBU_SWEEP
    dyn_budgets = DYNAMATIC_PRIMARY if args.phase == 1 else DYNAMATIC_SWEEP

    if args.tool in ("bambu", "both"):
        print("=" * 60)
        print(f"BAMBU phase {args.phase} | variant={args.variant} | "
              f"β_cov={args.beta_cov} | n_cov={args.n_cov}")
        print("=" * 60)
        run_bambu_phase(BAMBU, bam_budgets, logger, args)

    if args.tool in ("dynamatic", "both"):
        print("=" * 60)
        print(f"DYNAMATIC phase {args.phase} | variant={args.variant} | "
              f"β_cov={args.beta_cov} | n_cov={args.n_cov}")
        print("=" * 60)
        run_dynamatic_phase(DYNAMATIC, dyn_budgets, logger, args)

    print(f"\nDone. Logs → {logger.output_dir}")


if __name__ == "__main__":
    main()
