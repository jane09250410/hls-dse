#!/usr/bin/env python3
"""
run_ablation.py — Component ablation (Table IV).

Updated 2026-05-06:
  - Optional CC variant: --beta-cov 0.2 reproduces ablation with the
    categorical-coverage extension active. Default β_cov=0 = vanilla.
  - --n-perms controls the number of queue permutations per cell.
  - --tool to pick bambu / dynamatic / both.

Usage:
    # Reproduce paper Table IV (vanilla)
    python3 scripts/runners/run_ablation.py --tool both --n-perms 10
    # Ablation with CC variant for comparison
    python3 scripts/runners/run_ablation.py --tool both --beta-cov 0.2
"""

import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config_generator import generate_bambu_configs
from dynamatic_config_generator import generate_dynamatic_configs
from exp_logging.experiment_logger import ExperimentLogger
from runners.run_main_results import (BAMBU, DYNAMATIC,
                                       make_bambu_synth, make_dynamatic_synth)
from runners.run_single import run_single
from methods.pa_dse_method import PADSEMethod

ABLATION_CONFIGS = [
    "no-filter", "phago-only", "phago+RPE", "phago+OFRS",
    "phago+Full", "DFRL-only", "phago+RPE-reorder", "phago+OFRS-skip",
]


def run_ablation_for_tool(tool, benchmarks, configs, B, args, logger):
    for bname, binfo in benchmarks.items():
        src = binfo["src"]; top = binfo["top"]
        if tool == "bambu":
            base = f"results/experiments/ablation_bambu/{bname}/B{B}"
            synth = make_bambu_synth(src, top, base)
        else:
            base = f"results/experiments/ablation_dynamatic/{bname}/B{B}"
            synth = make_dynamatic_synth(src, top, base)

        for abl in ABLATION_CONFIGS:
            for pid in range(args.n_perms):
                tag = f"β={args.beta_cov}" if args.beta_cov > 0 else "vanilla"
                print(f"  {abl}[{tag}] / {bname} / B={B} / perm={pid}", flush=True)
                m = PADSEMethod(configs, bname, tool, B,
                                ablation_config=abl,
                                source_path=src,
                                queue_permutation_id=pid,
                                beta_cov=args.beta_cov, n_cov=args.n_cov)
                run_single(m, synth, logger, tool=tool,
                           ablation_config=abl,
                           queue_permutation_id=pid)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tool", choices=["bambu", "dynamatic", "both"], default="both")
    p.add_argument("--n-perms", type=int, default=10)
    p.add_argument("--beta-cov", type=float, default=0.0,
                   help="0=vanilla ablation; 0.2=ablation with CC")
    p.add_argument("--n-cov", type=int, default=2)
    p.add_argument("--budget", type=int, default=30)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    log_kwargs = {}
    if args.output_dir:
        log_kwargs["output_dir"] = args.output_dir
    logger = ExperimentLogger(**log_kwargs)

    if args.tool in ("bambu", "both"):
        configs = generate_bambu_configs(enable_pipeline=True)
        run_ablation_for_tool("bambu", BAMBU, configs, args.budget, args, logger)

    if args.tool in ("dynamatic", "both"):
        configs = generate_dynamatic_configs()
        run_ablation_for_tool("dynamatic", DYNAMATIC, configs, args.budget,
                               args, logger)

    print(f"\nDone. Logs → {logger.output_dir}")


if __name__ == "__main__":
    main()
