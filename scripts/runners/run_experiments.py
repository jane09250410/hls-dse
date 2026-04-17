#!/usr/bin/env python3
"""
run_experiments.py — E4c / E6 / E6a experiments.

Usage:
    python3 scripts/runners/run_experiments.py probe          # E4c
    python3 scripts/runners/run_experiments.py sensitivity theta    # E6 (main text)
    python3 scripts/runners/run_experiments.py sensitivity tau      # E6 (appendix)
    python3 scripts/runners/run_experiments.py sensitivity n_min    # E6 (appendix)
    python3 scripts/runners/run_experiments.py robustness     # E6a
"""

import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config_generator import generate_bambu_configs
from exp_logging.experiment_logger import ExperimentLogger
from runners.run_main_results import make_bambu_synth
from runners.run_single import run_single
from methods.pa_dse_method import PADSEMethod

# Frozen sensitivity benchmarks: matmul (high, concentrated), binary_search needs Dynamatic
SENSITIVITY_BENCH = {
    "matmul": {"src": "benchmarks/matmul/matmul.c", "top": "matmul", "B": 60},
}

ALL_BAMBU = {
    "matmul":    {"src": "benchmarks/matmul/matmul.c",       "top": "matmul",    "B": 60},
    "vadd":      {"src": "benchmarks/vadd/vadd.c",           "top": "vadd",      "B": 60},
    "fir":       {"src": "benchmarks/fir/fir.c",             "top": "fir",       "B": 60},
    "histogram": {"src": "benchmarks/histogram/histogram.c", "top": "histogram", "B": 60},
}


def cmd_probe(args):
    """E4c: Probe sensitivity. p_probe ∈ {0, 0.02, 0.05, 0.10, 0.20}."""
    configs = generate_bambu_configs(enable_pipeline=True)
    logger = ExperimentLogger()

    for bname, bi in SENSITIVITY_BENCH.items():
        for pp in [0, 0.02, 0.05, 0.10, 0.20]:
            print(f"  probe p={pp} / {bname}")
            synth = make_bambu_synth(bi["src"], bi["top"],
                                     f"results/experiments/probe/{bname}/p{pp}")
            m = PADSEMethod(configs, bname, "bambu", bi["B"],
                            ablation_config="phago+Full",
                            p_probe=pp, source_path=bi["src"])
            run_single(m, synth, logger, tool="bambu", p_probe=pp)

    print(f"Done. Logs → {logger.output_dir}")


def cmd_sensitivity(args):
    """E6: One-at-a-time parameter sensitivity."""
    GRIDS = {
        "theta": {"key": "theta", "values": [0.7, 0.8, 0.9]},
        "tau":   {"key": "tau",   "values": [2, 3, 4, 5]},
        "n_min": {"key": "n_min", "values": [3, 5, 8]},
    }
    grid = GRIDS[args.param]
    configs = generate_bambu_configs(enable_pipeline=True)
    logger = ExperimentLogger()

    for bname, bi in SENSITIVITY_BENCH.items():
        for val in grid["values"]:
            print(f"  {args.param}={val} / {bname}")
            synth = make_bambu_synth(bi["src"], bi["top"],
                f"results/experiments/sensitivity/{bname}/{args.param}_{val}")
            kwargs = {grid["key"]: val}
            m = PADSEMethod(configs, bname, "bambu", bi["B"],
                            ablation_config="phago+Full",
                            source_path=bi["src"], **kwargs)
            run_single(m, synth, logger, tool="bambu", **kwargs)

    print(f"Done. Logs → {logger.output_dir}")


def cmd_robustness(args):
    """E6a: Queue-order robustness. 5 permutations × all Bambu benchmarks."""
    configs = generate_bambu_configs(enable_pipeline=True)
    logger = ExperimentLogger()

    for bname, bi in ALL_BAMBU.items():
        for perm in range(5):
            print(f"  perm={perm} / {bname}")
            synth = make_bambu_synth(bi["src"], bi["top"],
                f"results/experiments/robustness/{bname}/perm{perm}")
            m = PADSEMethod(configs, bname, "bambu", bi["B"],
                            ablation_config="phago+Full",
                            source_path=bi["src"],
                            queue_permutation_id=perm)
            run_single(m, synth, logger, tool="bambu",
                       queue_permutation_id=perm)

    print(f"Done. Logs → {logger.output_dir}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("probe", help="E4c: probe sensitivity")

    p_sens = sub.add_parser("sensitivity", help="E6: parameter sensitivity")
    p_sens.add_argument("param", choices=["theta", "tau", "n_min"])

    sub.add_parser("robustness", help="E6a: queue-order robustness")

    args = parser.parse_args()
    {"probe": cmd_probe, "sensitivity": cmd_sensitivity,
     "robustness": cmd_robustness}[args.cmd](args)


if __name__ == "__main__":
    main()
