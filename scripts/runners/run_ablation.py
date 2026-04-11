#!/usr/bin/env python3
"""
run_ablation.py — E3: Component Ablation.

8 ablation configs × representative benchmarks (default) or all 4 Bambu benchmarks.

Usage:
    python3 scripts/runners/run_ablation.py                    # 5 representative
    python3 scripts/runners/run_ablation.py --all-benchmarks   # all 4 Bambu
"""

import argparse, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config_generator import generate_bambu_configs
from logging.experiment_logger import ExperimentLogger
from runners.run_main_results import make_bambu_synth
from runners.run_single import run_single
from methods.pa_dse_method import PADSEMethod

ABLATION_CONFIGS = [
    "no-filter", "phago-only", "phago+RPE", "phago+OFRS",
    "phago+Full", "DFRL-only", "phago+RPE-reorder", "phago+OFRS-skip",
]

# Representative Bambu benchmarks (by infeasibility group)
REPRESENTATIVE = {
    "matmul": {"src": "benchmarks/matmul/matmul.c", "top": "matmul", "B": 60},     # high
    "vadd":   {"src": "benchmarks/vadd/vadd.c",     "top": "vadd",   "B": 60},     # medium
}

ALL_BAMBU = {
    "matmul":    {"src": "benchmarks/matmul/matmul.c",       "top": "matmul",    "B": 60},
    "vadd":      {"src": "benchmarks/vadd/vadd.c",           "top": "vadd",      "B": 60},
    "fir":       {"src": "benchmarks/fir/fir.c",             "top": "fir",       "B": 60},
    "histogram": {"src": "benchmarks/histogram/histogram.c", "top": "histogram", "B": 60},
}


def main():
    parser = argparse.ArgumentParser(description="E3: Ablation")
    parser.add_argument("--all-benchmarks", action="store_true")
    args = parser.parse_args()

    benchmarks = ALL_BAMBU if args.all_benchmarks else REPRESENTATIVE
    configs = generate_bambu_configs(enable_pipeline=True)
    logger = ExperimentLogger()

    for bname, binfo in benchmarks.items():
        B = binfo["B"]
        synth = make_bambu_synth(binfo["src"], binfo["top"],
                                 f"results/experiments/ablation/{bname}/B{B}")

        for abl in ABLATION_CONFIGS:
            print(f"  {abl} / {bname} / B={B}")
            m = PADSEMethod(configs, bname, "bambu", B,
                            ablation_config=abl, source_path=binfo["src"])
            run_single(m, synth, logger, tool="bambu", ablation_config=abl)

    print(f"Done. Logs → {logger.output_dir}")


if __name__ == "__main__":
    main()
