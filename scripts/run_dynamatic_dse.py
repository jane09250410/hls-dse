#!/usr/bin/env python3
"""
run_dynamatic_dse.py
====================
PA-DSE experiment runner for Dynamatic.

Runs PA-DSE and baseline strategies (Grid, Random, LHS) on Dynamatic benchmarks.
Parallel to the Bambu version (run_pa_dse.py) but adapted for Dynamatic's
parameter space and synthesis flow.

Usage:
    cd ~/hls-dse
    python3 scripts/run_dynamatic_dse.py --benchmark fir --budget 30
    python3 scripts/run_dynamatic_dse.py --benchmark fir --budget 30 --ablation no_filter
    python3 scripts/run_dynamatic_dse.py --benchmark all --budget 30
"""

import os
import sys
import csv
import time
import random
import argparse
import shutil
from typing import Any, Dict, List, Optional

import numpy as np

from dynamatic_config_generator import (
    generate_dynamatic_configs,
    config_to_label,
    dynamatic_static_rules,
    apply_static_rules,
    analyze_source_for_dynamatic,
    Config,
)
from run_dynamatic_single import run_dynamatic_single, classify_dynamatic_error

# ============================================================
# Benchmark Registry (Dynamatic benchmarks)
# ============================================================

DYNAMATIC_PATH = os.path.expanduser("~/dynamatic")

BENCHMARK_DEFAULTS = {
    "fir": {
        "src": os.path.join(DYNAMATIC_PATH, "integration-test/fir/fir.c"),
        "top_func": "fir",
    },
    "gcd": {
        "src": os.path.join(DYNAMATIC_PATH, "integration-test/gcd/gcd.c"),
        "top_func": "gcd",
    },
    "iir": {
        "src": os.path.join(DYNAMATIC_PATH, "integration-test/iir/iir.c"),
        "top_func": "iir",
    },
    "triangular": {
        "src": os.path.join(DYNAMATIC_PATH, "integration-test/triangular/triangular.c"),
        "top_func": "triangular",
    },
    "pivot": {
        "src": os.path.join(DYNAMATIC_PATH, "integration-test/pivot/pivot.c"),
        "top_func": "pivot",
    },
    "histogram": {
        "src": os.path.join(DYNAMATIC_PATH, "integration-test/histogram/histogram.c"),
        "top_func": "histogram",
    },
    "binary_search": {
        "src": os.path.join(DYNAMATIC_PATH, "integration-test/binary_search/binary_search.c"),
        "top_func": "binary_search",
    },
    "gaussian": {
        "src": os.path.join(DYNAMATIC_PATH, "integration-test/gaussian/gaussian.c"),
        "top_func": "gaussian",
    },
    "matching": {
        "src": os.path.join(DYNAMATIC_PATH, "integration-test/matching/matching.c"),
        "top_func": "matching",
    },
}


# ============================================================
# Sampling Strategies
# ============================================================

def grid_search_sample(configs: List[Config], budget: int) -> List[Config]:
    """Uniform grid sampling."""
    if budget >= len(configs):
        return list(configs)
    step = max(1, len(configs) // budget)
    return [configs[i] for i in range(0, len(configs), step)][:budget]


def random_sample(configs: List[Config], budget: int, seed: int = 42) -> List[Config]:
    """Random sampling."""
    rng = random.Random(seed)
    if budget >= len(configs):
        return list(configs)
    return rng.sample(configs, budget)


def lhs_sample(configs: List[Config], budget: int, seed: int = 42) -> List[Config]:
    """Latin Hypercube Sampling over the config space."""
    rng = random.Random(seed)
    n = len(configs)
    if budget >= n:
        return list(configs)

    # Simple approach: divide into budget strata, pick one from each
    strata_size = n / budget
    selected = []
    for i in range(budget):
        start = int(i * strata_size)
        end = int((i + 1) * strata_size)
        idx = rng.randint(start, min(end - 1, n - 1))
        selected.append(configs[idx])
    return selected


# ============================================================
# Online Pattern Learner (Autophagy) for Dynamatic
# ============================================================

class DynamaticPatternLearner:
    """
    Learns failure patterns from Dynamatic synthesis results.
    When threshold failures of the same type share common parameters,
    generates a suppression rule.
    """

    def __init__(self, threshold: int = 2):
        self.threshold = threshold
        self.failure_log = []
        self.learned_patterns = []

    def add_failure(self, config: Config, error_type: str) -> Optional[Dict]:
        """Record a failure and check if a new pattern emerges."""
        self.failure_log.append({
            "config": config,
            "error_type": error_type,
        })

        # Group recent failures by error type
        same_type = [f for f in self.failure_log if f["error_type"] == error_type]
        if len(same_type) < self.threshold:
            return None

        # Find common parameters among recent failures of this type
        recent = same_type[-self.threshold:]
        common = {}

        # Check each parameter for commonality
        param_keys = ["clock_period", "buffer_algorithm", "sharing", "disable_lsq", "fast_token_delivery"]
        for key in param_keys:
            values = set(f["config"][key] for f in recent)
            if len(values) == 1:
                common[key] = values.pop()

        if not common:
            return None

        # Create pattern
        pattern = {
            "error_type": error_type,
            "common_params": common,
            "support": len(same_type),
        }
        self.learned_patterns.append(pattern)
        return pattern

    def should_suppress(self, config: Config) -> Optional[Dict]:
        """Check if a config matches any learned pattern."""
        for pattern in self.learned_patterns:
            match = all(
                config.get(k) == v
                for k, v in pattern["common_params"].items()
            )
            if match:
                return pattern
        return None


# ============================================================
# PA-DSE for Dynamatic
# ============================================================

def run_padse_dynamatic(
    configs: List[Config],
    src_file: str,
    top_func: str,
    benchmark_name: str,
    budget: int,
    results_dir: str,
    ablation_mode: str = "full",
) -> Dict[str, Any]:
    """
    Run PA-DSE exploration on Dynamatic.

    ablation_mode:
        "full"        - Phagocytosis + Autophagy (default)
        "static_only" - Phagocytosis only, no Autophagy
        "no_filter"   - No filtering at all
    """
    print(f"PA-DSE / {benchmark_name}")

    # Phase 1: Phagocytosis (static filtering)
    if ablation_mode == "no_filter":
        active = list(configs)
        blocked = []
        suppressed = []
        static_log = []
    else:
        rules = dynamatic_static_rules()

        # Add code-aware rules based on source analysis
        code_info = analyze_source_for_dynamatic(src_file)
        if code_info.get("has_while_loop"):
            rules.append({
                "name": "R_CA1_while_loop_complex_buffer",
                "severity": "suppress",
                "description": "While loops with complex buffer algorithms may cause MILP issues",
                "condition": lambda c: c["buffer_algorithm"] in ("fpga20", "fpl22"),
            })

        active, blocked, suppressed, static_log = apply_static_rules(configs, rules)

    print(f"  Initial configs:   {len(configs)}")
    print(f"  Static blocked:    {len(blocked)}")
    print(f"  Static suppressed: {len(suppressed)}")
    print(f"  Active queue:      {len(active)}")

    # Build exploration queue: active first, then suppressed
    queue = list(active) + list(suppressed)

    # Phase 2: Iterative exploration with Autophagy
    learner = DynamaticPatternLearner(threshold=2)
    results = []
    autophagy_suppressed = 0
    n_eval = 0

    for cfg in queue:
        if n_eval >= budget:
            break

        # Check autophagy suppression (only in "full" mode)
        if ablation_mode == "full":
            pattern = learner.should_suppress(cfg)
            if pattern is not None:
                autophagy_suppressed += 1
                continue

        n_eval += 1
        label = config_to_label(cfg)
        print(f"  [{n_eval}/{budget}] Config {cfg['id']}: {label}")

        success, metrics, output = run_dynamatic_single(
            config=cfg,
            src_file=src_file,
            top_func=top_func,
        )

        metrics["benchmark"] = benchmark_name
        results.append(metrics)

        if success:
            print(f"       -> OK (components={metrics['num_components']}, "
                  f"buffers={metrics['num_buffers']}, time={metrics['compile_time_s']:.1f}s)")
        else:
            error = metrics["error_type"]
            print(f"       -> FAILED ({metrics['compile_time_s']:.1f}s), error={error}")

            # Autophagy: learn from failure
            if ablation_mode == "full":
                pattern = learner.add_failure(cfg, error)
                if pattern is not None:
                    # Count how many remaining queue items match
                    remaining = queue[queue.index(cfg) + 1:]
                    matched = sum(1 for c in remaining if learner.should_suppress(c) is not None)
                    print(f"       -> AUTOPHAGY: learned pattern {pattern['common_params']}")
                    print(f"       -> AUTOPHAGY: suppressed {matched} future configs")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    save_results(results, results_dir, "PA_DSE")

    # Compute stats
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    stats = {
        "strategy": "PA-DSE",
        "total_evaluated": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate_pct": round(100 * len(successful) / max(1, len(results)), 1),
        "total_time_s": round(sum(r["compile_time_s"] for r in results), 1),
        "best_components": min((r["num_components"] for r in successful), default=0),
        "mean_components": round(np.mean([r["num_components"] for r in successful]), 1) if successful else 0,
        "best_buffers": min((r["num_buffers"] for r in successful), default=0),
        "mean_buffers": round(np.mean([r["num_buffers"] for r in successful]), 1) if successful else 0,
        "benchmark": benchmark_name,
        "initial_configs": len(configs),
        "static_blocked": len(blocked),
        "static_suppressed": len(suppressed),
        "autophagy_suppressed": autophagy_suppressed,
        "actual_runs": len(results),
        "ablation_mode": ablation_mode,
    }

    # Save static pruning log
    if static_log:
        log_path = os.path.join(results_dir, "pa_dse_static_pruning_log.csv")
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=static_log[0].keys())
            writer.writeheader()
            writer.writerows(static_log)

    # Save learned rules
    if learner.learned_patterns:
        rules_path = os.path.join(results_dir, "pa_dse_learned_rules.txt")
        with open(rules_path, "w") as f:
            for p in learner.learned_patterns:
                f.write(f"{p['error_type']}: {p['common_params']} (support={p['support']})\n")

    return stats


def run_baseline_strategy(
    strategy_name: str,
    configs: List[Config],
    src_file: str,
    top_func: str,
    benchmark_name: str,
    budget: int,
    results_dir: str,
) -> Dict[str, Any]:
    """Run a baseline strategy (Grid, Random, LHS)."""

    if strategy_name == "Grid Search":
        sample = grid_search_sample(configs, budget)
    elif strategy_name.startswith("Random"):
        sample = random_sample(configs, budget)
    elif strategy_name.startswith("LHS"):
        sample = lhs_sample(configs, budget)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    print(f"\n{'=' * 60}")
    print(f"  Strategy: {strategy_name} ({len(sample)} configs)")
    print(f"{'=' * 60}")

    results = []
    for i, cfg in enumerate(sample):
        label = config_to_label(cfg)
        print(f"  [{i + 1}/{len(sample)}] Config {cfg['id']}: {label}")

        success, metrics, output = run_dynamatic_single(
            config=cfg,
            src_file=src_file,
            top_func=top_func,
        )
        metrics["benchmark"] = benchmark_name
        results.append(metrics)

        if success:
            print(f"       -> OK (components={metrics['num_components']}, time={metrics['compile_time_s']:.1f}s)")
        else:
            print(f"       -> FAILED ({metrics['compile_time_s']:.1f}s)")

    save_results(results, results_dir, strategy_name.replace(" ", "_"))

    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    return {
        "strategy": strategy_name,
        "total_evaluated": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate_pct": round(100 * len(successful) / max(1, len(results)), 1),
        "total_time_s": round(sum(r["compile_time_s"] for r in results), 1),
        "best_components": min((r["num_components"] for r in successful), default=0),
        "mean_components": round(np.mean([r["num_components"] for r in successful]), 1) if successful else 0,
        "best_buffers": min((r["num_buffers"] for r in successful), default=0),
        "mean_buffers": round(np.mean([r["num_buffers"] for r in successful]), 1) if successful else 0,
        "benchmark": benchmark_name,
    }


# ============================================================
# Results I/O
# ============================================================

def save_results(results: List[Dict], results_dir: str, strategy_name: str):
    """Save results as CSV."""
    if not results:
        return
    path = os.path.join(results_dir, f"{strategy_name}_results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {path}")


def save_comparison_stats(all_stats: List[Dict], results_dir: str):
    """Save comparison stats as CSV."""
    if not all_stats:
        return
    path = os.path.join(results_dir, "pa_dse_comparison_stats.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_stats[0].keys(), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_stats)
    print(f"Saved {path}")


# ============================================================
# Main
# ============================================================

def run_all(benchmark: str, budget: int, results_root: str, ablation_mode: str = "full"):
    """Run full comparison for one benchmark."""
    info = BENCHMARK_DEFAULTS[benchmark]
    src_file = info["src"]
    top_func = info["top_func"]

    if not os.path.isfile(src_file):
        print(f"[ERROR] Source file not found: {src_file}")
        return

    print("=" * 60)
    print(f"PA-DSE comparison on benchmark: {benchmark}")
    print(f"Source file: {src_file}")
    print(f"Top function: {top_func}")
    print(f"Ablation mode: {ablation_mode}")
    print("=" * 60)

    # Generate configs
    configs = generate_dynamatic_configs()
    print(f"Total design points: {len(configs)}")

    # Set up results directory
    mode_suffix = "" if ablation_mode == "full" else f"_{ablation_mode}"
    results_dir = os.path.join(results_root, benchmark, f"dynamatic{mode_suffix}")
    os.makedirs(results_dir, exist_ok=True)

    all_stats = []

    # Run baselines
    for strategy in ["Grid Search", "Random 30%", "LHS 20%"]:
        stats = run_baseline_strategy(
            strategy_name=strategy,
            configs=configs,
            src_file=src_file,
            top_func=top_func,
            benchmark_name=benchmark,
            budget=budget,
            results_dir=results_dir,
        )
        all_stats.append(stats)

    # Run PA-DSE
    padse_stats = run_padse_dynamatic(
        configs=configs,
        src_file=src_file,
        top_func=top_func,
        benchmark_name=benchmark,
        budget=budget,
        results_dir=results_dir,
        ablation_mode=ablation_mode,
    )
    all_stats.append(padse_stats)

    # Print summary
    print(f"\n{'=' * 60}")
    print("=== Summary ===")
    header = f"{'strategy':>15s}  {'eval':>5s}  {'succ':>5s}  {'fail':>5s}  {'SR%':>6s}  {'time':>8s}  {'best_comp':>10s}  {'mean_comp':>10s}"
    print(header)
    for s in all_stats:
        line = (
            f"{s['strategy']:>15s}  "
            f"{s['total_evaluated']:>5d}  "
            f"{s['successful']:>5d}  "
            f"{s['failed']:>5d}  "
            f"{s['success_rate_pct']:>6.1f}  "
            f"{s['total_time_s']:>8.1f}  "
            f"{s['best_components']:>10d}  "
            f"{s['mean_components']:>10.1f}"
        )
        print(line)

    save_comparison_stats(all_stats, results_dir)


def main():
    parser = argparse.ArgumentParser(description="PA-DSE for Dynamatic")
    parser.add_argument(
        "--benchmark", type=str, required=True,
        choices=list(BENCHMARK_DEFAULTS.keys()) + ["all"],
        help="Benchmark name",
    )
    parser.add_argument("--budget", type=int, default=30, help="Evaluation budget (default: 30)")
    parser.add_argument(
        "--ablation", type=str, default="full",
        choices=["no_filter", "static_only", "full"],
        help="Ablation mode",
    )
    parser.add_argument(
        "--results-root", type=str, default="results/pa_dse",
        help="Results root directory",
    )

    args = parser.parse_args()

    if args.benchmark == "all":
        benchmarks = list(BENCHMARK_DEFAULTS.keys())
    else:
        benchmarks = [args.benchmark]

    for bm in benchmarks:
        run_all(
            benchmark=bm,
            budget=args.budget,
            results_root=args.results_root,
            ablation_mode=args.ablation,
        )


if __name__ == "__main__":
    main()
