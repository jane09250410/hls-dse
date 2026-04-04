#!/usr/bin/env python3
"""
pa_dse.py

Minimal runnable PA-DSE implementation:
- Phagocytosis: static feasibility filtering
- Autophagy L1: online recurrent failure-pattern pruning
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import time
import pandas as pd

from feasibility_filter import phagocytosis, default_static_rules
from pattern_learner import FailurePatternLearner, LearnedPattern
from config_generator import config_to_bambu_cmd
from run_exploration import run_bambu_single, extract_bambu_metrics
from analyze import find_pareto


Config = Dict[str, Any]


@dataclass
class PADSEResult:
    all_results: List[Dict[str, Any]]
    stats: Dict[str, Any]
    static_blocked: List[Config]
    static_suppressed: List[Config]
    static_log: List[Dict[str, Any]]
    learned_rules: List[str]
    autophagy_suppressed_ids: List[int]


class PADSE:
    def __init__(
        self,
        src_file: str,
        top_func: str,
        benchmark_name: str,
        results_dir: str = "results/pa_dse",
        budget: Optional[int] = None,
        autophagy_threshold: int = 2
    ) -> None:
        self.src_file = src_file
        self.top_func = top_func
        self.benchmark_name = benchmark_name
        self.results_dir = results_dir
        self.budget = budget
        self.autophagy_threshold = autophagy_threshold
        self.learner = FailurePatternLearner(threshold=autophagy_threshold)

    def _suppress_matching_configs(
        self,
        active_configs: List[Config],
        pattern: LearnedPattern
    ) -> Tuple[List[Config], List[int]]:
        kept = []
        suppressed = []
        for cfg in active_configs:
            if pattern.matches(cfg, self.benchmark_name):
                suppressed.append(int(cfg.get("id", -1)))
            else:
                kept.append(cfg)
        return kept, suppressed

    def explore(self, configs: List[Config]) -> PADSEResult:
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)

        active_configs, static_blocked, static_suppressed, static_log = phagocytosis(
            configs=configs,
            rules=default_static_rules(),
            source_path=self.src_file,
            benchmark_name=self.benchmark_name,
        )

        queue = list(active_configs) + list(static_suppressed)

        print(f"PA-DSE / {self.benchmark_name}")
        print(f"  Initial configs:   {len(configs)}")
        print(f"  Static blocked:    {len(static_blocked)}")
        print(f"  Static suppressed: {len(static_suppressed)}")
        print(f"  Active queue:      {len(queue)}")

        if self.budget is None:
            budget = len(queue)
        else:
            budget = min(self.budget, len(queue))

        all_results: List[Dict[str, Any]] = []
        autophagy_suppressed_ids: List[int] = []
        learned_rules: List[str] = []
        start_time = time.time()

        abs_src = os.path.abspath(self.src_file)
        runs = 0

        while queue and runs < budget:
            cfg = queue.pop(0)
            runs += 1

            pipe_desc = "off"
            if cfg.get("pipeline"):
                pipe_desc = f"ii={cfg.get('pipeline_ii')}"

            cmd = config_to_bambu_cmd(cfg, abs_src, self.top_func)
            work_dir = os.path.join(self.results_dir, f"config_{cfg['id']}")

            print(
                f"  [{runs}/{budget}] Config {cfg['id']}: "
                f"clock={cfg.get('clock_period')}ns, "
                f"pipe={pipe_desc}, "
                f"mem={cfg.get('memory_policy')}, "
                f"ch={cfg.get('channels_type')}, "
                f"ch_num={cfg.get('channels_number')}"
            )

            output, elapsed, success = run_bambu_single(cmd, work_dir)
            metrics = extract_bambu_metrics(output)

            result = {
                **cfg,
                **metrics,
                "runtime_s": round(elapsed, 2),
                "success": bool(success),
                "strategy": "PA-DSE",
                "benchmark": self.benchmark_name,
                "error_type": None,
                "pruned_by_autophagy": False,
            }

            if success:
                print(
                    f"       -> area={metrics.get('total_area')}, "
                    f"states={metrics.get('num_states')}, "
                    f"freq={metrics.get('max_freq_mhz')}MHz"
                )
            else:
                pattern = self.learner.add_failure(
                    config=cfg,
                    output=output,
                    runtime_s=elapsed,
                    benchmark_name=self.benchmark_name,
                )
                result["error_type"] = self.learner.failure_log[-1].error_type
                print(f"       -> FAILED ({elapsed:.1f}s), error={result['error_type']}")

                if pattern is not None:
                    desc = pattern.description()
                    learned_rules.append(desc)
                    before = len(queue)
                    queue, newly_suppressed = self._suppress_matching_configs(queue, pattern)
                    autophagy_suppressed_ids.extend(newly_suppressed)
                    print(f"       -> AUTOPHAGY: learned pattern [{desc}]")
                    print(f"       -> AUTOPHAGY: suppressed {before - len(queue)} future configs")

            all_results.append(result)

        total_time = time.time() - start_time
        df = pd.DataFrame(all_results)
        df_ok = df[df["success"] == True].copy()

        stats = {
            "strategy": "PA-DSE",
            "benchmark": self.benchmark_name,
            "initial_configs": len(configs),
            "static_blocked": len(static_blocked),
            "static_suppressed": len(static_suppressed),
            "autophagy_suppressed": len(set(autophagy_suppressed_ids)),
            "actual_runs": runs,
            "successful": int(len(df_ok)),
            "failed": int(len(df) - len(df_ok)),
            "success_rate_pct": round((len(df_ok) / runs) * 100, 1) if runs else 0.0,
            "budget_saving_vs_full_pct": round(((len(configs) - runs) / len(configs)) * 100, 1) if configs else 0.0,
            "total_time_s": round(total_time, 1),
        }

        if not df_ok.empty and df_ok["total_area"].notna().any():
            stats["best_area"] = float(df_ok["total_area"].min())
            stats["mean_area"] = round(float(df_ok["total_area"].mean()), 1)
            stats["best_latency"] = float(df_ok["num_states"].min())
            stats["mean_latency"] = round(float(df_ok["num_states"].mean()), 1)
            pareto = find_pareto(df_ok, "total_area", "num_states")
            stats["pareto_points"] = int(len(pareto))
            stats["unique_qor_points"] = int(len(df_ok[["total_area", "num_states"]].drop_duplicates()))
        else:
            stats["best_area"] = None
            stats["mean_area"] = None
            stats["best_latency"] = None
            stats["mean_latency"] = None
            stats["pareto_points"] = 0
            stats["unique_qor_points"] = 0

        return PADSEResult(
            all_results=all_results,
            stats=stats,
            static_blocked=static_blocked,
            static_suppressed=static_suppressed,
            static_log=static_log,
            learned_rules=sorted(set(learned_rules)),
            autophagy_suppressed_ids=sorted(set(autophagy_suppressed_ids)),
        )
