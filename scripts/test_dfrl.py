#!/usr/bin/env python3
"""
test_dfrl.py — Conservative DFRL comparison test.

Replays actual synthesis data through L1 and DFRL modes.
RPE controls skip. OFRS controls ranking only.

Metrics:
  - Patterns learned (count, coverage)
  - Wasted synthesis calls (failures before suppression takes effect)
  - False skip rate (patterns that would incorrectly skip feasible configs)
  - Risk gap (separation between fail and success risk scores)
  - OFRS ranking quality (do low-risk configs succeed more?)

Usage:
    cd hls-dse-main
    python3 scripts/test_dfrl.py
"""

import csv
import sys
import os
import time as _time

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pattern_learner import FailurePatternLearner as L1Learner, extract_error_type
from dynamic_failure_learner import DynamicFailureRiskLearner
from config_generator import generate_bambu_configs


def load_results(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def make_config(row):
    c = {}
    for k in ["id", "clock_period", "pipeline", "pipeline_ii",
               "memory_policy", "channels_type", "channels_number"]:
        v = row.get(k)
        if v is None or v == "":
            c[k] = None
        elif k == "id":
            c[k] = int(float(v))
        elif k == "pipeline":
            c[k] = v == "True"
        elif k in ("clock_period", "pipeline_ii", "channels_number"):
            try:
                c[k] = int(float(v))
            except (ValueError, TypeError):
                c[k] = None
        else:
            c[k] = v
    return c


def fake_output(error_type):
    mapping = {
        "pipeline_phi_conflict": "Error: phi operations conflict with pipeline",
        "generic_error": "Error: synthesis failed",
        "timeout": "Error: timeout",
    }
    return mapping.get(error_type or "", "Error: generic_error")


def run_l1(rows, all_configs, bench):
    """Simulate L1 baseline."""
    learner = L1Learner(threshold=2)
    wasted = 0
    for row in rows:
        config = make_config(row)
        if row.get("success") != "True":
            wasted += 1
            err = row.get("error_type", "") or "generic_error"
            learner.add_failure(config, fake_output(err),
                                float(row.get("runtime_s", 20)), bench)

    patterns = learner.get_patterns()
    pipe = [c for c in all_configs if c.get("pipeline")]
    covered = sum(1 for c in pipe
                  if any(p.matches(dict(**c, benchmark=bench), bench) for p in patterns))

    return {
        "method": "L1 (intersection)",
        "patterns": len(patterns),
        "wasted": wasted,
        "pat_coverage": f"{covered}/{len(pipe)}",
    }


def run_dfrl(rows, all_configs, bench, mode, label):
    """
    Simulate DFRL in conservative mode.
    RPE: hard skip only. OFRS: ranking only.
    """
    learner = DynamicFailureRiskLearner(tau=2, mode=mode)

    wasted = 0
    rpe_would_skip = 0
    risk_before_fail = []
    risk_before_success = []

    t0 = _time.perf_counter()

    for i, row in enumerate(rows):
        config = make_config(row)
        success = row.get("success") == "True"

        # What would DFRL decide BEFORE evaluating?
        if i > 0:
            skip = learner.should_skip(config, bench)
            rank = learner.rank_priority(config)
            score = learner.risk_score(config)

            if not success and skip:
                rpe_would_skip += 1

            if not success:
                risk_before_fail.append(score)
            else:
                risk_before_success.append(score)

        # Feed result
        if success:
            learner.add_success(config, bench)
        else:
            wasted += 1
            err = row.get("error_type", "") or "generic_error"
            learner.add_failure(config, fake_output(err),
                                float(row.get("runtime_s", 20)), bench)

    elapsed_ms = (_time.perf_counter() - t0) * 1000

    patterns = learner.get_patterns()
    pipe = [c for c in all_configs if c.get("pipeline")]
    pat_covered = sum(1 for c in pipe
                      if any(p.matches(dict(**c, benchmark=bench), bench) for p in patterns))

    # False skip rate: how many SUCCESS configs would RPE incorrectly skip?
    success_configs = [make_config(r) for r in rows if r.get("success") == "True"]
    false_skips = sum(1 for c in success_configs
                      if learner.should_skip(c, bench))

    # Risk gap
    avg_rf = sum(risk_before_fail) / len(risk_before_fail) if risk_before_fail else 0
    avg_rs = sum(risk_before_success) / len(risk_before_success) if risk_before_success else 0

    return {
        "method": label,
        "patterns": len(patterns),
        "wasted": wasted,
        "rpe_would_skip": rpe_would_skip,
        "pat_coverage": f"{pat_covered}/{len(pipe)}",
        "false_skips": false_skips,
        "false_skip_rate": f"{false_skips}/{len(success_configs)}" if success_configs else "0/0",
        "avg_risk_fail": avg_rf,
        "avg_risk_succ": avg_rs,
        "risk_gap": avg_rf - avg_rs,
        "overhead_ms": elapsed_ms,
        "learner": learner,
    }


def main():
    all_configs = generate_bambu_configs(enable_pipeline=True)

    benchmarks = []
    for bench in ["matmul", "fir", "histogram", "vadd"]:
        path = f"results/budget_sweep/b60/{bench}/with_pipeline/PA_DSE_results.csv"
        if os.path.exists(path):
            benchmarks.append((bench, path))

    if not benchmarks:
        print("No data found. Run from hls-dse-main directory.")
        sys.exit(1)

    print("=" * 76)
    print("  Conservative DFRL Test (RPE skip / OFRS rank)")
    print("=" * 76)

    for bench, path in benchmarks:
        rows = load_results(path)
        n_fail = len([r for r in rows if r.get("success") != "True"])

        print(f"\n{'─' * 76}")
        print(f"  {bench}: {len(rows)} evals, {n_fail} failures")
        print(f"{'─' * 76}")

        l1 = run_l1(rows, all_configs, bench)
        modes = [
            ("intersection", "DFRL-intersection"),
            ("rpe_only",     "DFRL-RPE"),
            ("ofrs_only",    "DFRL-OFRS (rank only)"),
            ("full",         "DFRL-Full (RPE+rank)"),
        ]
        results = [run_dfrl(rows, all_configs, bench, m, lbl) for m, lbl in modes]

        # Table 1: Pattern & skip comparison
        print(f"\n  {'Method':<28} {'Pat':>4} {'Wasted':>7} {'RPE skip':>9} {'Pat.Cov':>10} {'FalseSkip':>10}")
        print(f"  {'─' * 70}")
        print(f"  {'L1 (intersection)':<28} {l1['patterns']:>4} {l1['wasted']:>7} {'—':>9} {l1['pat_coverage']:>10} {'—':>10}")
        for r in results:
            print(f"  {r['method']:<28} {r['patterns']:>4} {r['wasted']:>7} "
                  f"{r['rpe_would_skip']:>9} {r['pat_coverage']:>10} {r['false_skip_rate']:>10}")

        # Table 2: Risk quality (OFRS modes only)
        ofrs_modes = [r for r in results if r["method"] in ("DFRL-OFRS (rank only)", "DFRL-Full (RPE+rank)")]
        if ofrs_modes:
            print(f"\n  OFRS risk quality:")
            print(f"  {'Method':<28} {'Avg risk(F)':>12} {'Avg risk(S)':>12} {'Gap':>8} {'Overhead':>10}")
            print(f"  {'─' * 70}")
            for r in ofrs_modes:
                print(f"  {r['method']:<28} {r['avg_risk_fail']:>12.3f} {r['avg_risk_succ']:>12.3f} "
                      f"{r['risk_gap']:>8.3f} {r['overhead_ms']:>8.1f}ms")

        # Detail: DFRL-Full
        full = [r for r in results if r["method"] == "DFRL-Full (RPE+rank)"]
        if full:
            learner = full[0]["learner"]
            print(f"\n  DFRL-Full details:")
            print(f"  {learner.summary()}")

            # Sample risk scores
            samples = [
                {"pipeline": True, "pipeline_ii": 1, "clock_period": 5,
                 "memory_policy": "ALL_BRAM", "channels_type": "MEM_ACC_N1", "channels_number": 1},
                {"pipeline": True, "pipeline_ii": 1, "clock_period": 20,
                 "memory_policy": "ALL_BRAM", "channels_type": "MEM_ACC_N1", "channels_number": 1},
                {"pipeline": False, "clock_period": 10,
                 "memory_policy": "ALL_BRAM", "channels_type": "MEM_ACC_N1", "channels_number": 1},
            ]
            print(f"\n  Sample risk scores:")
            for s in samples:
                pipe = f"II={s.get('pipeline_ii')}" if s.get("pipeline") else "off"
                skip = learner.should_skip(s, bench)
                rank = learner.rank_priority(s)
                print(f"    pipe={pipe}, clk={s.get('clock_period')}: "
                      f"skip={skip}, rank={rank:.3f}")

    print(f"\n{'=' * 76}")
    print("  Interpretation:")
    print("  - RPE skip: how many failures RPE would have prevented (before they happen)")
    print("  - Pat.Cov: pattern coverage on full 180 pipeline configs")
    print("  - FalseSkip: successful configs that RPE would incorrectly skip (must be 0)")
    print("  - Risk gap: OFRS ability to separate fail from success (higher=better)")
    print("  - Overhead: DFRL computation time (must be negligible vs synthesis)")
    print("=" * 76)


if __name__ == "__main__":
    main()
