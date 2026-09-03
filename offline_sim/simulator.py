#!/usr/bin/env python3
"""
offline_sim/simulator.py
========================
Offline simulator for Dynamatic DSE methods.

Uses Grid_offline_ref ground-truth data as a synthesis oracle: given a config_id,
returns the (success, area, latency, error_type) recorded for that config in
the full grid sweep. This lets us run any DSE algorithm hundreds of times
without invoking the real Dynamatic toolchain.

Mirrors run_single.py main loop exactly: apply_skips → apply_reorder →
select_next → "synthesize" (table lookup) → update.

Coverage: the 8 shared benchmarks (matmul, vadd, fir, histogram, atax,
bicg, gemm, gesummv); 192 configs each for Dynamatic, 420 for Bambu.
"""

import json
import sys
import time
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from dynamatic_config_generator import generate_dynamatic_configs


# ──────────────────────────────────────────────────────────────
#  Oracle: load ground-truth and serve config-keyed queries
# ──────────────────────────────────────────────────────────────

class GroundTruthOracle:
    """Lookup table for synthesis outcomes from Grid_offline_ref data."""

    def __init__(self, ground_truth_csv: str, tool: str = "dynamatic"):
        df = pd.read_csv(ground_truth_csv)
        df = df[df["strategy"] == "Grid_offline_ref"].copy()
        # Keep first occurrence per (benchmark, config_id) — Bambu GT has duplicates
        df = df.drop_duplicates(subset=["benchmark", "config_id"], keep="first")
        self._tool = tool
        self._lookup: Dict[Tuple[str, int], Dict] = {}
        for _, row in df.iterrows():
            key = (row["benchmark"], int(row["config_id"]))
            self._lookup[key] = {
                "success": bool(row["success"]),
                "area": row["area"] if pd.notna(row["area"]) else None,
                "latency": row["latency"] if pd.notna(row["latency"]) else None,
                "error_type": row["error_type"] if pd.notna(row["error_type"]) else None,
                "synthesis_time_s": float(row["synthesis_time_s"]),
            }

    def benchmarks(self) -> List[str]:
        return sorted({b for b, _ in self._lookup.keys()})

    def synthesize(self, benchmark: str, config: Dict) -> Tuple[str, float, bool]:
        """Return (output, elapsed, success) — same interface as real synth_fn."""
        key = (benchmark, int(config["id"]))
        if key not in self._lookup:
            return ("CONFIG NOT IN GROUND TRUTH", 0.0, False)
        r = self._lookup[key]
        if r["success"]:
            if self._tool == "bambu":
                # Bambu QoR regex: "Total estimated area = N", "Number of states = N"
                out = (f"Total estimated area = {r['area']}\n"
                       f"Number of states = {int(r['latency'])}\n")
            else:  # dynamatic
                out = (f"components = {int(r['area'])}\n"
                       f"handshake_ops = {int(r['latency'])}\n")
            return (out, r["synthesis_time_s"], True)
        else:
            err = r["error_type"] or "unknown_error"
            if self._tool == "bambu":
                # Map common Bambu error types
                if "pipeline_phi" in (err or "").lower():
                    out = "ERROR: pipeline_phi conflict detected\n"
                elif "param_incompat" in (err or "").lower():
                    out = "ERROR: parameter incompatibility\n"
                else:
                    out = f"ERROR: {err}\n"
            else:
                if err == "buffer_placement_failed":
                    out = "buffer placement failed: MILP infeasible\n"
                elif err == "milp_infeasible":
                    out = "MILP infeasible: no valid buffer assignment\n"
                elif err == "timeout":
                    out = "synthesis timeout exceeded\n"
                else:
                    out = f"error: {err}\n"
            return (out, r["synthesis_time_s"], False)


# ──────────────────────────────────────────────────────────────
#  Simulator: replicate run_single.py loop with the oracle
# ──────────────────────────────────────────────────────────────

def _extract_int(pattern, text):
    m = re.search(pattern, text, re.IGNORECASE)
    return int(m.group(1)) if m else None

def _extract_float(pattern, text):
    m = re.search(pattern, text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def classify_dynamatic_error_local(output: str) -> str:
    if "buffer placement failed" in output.lower():
        return "buffer_placement_failed"
    if "milp infeasible" in output.lower():
        return "milp_infeasible"
    if "timeout" in output.lower():
        return "timeout"
    return "generic_error"


def simulate_run(method, oracle: GroundTruthOracle, benchmark: str) -> Dict:
    """
    Run one (method, benchmark) experiment offline.
    Returns a result dict with the same fields as run_summary.csv rows.
    """
    queue = method.initialize()

    n_evals = 0
    n_success = 0
    n_wasted = 0
    total_skipped = 0
    ttff_step = None
    best_area = None
    best_latency = None
    qor_set = set()
    probes_triggered = 0
    probes_succeeded = 0
    cumul_synth_time = 0.0

    eval_step = 0
    eval_log = []  # for detailed analysis

    while queue and n_evals < method.budget:
        eval_step += 1

        # Step 1: skip
        queue, skip_records = method.apply_skips(queue, eval_step)
        total_skipped += len(skip_records)

        if not queue:
            break

        # Step 2: reorder
        queue = method.apply_reorder(queue)

        # Step 3: select
        config, action = method.select_next(queue)
        n_evals += 1
        if action == "probe":
            probes_triggered += 1

        # Step 4: synthesize (via oracle)
        output, elapsed, success = oracle.synthesize(benchmark, config)
        cumul_synth_time += elapsed

        # Extract QoR (tool-specific, mirrors run_single.py)
        if oracle._tool == "bambu":
            area = _extract_float(r"Total\s+estimated\s+area\s*[=:]\s*([\d.]+)", output)
            latency = _extract_int(r"Number\s+of\s+states\s*[=:]\s*(\d+)", output)
        else:
            area = _extract_float(r"components\s*=\s*(\d+)", output)
            latency = _extract_int(r"handshake_ops\s*=\s*(\d+)", output)

        if success:
            n_success += 1
            if ttff_step is None:
                ttff_step = eval_step
            if area is not None and latency is not None:
                qor_set.add((area, latency))
            if best_area is None or (area is not None and area < best_area):
                best_area = area
            if best_latency is None or (latency is not None and latency < best_latency):
                best_latency = latency
            if action == "probe":
                probes_succeeded += 1
        else:
            n_wasted += 1

        # Step 5: update
        method.update(config, success, output, elapsed)

        eval_log.append({
            "step": eval_step,
            "config_id": int(config["id"]),
            "config": dict(config),
            "success": success,
            "area": area,
            "latency": latency,
            "action": action,
        })

    sr_pct = 100.0 * n_success / n_evals if n_evals else 0.0

    # Compute TTFF in synthesis-seconds: sum oracle synth time over steps 1..ttff_step
    ttff_synth_s = None
    if ttff_step is not None:
        ttff_synth_s = sum(
            oracle._lookup.get((benchmark, e["config_id"]), {}).get("synthesis_time_s", 0.0)
            for e in eval_log if e["step"] <= ttff_step
        )

    return {
        "strategy": method.method_name,
        "benchmark": benchmark,
        "budget": method.budget,
        "total_evals": n_evals,
        "successful_evals": n_success,
        "wasted_calls": n_wasted,
        "sr_pct": sr_pct,
        "best_area": best_area,
        "best_latency": best_latency,
        "uqor": len(qor_set),
        "ttff_step": ttff_step,
        "ttff_synth_s": ttff_synth_s,
        "total_skipped": total_skipped,
        "signatures_learned": method.get_active_signature_count(),
        "probes_triggered": probes_triggered,
        "probes_succeeded": probes_succeeded,
        "qor_set": list(qor_set),
        "eval_log": eval_log,
    }


# ──────────────────────────────────────────────────────────────
#  Top-level convenience runner
# ──────────────────────────────────────────────────────────────

def run_padse_offline(
    benchmark: str,
    oracle: GroundTruthOracle,
    *,
    tool: str = "dynamatic",
    budget: int = 30,
    queue_permutation_id: int = 0,
    ablation_config: str = "SCF+DFRL",
    tau: int = 2,
    theta: float = 0.8,
    n_min: int = 5,
    p_probe: float = 0.05,
    # CC extension
    beta_cov: float = 0.0,
    n_cov: int = 2,
    use_cc: bool = False,
    # LSQD extension
    lsqd: bool = False,
    lsqd_start_frac: float = 0.6,
    lsqd_period: int = 5,
    # QSE extension
    gamma_qsat: float = 0.0,
    qsat_min_succ: int = 4,
    # QSD extension
    delta_qsd: float = 0.0,
    qsd_min_succ: int = 3,
    # QAT extension
    alpha_attract: float = 0.0,
    qat_min_succ: int = 4,
):
    """Run one FA-DSE offline simulation (Bambu or Dynamatic)."""
    from methods.pa_dse_method import PADSEMethod

    if tool == "bambu":
        from config_generator import generate_bambu_configs
        configs = generate_bambu_configs(enable_pipeline=True)
        src = str(REPO_ROOT / "benchmarks" / benchmark / f"{benchmark}.c")
    else:
        configs = generate_dynamatic_configs()
        src = None

    kwargs = dict(
        ablation_config=ablation_config,
        tau=tau, theta=theta, n_min=n_min, p_probe=p_probe,
        seed=queue_permutation_id,
        queue_permutation_id=queue_permutation_id,
        source_path=src,
    )
    if use_cc:
        kwargs["beta_cov"] = beta_cov
        kwargs["n_cov"] = n_cov
    if lsqd:
        kwargs["lsqd"] = True
        kwargs["lsqd_start_frac"] = lsqd_start_frac
        kwargs["lsqd_period"] = lsqd_period
    if gamma_qsat > 0.0:
        kwargs["gamma_qsat"] = gamma_qsat
        kwargs["qsat_min_succ"] = qsat_min_succ
    if delta_qsd > 0.0:
        kwargs["delta_qsd"] = delta_qsd
        kwargs["qsd_min_succ"] = qsd_min_succ
    if alpha_attract > 0.0:
        kwargs["alpha_attract"] = alpha_attract
        kwargs["qat_min_succ"] = qat_min_succ

    method = PADSEMethod(configs, benchmark, tool, budget, **kwargs)
    return simulate_run(method, oracle, benchmark)


def run_random_offline(
    benchmark: str, oracle: GroundTruthOracle,
    *, tool: str = "dynamatic", budget: int = 30, queue_permutation_id: int = 0,
):
    from methods.baseline_methods import RandomMethod
    if tool == "bambu":
        from config_generator import generate_bambu_configs
        configs = generate_bambu_configs(enable_pipeline=True)
    else:
        configs = generate_dynamatic_configs()
    method = RandomMethod(configs, benchmark, tool, budget,
                          seed=queue_permutation_id)
    return simulate_run(method, oracle, benchmark)
