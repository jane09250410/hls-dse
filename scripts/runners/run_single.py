#!/usr/bin/env python3
"""
run_single.py — Atomic experiment runner.

Executes one (method, benchmark, budget, seed) combination with full logging.

Flow per iteration:
  1. method.apply_skips(queue)  → remove + log skips
  2. method.apply_reorder(queue)
  3. method.select_next(queue)  → config, action
  4. synthesize_fn(config)      → output, time, success
  5. method.update(...)
  6. log eval + signature events

synthesize_fn is injected by the caller (runner scripts).
"""

import re, time
from typing import Callable, Optional, Tuple

from methods.base import DSEMethod
from logging.experiment_logger import ExperimentLogger
from pattern_learner import extract_error_type


def run_single(
    method: DSEMethod,
    synthesize_fn: Callable,
    logger: ExperimentLogger,
    *,
    tool: str,
    ablation_config: str = "phago+Full",
    tau: int = 2,
    theta: float = 0.8,
    n_min: int = 5,
    p_probe: float = 0.05,
    queue_permutation_id: Optional[int] = None,
) -> str:
    """Run one complete experiment. Returns run_id."""

    qor_name = "area" if tool == "bambu" else "component_count"

    run_id = logger.start_run(
        strategy=method.method_name, benchmark=method.benchmark_name,
        tool=tool, budget=method.budget, seed=method.seed,
        queue_permutation_id=queue_permutation_id,
        ablation_config=ablation_config,
        tau=tau, theta=theta, n_min=n_min, p_probe=p_probe,
    )

    queue = method.initialize()

    # ── tracking ────────────────────────────────────────────────
    eval_step = 0
    n_evals = 0
    n_success = 0
    n_wasted = 0
    total_skipped = 0
    ttff = None
    best_area = None
    best_latency = None
    best_qor = None
    qor_set = set()
    probes_triggered = 0
    probes_succeeded = 0

    # ── main loop ───────────────────────────────────────────────
    while queue and n_evals < method.budget:
        eval_step += 1

        # Step 1: skip
        queue, skip_records = method.apply_skips(queue, eval_step)
        total_skipped += len(skip_records)
        for sr in skip_records:
            logger.log_skip(
                config_id=sr.config_id, skip_step=eval_step,
                skip_reason=sr.skip_reason,
                sig_id=sr.signature_id, sig_type=sr.signature_type,
                sig_cond=sr.signature_cond, sig_conf=sr.signature_conf,
                sig_n_fail=sr.signature_n_fail,
                sig_n_counter=sr.signature_n_counter,
                is_subsumed_product=sr.is_subsumed_product,
            )

        if not queue:
            break

        # Step 2: reorder
        queue = method.apply_reorder(queue)

        # Step 3: select
        config, action = method.select_next(queue)
        n_evals += 1
        if action == "probe":
            probes_triggered += 1

        # Step 4: synthesize
        risk = method.get_risk_score(config)
        active_sigs = method.get_active_signature_count()

        t_start = time.time()
        output, elapsed, success = synthesize_fn(config)
        t_end = time.time()

        # Extract QoR (tool-specific)
        if tool == "bambu":
            area = _extract_float(r"Total\s+estimated\s+area\s*[=:]\s*([\d.]+)", output)
            latency = _extract_int(r"Number\s+of\s+states\s*[=:]\s*(\d+)", output)
        else:  # dynamatic
            area = _extract_float(r"components\s*=\s*(\d+)", output)       # component count as area proxy
            latency = _extract_int(r"handshake_ops\s*=\s*(\d+)", output)   # handshake ops as latency proxy
        error_type = None
        qor_value = None

        if success:
            n_success += 1
            qor_value = area  # area for bambu, components for dynamatic

            if ttff is None:
                ttff = t_end - logger._run_start

            if area is not None and latency is not None:
                qor_set.add((area, latency))
            if best_area is None or (area is not None and area < best_area):
                best_area = area
            if best_latency is None or (latency is not None and latency < best_latency):
                best_latency = latency
            if best_qor is None or (qor_value is not None and qor_value < best_qor):
                best_qor = qor_value

            if action == "probe":
                probes_succeeded += 1
        else:
            n_wasted += 1
            if tool == "bambu":
                error_type = extract_error_type(output)
            else:
                from run_dynamatic_single import classify_dynamatic_error
                error_type = classify_dynamatic_error(output)

        # Step 5: update
        method.update(config, success, output, elapsed)

        # Step 6: log eval
        logger.log_eval(
            eval_step=eval_step,
            config_id=int(config.get("id", -1)),
            config_params={k: v for k, v in config.items()
                          if not k.startswith("_")},
            action=action,
            wall_clock_start=t_start, wall_clock_end=t_end,
            synthesis_time_s=elapsed,
            success=success, error_type=error_type,
            area=area, latency=latency,
            best_qor_name=qor_name, best_qor_value=qor_value,
            risk_score=risk,
            rpe_active_signatures=active_sigs,
            rpe_skipped_this_step=len(skip_records),
            is_probe=(action == "probe"),
        )

        # Step 7: log signature events
        for evt in method.get_signature_events():
            logger.log_signature_event(
                event=evt.event, eval_step=eval_step,
                sig_id=evt.signature_id, error_type=evt.error_type,
                conditions=evt.conditions, conf=evt.conf,
                n_fail=evt.n_fail, n_counter=evt.n_counter,
                subsumed_sig_id=evt.subsumed_signature_id,
            )

    # ── end run ─────────────────────────────────────────────────
    overhead = method.get_overhead_ms()
    logger.end_run(
        total_evals=n_evals, successful=n_success, wasted=n_wasted,
        best_qor_name=qor_name, best_qor_value=best_qor,
        best_area=best_area, best_latency=best_latency,
        uqor=len(qor_set), ttff_s=ttff,
        total_skipped=total_skipped,
        false_skips_pending=total_skipped,
        false_skips_verified=None,
        signatures_learned=method.get_active_signature_count(),
        probes_triggered=probes_triggered,
        probes_succeeded=probes_succeeded,
        overhead_phago_ms=overhead.get("phago", 0),
        overhead_rpe_ms=overhead.get("rpe", 0),
        overhead_ofrs_ms=overhead.get("ofrs", 0),
    )
    return run_id


def _extract_float(pattern, text):
    m = re.search(pattern, text, re.IGNORECASE)
    return float(m.group(1)) if m else None

def _extract_int(pattern, text):
    m = re.search(pattern, text, re.IGNORECASE)
    return int(m.group(1)) if m else None
