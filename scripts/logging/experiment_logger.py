#!/usr/bin/env python3
"""
experiment_logger.py — Unified logger for PA-DSE experiments.

Four log types (append-only CSV):
  1. eval_log      — one row per synthesis evaluation
  2. skip_log      — one row per skipped configuration
  3. signature_log — one row per signature lifecycle event
  4. run_summary   — one row per completed run

All joined by run_id.  Aligned with frozen experiment checklist.
"""

import csv, json, os, time
from typing import Any, Dict, Optional

EVAL_COLUMNS = [
    "run_id","strategy","benchmark","tool","budget","seed",
    "queue_permutation_id","ablation_config",
    "tau","theta","n_min","p_probe",
    "eval_step","config_id","config_params","action",
    "wall_clock_start","wall_clock_end","synthesis_time_s","cumulative_time_s",
    "success","error_type",
    "best_qor_name","best_qor_value","area","latency",
    "risk_score","rpe_active_signatures","rpe_skipped_this_step","is_probe",
]
SKIP_COLUMNS = [
    "run_id","config_id","skip_step","skip_reason",
    "matching_signature_id","matching_signature_type","matching_signature_cond",
    "signature_conf_at_skip","signature_n_fail","signature_n_counter",
    "is_subsumed_product","ground_truth_feasible",
]
SIG_COLUMNS = [
    "run_id","event","eval_step","signature_id",
    "error_type","conditions","conf","n_fail","n_counter",
    "subsumed_signature_id",
]
SUMMARY_COLUMNS = [
    "run_id","strategy","benchmark","tool","budget","seed",
    "queue_permutation_id","ablation_config",
    "tau","theta","n_min","p_probe",
    "total_evals","successful_evals","wasted_calls","sr_pct",
    "best_qor_name","best_qor_value","best_area","best_latency",
    "uqor","ttff_s","total_wall_clock_s",
    "total_skipped","false_skips_pending","false_skips_verified",
    "signatures_learned","probes_triggered","probes_succeeded",
    "overhead_phago_ms","overhead_rpe_ms","overhead_ofrs_ms",
]

class _CSVWriter:
    def __init__(self, path, columns):
        self.path, self.columns = path, columns
        self._ready = os.path.exists(path)
    def write_row(self, row):
        if not self._ready:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path,"w",newline="") as f:
                csv.DictWriter(f, self.columns).writeheader()
            self._ready = True
        with open(self.path,"a",newline="") as f:
            csv.DictWriter(f, self.columns, extrasaction="ignore").writerow(row)

class ExperimentLogger:
    def __init__(self, output_dir="results/experiments/logs"):
        self.output_dir = output_dir
        self._eval = _CSVWriter(f"{output_dir}/eval_log.csv", EVAL_COLUMNS)
        self._skip = _CSVWriter(f"{output_dir}/skip_log.csv", SKIP_COLUMNS)
        self._sig  = _CSVWriter(f"{output_dir}/signature_log.csv", SIG_COLUMNS)
        self._summ = _CSVWriter(f"{output_dir}/run_summary.csv", SUMMARY_COLUMNS)
        self.run_id = None
        self._run_start = 0.0
        self._meta = {}

    def start_run(self, *, strategy, benchmark, tool, budget,
                  seed=None, queue_permutation_id=None,
                  ablation_config="phago+Full",
                  tau=2, theta=0.8, n_min=5, p_probe=0.05):
        ts = int(time.time()*1000) % 10_000_000
        self.run_id = f"{strategy}_{benchmark}_B{budget}_s{seed}_{ts}"
        self._run_start = time.time()
        self._meta = dict(strategy=strategy, benchmark=benchmark, tool=tool,
                          budget=budget, seed=seed,
                          queue_permutation_id=queue_permutation_id,
                          ablation_config=ablation_config,
                          tau=tau, theta=theta, n_min=n_min, p_probe=p_probe)
        return self.run_id

    def log_eval(self, *, eval_step, config_id, config_params, action,
                 wall_clock_start, wall_clock_end, synthesis_time_s,
                 success, error_type, area, latency,
                 best_qor_name, best_qor_value,
                 risk_score, rpe_active_signatures,
                 rpe_skipped_this_step, is_probe):
        self._eval.write_row({
            "run_id": self.run_id, **self._meta,
            "eval_step": eval_step, "config_id": config_id,
            "config_params": json.dumps(config_params), "action": action,
            "wall_clock_start": f"{wall_clock_start:.4f}",
            "wall_clock_end": f"{wall_clock_end:.4f}",
            "synthesis_time_s": f"{synthesis_time_s:.3f}",
            "cumulative_time_s": f"{wall_clock_end - self._run_start:.3f}",
            "success": success, "error_type": error_type or "",
            "best_qor_name": best_qor_name,
            "best_qor_value": best_qor_value if best_qor_value is not None else "",
            "area": area if area is not None else "",
            "latency": latency if latency is not None else "",
            "risk_score": f"{risk_score:.4f}" if risk_score is not None else "",
            "rpe_active_signatures": rpe_active_signatures,
            "rpe_skipped_this_step": rpe_skipped_this_step,
            "is_probe": is_probe,
        })

    def log_skip(self, *, config_id, skip_step, skip_reason,
                 sig_id=None, sig_type=None, sig_cond=None,
                 sig_conf=None, sig_n_fail=None, sig_n_counter=None,
                 is_subsumed_product=False):
        self._skip.write_row({
            "run_id": self.run_id, "config_id": config_id,
            "skip_step": skip_step, "skip_reason": skip_reason,
            "matching_signature_id": sig_id or "",
            "matching_signature_type": sig_type or "",
            "matching_signature_cond": sig_cond or "",
            "signature_conf_at_skip": f"{sig_conf:.4f}" if sig_conf else "",
            "signature_n_fail": sig_n_fail if sig_n_fail is not None else "",
            "signature_n_counter": sig_n_counter if sig_n_counter is not None else "",
            "is_subsumed_product": is_subsumed_product,
            "ground_truth_feasible": "",
        })

    def log_signature_event(self, *, event, eval_step, sig_id,
                            error_type, conditions, conf,
                            n_fail, n_counter, subsumed_sig_id=None):
        self._sig.write_row({
            "run_id": self.run_id, "event": event, "eval_step": eval_step,
            "signature_id": sig_id, "error_type": error_type,
            "conditions": conditions, "conf": f"{conf:.4f}",
            "n_fail": n_fail, "n_counter": n_counter,
            "subsumed_signature_id": subsumed_sig_id or "",
        })

    def end_run(self, *, total_evals, successful, wasted,
                best_qor_name, best_qor_value,
                best_area, best_latency, uqor, ttff_s,
                total_skipped, false_skips_pending,
                false_skips_verified,
                signatures_learned, probes_triggered, probes_succeeded,
                overhead_phago_ms, overhead_rpe_ms, overhead_ofrs_ms):
        sr = round(successful/total_evals*100, 1) if total_evals else 0.0
        self._summ.write_row({
            "run_id": self.run_id, **self._meta,
            "total_evals": total_evals, "successful_evals": successful,
            "wasted_calls": wasted, "sr_pct": sr,
            "best_qor_name": best_qor_name,
            "best_qor_value": best_qor_value if best_qor_value is not None else "",
            "best_area": best_area if best_area is not None else "",
            "best_latency": best_latency if best_latency is not None else "",
            "uqor": uqor,
            "ttff_s": f"{ttff_s:.3f}" if ttff_s is not None else "",
            "total_wall_clock_s": f"{time.time()-self._run_start:.3f}",
            "total_skipped": total_skipped,
            "false_skips_pending": false_skips_pending,
            "false_skips_verified": false_skips_verified if false_skips_verified is not None else "",
            "signatures_learned": signatures_learned,
            "probes_triggered": probes_triggered,
            "probes_succeeded": probes_succeeded,
            "overhead_phago_ms": f"{overhead_phago_ms:.2f}",
            "overhead_rpe_ms": f"{overhead_rpe_ms:.2f}",
            "overhead_ofrs_ms": f"{overhead_ofrs_ms:.2f}",
        })
