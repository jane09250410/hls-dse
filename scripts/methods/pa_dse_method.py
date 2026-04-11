#!/usr/bin/env python3
"""
pa_dse_method.py — Unified PA-DSE method for all 8 ablation configurations.

Config map:
  no-filter          phago=off  rpe=off      ofrs=off
  phago-only         phago=on   rpe=off      ofrs=off
  phago+RPE          phago=on   rpe=skip     ofrs=off
  phago+OFRS         phago=on   rpe=off      ofrs=rank
  phago+Full         phago=on   rpe=skip     ofrs=rank   [recommended]
  DFRL-only          phago=off  rpe=skip     ofrs=rank
  phago+RPE-reorder  phago=on   rpe=reorder  ofrs=off    [hierarchy test]
  phago+OFRS-skip    phago=on   rpe=off      ofrs=skip   [hierarchy test]

Calls DynamicFailureRiskLearner API exactly as defined in
dynamic_failure_learner.py (should_skip, rank_priority, risk_score,
add_failure, add_success, get_patterns).
"""

import json, random, time
from typing import List, Optional, Tuple

from methods.base import DSEMethod, SkipRecord, SignatureEvent, Config
from dynamic_failure_learner import DynamicFailureRiskLearner
from feasibility_filter import phagocytosis, default_static_rules

# ── Valid configs ───────────────────────────────────────────────

VALID_CONFIGS = frozenset({
    "no-filter", "phago-only", "phago+RPE", "phago+OFRS",
    "phago+Full", "DFRL-only", "phago+RPE-reorder", "phago+OFRS-skip",
})

ABLATION_MAP = {
    # config_name:       (use_phago, rpe_mode,  ofrs_mode)
    "no-filter":         (False,     "off",     "off"),
    "phago-only":        (True,      "off",     "off"),
    "phago+RPE":         (True,      "skip",    "off"),
    "phago+OFRS":        (True,      "off",     "rank"),
    "phago+Full":        (True,      "skip",    "rank"),
    "DFRL-only":         (False,     "skip",    "rank"),
    "phago+RPE-reorder": (True,      "reorder", "off"),
    "phago+OFRS-skip":   (True,      "off",     "skip"),
}


class PADSEMethod(DSEMethod):

    def __init__(self, configs, benchmark_name, tool, budget, *,
                 ablation_config="phago+Full",
                 tau=2, theta=0.8, n_min=5, p_probe=0.05,
                 seed=None, queue_permutation_id=None,
                 source_path=None,
                 dynamic_mode="full",   # "full" or "intersection" for L1
                 **kwargs):
        super().__init__(configs, benchmark_name, tool, budget, seed=seed)
        assert ablation_config in VALID_CONFIGS, \
            f"Invalid: {ablation_config}. Must be in {VALID_CONFIGS}"

        self.ablation_config = ablation_config
        self.tau = tau
        self.theta = theta
        self.n_min = n_min
        self.p_probe = p_probe
        self.queue_permutation_id = queue_permutation_id
        self.source_path = source_path
        self.dynamic_mode = dynamic_mode

        use_phago, rpe_mode, ofrs_mode = ABLATION_MAP[ablation_config]
        self._use_phago = use_phago
        self._rpe_mode = rpe_mode      # off | skip | reorder
        self._ofrs_mode = ofrs_mode    # off | rank | skip

        # Map ablation to DFRL mode string
        need_dfrl = (rpe_mode != "off" or ofrs_mode != "off")
        if dynamic_mode == "intersection":
            dfrl_mode = "intersection"
        elif rpe_mode != "off" and ofrs_mode != "off":
            dfrl_mode = "full"
        elif rpe_mode != "off":
            dfrl_mode = "rpe_only"
        elif ofrs_mode != "off":
            dfrl_mode = "ofrs_only"
        else:
            dfrl_mode = "full"  # unused

        self._dfrl_active = need_dfrl
        self.learner = DynamicFailureRiskLearner(
            tau=tau, rpe_min_confidence=theta,
            min_support=n_min, enable_pairwise=False,
            mode=dfrl_mode,
        ) if need_dfrl else None

        self._sig_buffer: List[SignatureEvent] = []
        self._overhead = {"phago": 0.0, "rpe": 0.0, "ofrs": 0.0}
        self._rng = random.Random(seed if seed is not None else 42)
        self._probe_flags = set()  # config ids marked as probe

    @property
    def method_name(self) -> str:
        if self.dynamic_mode == "intersection":
            return "PA-DSE_L1"
        return f"PA-DSE_{self.ablation_config}"

    def initialize(self) -> List[Config]:
        t0 = time.perf_counter()
        if self._use_phago and self.source_path:
            active, blocked, suppressed, _ = phagocytosis(
                configs=self.configs,
                rules=default_static_rules(),
                source_path=self.source_path,
                benchmark_name=self.benchmark_name,
            )
            queue = list(active) + list(suppressed)
        else:
            queue = list(self.configs)
        self._overhead["phago"] += (time.perf_counter() - t0) * 1000

        # Apply queue permutation if requested
        if self.queue_permutation_id is not None:
            rng = random.Random(self.queue_permutation_id)
            rng.shuffle(queue)
        return queue

    def apply_skips(self, queue, eval_step):
        if not self._dfrl_active or self.learner is None:
            return queue, []

        t0 = time.perf_counter()
        remaining, skip_records = [], []

        for cfg in queue:
            cid = int(cfg.get("id", -1))
            should_skip = False
            skip_reason = ""
            matched_pat = None

            # RPE hard skip (only in rpe_mode == "skip")
            if self._rpe_mode == "skip":
                if self.learner.should_skip(cfg, self.benchmark_name):
                    # Probe check
                    if self._rng.random() < self.p_probe:
                        self._probe_flags.add(cid)
                        remaining.append(cfg)
                        continue
                    should_skip = True
                    skip_reason = "rpe_signature"
                    matched_pat = self._find_matching_pattern(cfg)

            # OFRS skip (only in ablation config 8: ofrs_mode == "skip")
            if not should_skip and self._ofrs_mode == "skip":
                risk = self.learner.risk_score(cfg)
                if risk >= 0.85:
                    should_skip = True
                    skip_reason = "ofrs_skip"

            if should_skip:
                sr = SkipRecord(config_id=cid, skip_reason=skip_reason)
                if matched_pat:
                    sr.signature_id = id(matched_pat) % 100000
                    sr.signature_type = matched_pat.error_type
                    sr.signature_cond = json.dumps(
                        {k: str(v) for k, v in matched_pat.conditions.items()})
                    sr.signature_conf = matched_pat.confidence
                    sr.signature_n_fail = matched_pat.count
                    sr.signature_n_counter = self._count_counterexamples(matched_pat)
                skip_records.append(sr)
            else:
                remaining.append(cfg)

        self._overhead["rpe"] += (time.perf_counter() - t0) * 1000
        return remaining, skip_records

    def apply_reorder(self, queue):
        if not self._dfrl_active or self.learner is None:
            return queue
        t0 = time.perf_counter()

        if self._ofrs_mode in ("rank", "skip"):
            # OFRS global reorder
            queue.sort(key=lambda c: self.learner.rank_priority(c))
        elif self._rpe_mode == "reorder":
            # Ablation 7: RPE acts as ranker instead of skipper
            queue.sort(key=lambda c: self.learner.risk_score(c))

        self._overhead["ofrs"] += (time.perf_counter() - t0) * 1000
        return queue

    def select_next(self, queue):
        cfg = queue.pop(0)
        cid = int(cfg.get("id", -1))
        action = "probe" if cid in self._probe_flags else "evaluate"
        self._probe_flags.discard(cid)
        return cfg, action

    def update(self, config, success, output, synthesis_time):
        if not self._dfrl_active or self.learner is None:
            return
        t0 = time.perf_counter()

        if success:
            self.learner.add_success(config, self.benchmark_name)
        else:
            old_ids = {id(p) for p in self.learner.learned_patterns}
            pattern = self.learner.add_failure(
                config=config, output=output,
                runtime_s=synthesis_time,
                benchmark_name=self.benchmark_name,
            )
            if pattern is not None:
                sid = id(pattern) % 100000
                is_new = id(pattern) not in old_ids
                self._sig_buffer.append(SignatureEvent(
                    event="activated" if is_new and pattern.confidence >= self.theta else
                          "created" if is_new else "updated",
                    signature_id=sid,
                    error_type=pattern.error_type,
                    conditions=json.dumps(
                        {k: str(v) for k, v in pattern.conditions.items()}),
                    conf=pattern.confidence,
                    n_fail=pattern.count,
                    n_counter=self._count_counterexamples(pattern),
                ))

        self._overhead["rpe"] += (time.perf_counter() - t0) * 1000

    def get_signature_events(self):
        evts = list(self._sig_buffer)
        self._sig_buffer.clear()
        return evts

    def get_risk_score(self, config):
        if self.learner and self._ofrs_mode != "off":
            return self.learner.risk_score(config)
        return None

    def get_active_signature_count(self):
        return len(self.learner.learned_patterns) if self.learner else 0

    def get_overhead_ms(self):
        return dict(self._overhead)

    def _find_matching_pattern(self, config):
        if not self.learner:
            return None
        for pat in self.learner.learned_patterns:
            if pat.matches(config, self.benchmark_name):
                return pat
        return None

    def _count_counterexamples(self, pattern):
        if not self.learner:
            return 0
        return sum(1 for c in self.learner.success_log
                   if all(str(c.get(k)) == str(v)
                          for k, v in pattern.conditions.items()))
