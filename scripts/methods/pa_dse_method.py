#!/usr/bin/env python3
"""
pa_dse_method.py — Unified PA-DSE method for all 8 ablation configurations.

Config map:
  no-filter        scf=off  rpe=off      ofrs=off
  SCF-only         scf=on   rpe=off      ofrs=off
  SCF+RPE          scf=on   rpe=skip     ofrs=off
  SCF+OFRS         scf=on   rpe=off      ofrs=rank
  SCF+DFRL         scf=on   rpe=skip     ofrs=rank   [recommended]
  DFRL-only        scf=off  rpe=skip     ofrs=rank
  SCF+RPE-reorder  scf=on   rpe=reorder  ofrs=off    [hierarchy test]
  SCF+OFRS-skip    scf=on   rpe=off      ofrs=skip   [hierarchy test]

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
    "no-filter", "SCF-only", "SCF+RPE", "SCF+OFRS",
    "SCF+DFRL", "DFRL-only", "SCF+RPE-reorder", "SCF+OFRS-skip",
})

ABLATION_MAP = {
    # config_name:     (use_scf, rpe_mode,  ofrs_mode)
    "no-filter":       (False,   "off",     "off"),
    "SCF-only":        (True,    "off",     "off"),
    "SCF+RPE":         (True,    "skip",    "off"),
    "SCF+OFRS":        (True,    "off",     "rank"),
    "SCF+DFRL":        (True,    "skip",    "rank"),
    "DFRL-only":       (False,   "skip",    "rank"),
    "SCF+RPE-reorder": (True,    "reorder", "off"),
    "SCF+OFRS-skip":   (True,    "off",     "skip"),
}


class PADSEMethod(DSEMethod):

    def __init__(self, configs, benchmark_name, tool, budget, *,
                 ablation_config="SCF+DFRL",
                 tau=2, theta=0.8, n_min=5, p_probe=0.05,
                 seed=None, queue_permutation_id=None,
                 source_path=None,
                 dynamic_mode="full",   # "full" or "intersection" for L1
                 # ── Categorical-coverage extension (CC) ────────────────
                 # Forwarded to OFRS. beta_cov=0 reproduces vanilla PA-DSE.
                 # method_name appends "+CC" when active so logs distinguish
                 # the variant.
                 beta_cov=0.0, n_cov=2,
                 # ── Late-Stage QoR Diversification (LSQD) ─────────────
                 # When enabled (lsqd=True), in the late budget phase
                 # (frac >= lsqd_start_frac), every lsqd_period steps the
                 # algorithm picks the queue config maximally distant in
                 # parameter space from already-evaluated configs (Hamming
                 # distance), instead of popping the OFRS-lowest-risk head.
                 # This pushes coverage of OFRS-deferred regions and
                 # reduces best_lat / best_area / UQoR variance across runs.
                 # method_name appends "+LSQD" when enabled.
                 lsqd=False, lsqd_start_frac=0.6, lsqd_period=5,
                 # ── QoR-Saturation Extension (QSE) ────────────────────
                 # Penalty added to OFRS risk_score for (dim, value) regions
                 # already QoR-saturated: many successes but few unique
                 # (area, latency) outputs. Pushes the queue toward
                 # under-explored QoR regions, improving best_lat/area/UQoR
                 # without losing the SR/wasted advantage of vanilla OFRS.
                 # gamma_qsat=0 reproduces vanilla. method_name appends
                 # "+QSE" when active.
                 gamma_qsat=0.0, qsat_min_succ=4,
                 # ── QoR-Space Diversification (QSD) ───────────────────
                 # Penalty added to risk_score based on the predicted
                 # (area, latency) of a candidate's PROXIMITY to the
                 # already-visited cloud. Configs predicted to land in
                 # dense regions of QoR-space are deferred; configs
                 # predicted to land in sparse regions are advanced.
                 # Targets UQoR specifically. delta_qsd=0 disables.
                 # method_name appends "+QSD" when active.
                 delta_qsd=0.0, qsd_min_succ=3,
                 # ── QoR-Attractor extension (QAT) ───────────────────
                 # Bonus (negative penalty) for candidate configs whose
                 # parameter values have historically produced LOW lat or
                 # area. Pulls queue toward best-lat / best-area cluster
                 # once it has been discovered. alpha_attract=0 disables.
                 # method_name appends "+QAT" when active.
                 alpha_attract=0.0, qat_min_succ=4,
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
        self.beta_cov = beta_cov
        self.n_cov = n_cov

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
            beta_cov=beta_cov, n_cov=n_cov,
            gamma_qsat=gamma_qsat, qsat_min_succ=qsat_min_succ,
            delta_qsd=delta_qsd, qsd_min_succ=qsd_min_succ,
            alpha_attract=alpha_attract, qat_min_succ=qat_min_succ,
        ) if need_dfrl else None

        # Save extension flags for method_name suffix
        self._qse_active = (gamma_qsat > 0.0)
        self._qsd_active = (delta_qsd > 0.0)
        self._qat_active = (alpha_attract > 0.0)
        self._tool = tool

        self._sig_buffer: List[SignatureEvent] = []
        self._overhead = {"phago": 0.0, "rpe": 0.0, "ofrs": 0.0}
        self._rng = random.Random(seed if seed is not None else 42)
        self._probe_flags = set()  # config ids marked as probe

        # ── LSQD state ──
        self._lsqd = bool(lsqd)
        self._lsqd_start_frac = float(lsqd_start_frac)
        self._lsqd_period = int(lsqd_period)
        # Track parameter dicts of already-evaluated configs (success or not),
        # used as the reference set for LSQD novelty computation.
        self._evaluated_param_dicts: List[dict] = []
        # Step counter for LSQD periodicity.
        self._step_for_lsqd = 0

    @property
    def method_name(self) -> str:
        if self.dynamic_mode == "intersection":
            return "PA-DSE_L1"
        suffix = ""
        if self.beta_cov > 0.0:
            suffix += "+CC"
        if self._lsqd:
            suffix += "+LSQD"
        if self._qse_active:
            suffix += "+QSE"
        if self._qsd_active:
            suffix += "+QSD"
        if self._qat_active:
            suffix += "+QAT"
        return f"PA-DSE_{self.ablation_config}{suffix}"

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

    def commit_selection(self, queue, index, *, via_lsqd=False):
        """Unified dequeue primitive (HFDS integration refactor).

        Pops queue[index] and performs ALL selection bookkeeping exactly once:
        _step_for_lsqd increment, base_action derivation, probe-flag
        consumption, _evaluated_param_dicts tracking.

        base_action is computed HERE (evaluate | probe | lsqd) from internal
        state — callers must not override it, so a HYP-selected config that
        RPE marked as probe keeps its probe accounting. External selector
        labels (HYP/BASE/AUDIT) are the adapter's business and never enter
        PA-DSE statistics. Both vanilla select_next() and the HFDS HYP path
        go through here.
        """
        if not (0 <= index < len(queue)):
            raise IndexError(
                f"commit_selection: index {index} out of range (queue len {len(queue)})")
        self._step_for_lsqd += 1
        cfg = queue.pop(index)
        cid = int(cfg.get("id", -1))
        if via_lsqd:
            base_action = "lsqd"
        else:
            base_action = "probe" if cid in self._probe_flags else "evaluate"
        self._probe_flags.discard(cid)
        # Track for LSQD novelty (record all evaluated, success or not)
        self._evaluated_param_dicts.append(
            {k: v for k, v in cfg.items()
             if not k.startswith("_") and k != "id"}
        )
        return cfg, base_action

    def select_next(self, queue):
        """
        Select the next config to evaluate.

        Default: pop queue[0] (which is OFRS-lowest-risk after apply_reorder).

        LSQD: in the late budget phase (>= lsqd_start_frac), every
        lsqd_period steps and only when the queue still has alternatives,
        pick the config maximally distant in parameter space from the
        already-evaluated set. This forces coverage of OFRS-deferred
        regions, reducing best_lat / best_area / UQoR variance across runs.

        Refactored to route through commit_selection(); the LSQD decision
        uses next_step = _step_for_lsqd + 1, equivalent to the previous
        increment-then-test order (verified by no-op equivalence tests).
        """
        next_step = self._step_for_lsqd + 1

        # Decide whether this step is an LSQD diversification step
        do_lsqd = (
            self._lsqd
            and len(queue) >= 2
            and len(self._evaluated_param_dicts) >= self.n_min   # need a reference set
            and next_step / max(1, self.budget) >= self._lsqd_start_frac
            and (next_step % self._lsqd_period) == 0
        )

        if do_lsqd:
            # Pick queue index maximizing min-Hamming-distance to evaluated set
            ref = self._evaluated_param_dicts
            best_idx, best_d = 0, -1
            for i, c in enumerate(queue):
                d = self._min_hamming(c, ref)
                if d > best_d:
                    best_d, best_idx = d, i
            return self.commit_selection(queue, best_idx, via_lsqd=True)
        return self.commit_selection(queue, 0)

    @staticmethod
    def _min_hamming(c, ref_list):
        """Min Hamming distance between c and any d in ref_list, over
        the parameter dict (id and underscore-prefixed keys excluded)."""
        c_items = {k: v for k, v in c.items()
                   if not k.startswith("_") and k != "id"}
        if not ref_list:
            return len(c_items)
        best = None
        for d in ref_list:
            keys = set(c_items) | set(d)
            dist = sum(1 for k in keys if c_items.get(k) != d.get(k))
            if best is None or dist < best:
                best = dist
                if best == 0:
                    return 0
        return best if best is not None else len(c_items)

    def update(self, config, success, output, synthesis_time):
        if not self._dfrl_active or self.learner is None:
            return
        t0 = time.perf_counter()

        if success:
            qor = self._extract_qor(output)
            self.learner.add_success(config, self.benchmark_name, qor=qor)
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

    def _extract_qor(self, output):
        """Extract (area, latency) from synthesis output.
        Returns (area, latency) or None if not extractable.
        Mirrors the regexes in run_single.py.
        """
        if not output:
            return None
        import re
        if self._tool == "bambu":
            am = re.search(r"Total\s+estimated\s+area\s*[=:]\s*([\d.]+)", output, re.IGNORECASE)
            lm = re.search(r"Number\s+of\s+states\s*[=:]\s*(\d+)", output, re.IGNORECASE)
        else:  # dynamatic
            am = re.search(r"components\s*=\s*(\d+)", output, re.IGNORECASE)
            lm = re.search(r"handshake_ops\s*=\s*(\d+)", output, re.IGNORECASE)
        if am and lm:
            try:
                return (float(am.group(1)), float(lm.group(1)))
            except Exception:
                return None
        return None

    def _count_counterexamples(self, pattern):
        if not self.learner:
            return 0
        return sum(1 for c in self.learner.success_log
                   if all(str(c.get(k)) == str(v)
                          for k, v in pattern.conditions.items()))
