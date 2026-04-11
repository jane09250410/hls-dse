#!/usr/bin/env python3
"""
dynamic_failure_learner.py

Conservative Dynamic Failure Risk Learning (DFRL) for HLS DSE.

Two components:
  RPE (Recurrent Pattern Extractor) — high-confidence hard skip
  OFRS (Online Failure Risk Scorer) — queue priority ranking only

Design rules:
  - should_skip() is controlled ONLY by RPE
  - rank_priority() is controlled ONLY by OFRS
  - OFRS never does hard skip
  - Pairwise interactions off by default
  - Cold-start protection on relevance (min 5 observations)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from pattern_learner import extract_error_type

Config = Dict[str, Any]

BAMBU_PARAMS = [
    "clock_period", "pipeline", "pipeline_ii",
    "memory_policy", "channels_type", "channels_number",
]

DYNAMATIC_PARAMS = [
    "clock_period", "buffer_algorithm", "sharing",
    "disable_lsq", "fast_token_delivery",
]

BAMBU_PAIRS = [
    ("pipeline", "pipeline_ii"),
    ("pipeline", "clock_period"),
]

DYNAMATIC_PAIRS = [
    ("buffer_algorithm", "clock_period"),
    ("buffer_algorithm", "sharing"),
]


def _detect_params(config: Config) -> Tuple[List[str], List[Tuple[str, str]]]:
    if "buffer_algorithm" in config:
        return DYNAMATIC_PARAMS, DYNAMATIC_PAIRS
    return BAMBU_PARAMS, BAMBU_PAIRS


# ──────────────────────────────────────────────────────────────
#  Data classes
# ──────────────────────────────────────────────────────────────

@dataclass
class FailureRecord:
    config_id: int
    config: Config
    error_type: str
    runtime_s: float


@dataclass
class LearnedPattern:
    """Discrete failure pattern for hard skip."""
    error_type: str
    conditions: Dict[str, Any]
    count: int = 0
    support_config_ids: List[int] = field(default_factory=list)
    confidence: float = 1.0

    def matches(self, config: Config, benchmark_name: Optional[str] = None) -> bool:
        for k, v in self.conditions.items():
            if str(config.get(k)) != str(v):
                return False
        return True

    def description(self) -> str:
        conds = ", ".join(f"{k}={v}" for k, v in sorted(self.conditions.items()))
        return f"{self.error_type}: {{{conds}}} (conf={self.confidence:.2f}, n={self.count})"


# ──────────────────────────────────────────────────────────────
#  OFRS: Online Failure Risk Scorer (ranking only, never skip)
# ──────────────────────────────────────────────────────────────

class OnlineFailureRiskScorer:

    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0,
                 min_support: int = 5, enable_pairwise: bool = False):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.min_support = min_support
        self.enable_pairwise = enable_pairwise

        self._counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"fail": 0, "success": 0})
        )
        self._pair_counts: Dict[Tuple[str,str], Dict[Tuple[str,str], Dict[str,int]]] = defaultdict(
            lambda: defaultdict(lambda: {"fail": 0, "success": 0})
        )
        self._total_fail = 0
        self._total_success = 0

    def update(self, config: Config, failed: bool,
               param_keys: List[str], pair_keys: List[Tuple[str, str]]) -> None:
        outcome = "fail" if failed else "success"
        if failed:
            self._total_fail += 1
        else:
            self._total_success += 1

        for d in param_keys:
            v = str(config.get(d, ""))
            if v:
                self._counts[d][v][outcome] += 1

        if self.enable_pairwise:
            for d1, d2 in pair_keys:
                v1 = str(config.get(d1, ""))
                v2 = str(config.get(d2, ""))
                if v1 and v2:
                    self._pair_counts[(d1, d2)][(v1, v2)][outcome] += 1

    def phi(self, dim: str, val: str) -> float:
        c = self._counts[dim][val]
        nf = c["fail"]
        ns = c["success"]
        return (nf + self.prior_alpha) / (nf + ns + self.prior_alpha + self.prior_beta)

    def relevance(self, dim: str) -> float:
        vals = self._counts[dim]
        if not vals:
            return 0.0
        total = sum(c["fail"] + c["success"] for c in vals.values())
        if total < self.min_support:
            return 0.0
        phis = [self.phi(dim, v) for v in vals]
        return max(phis) - min(phis) if len(phis) > 1 else 0.0

    def risk_score(self, config: Config, param_keys: List[str],
                   pair_keys: List[Tuple[str, str]]) -> float:
        numerator = 0.0
        denominator = 0.0

        for d in param_keys:
            v = str(config.get(d, ""))
            if not v:
                continue
            w = self.relevance(d)
            if w > 1e-9:
                numerator += w * self.phi(d, v)
                denominator += w

        if denominator < 1e-9:
            total = self._total_fail + self._total_success
            if total == 0:
                return 0.5
            return self._total_fail / total

        return numerator / denominator

    def risk_decomposition(self, config: Config,
                           param_keys: List[str]) -> List[Tuple[str, str, float, float]]:
        result = []
        for d in param_keys:
            v = str(config.get(d, ""))
            if not v:
                continue
            result.append((d, v, self.relevance(d), self.phi(d, v)))
        result.sort(key=lambda t: t[2] * t[3], reverse=True)
        return result


# ──────────────────────────────────────────────────────────────
#  DFRL main class
# ──────────────────────────────────────────────────────────────

class DynamicFailureRiskLearner:
    """
    Modes:
      "intersection" — L1-compatible exact intersection
      "rpe_only"     — causal RPE, hard skip only
      "ofrs_only"    — risk ranking only, no skip
      "full"         — RPE hard skip + OFRS ranking
    """

    def __init__(
        self,
        tau: int = 2,
        rpe_min_confidence: float = 0.8,
        min_support: int = 5,
        enable_pairwise: bool = False,
        mode: str = "full",
        threshold: int = 2,
    ) -> None:
        self.tau = tau or threshold
        self.rpe_min_confidence = rpe_min_confidence
        self.mode = mode

        self.scorer = OnlineFailureRiskScorer(
            min_support=min_support,
            enable_pairwise=enable_pairwise,
        )
        self.failure_log: List[FailureRecord] = []
        self.success_log: List[Config] = []
        self.learned_patterns: List[LearnedPattern] = []

        self._param_keys: Optional[List[str]] = None
        self._pair_keys: Optional[List[Tuple[str, str]]] = None

    def _ensure_keys(self, config: Config) -> None:
        if self._param_keys is None:
            self._param_keys, self._pair_keys = _detect_params(config)

    def add_failure(self, config: Config, output: str, runtime_s: float,
                    benchmark_name: Optional[str] = None) -> Optional[LearnedPattern]:
        self._ensure_keys(config)
        err = extract_error_type(output)
        rec = FailureRecord(
            config_id=int(config.get("id", -1)),
            config=config, error_type=err,
            runtime_s=float(runtime_s),
        )
        self.failure_log.append(rec)
        self.scorer.update(config, failed=True,
                           param_keys=self._param_keys, pair_keys=self._pair_keys)

        if self.mode in ("full", "rpe_only", "intersection"):
            return self._try_extract_pattern(err, benchmark_name)
        return None

    def add_success(self, config: Config,
                    benchmark_name: Optional[str] = None) -> None:
        self._ensure_keys(config)
        self.success_log.append(config)
        self.scorer.update(config, failed=False,
                           param_keys=self._param_keys, pair_keys=self._pair_keys)

    def should_skip(self, config: Config,
                    benchmark_name: Optional[str] = None) -> bool:
        """Hard skip. Controlled ONLY by RPE. OFRS does not participate."""
        if self.mode in ("full", "rpe_only", "intersection"):
            for pat in self.learned_patterns:
                if pat.matches(config, benchmark_name):
                    return True
        return False

    def rank_priority(self, config: Config) -> float:
        """Queue priority. Controlled ONLY by OFRS. Higher = defer later."""
        if self.mode in ("full", "ofrs_only"):
            self._ensure_keys(config)
            return self.scorer.risk_score(
                config, self._param_keys, self._pair_keys)
        return 0.5

    def risk_score(self, config: Config) -> float:
        self._ensure_keys(config)
        return self.scorer.risk_score(
            config, self._param_keys, self._pair_keys)

    def get_patterns(self) -> List[LearnedPattern]:
        return self.learned_patterns

    def summary(self) -> str:
        lines = [f"DFRL ({self.mode}): {len(self.failure_log)}F/{len(self.success_log)}S, "
                 f"{len(self.learned_patterns)} patterns"]
        if self._param_keys:
            active = [(d, self.scorer.relevance(d)) for d in self._param_keys
                      if self.scorer.relevance(d) > 0]
            if active:
                lines.append("  Relevance: " + ", ".join(f"{d}={w:.2f}" for d, w in active))
        for pat in self.learned_patterns:
            lines.append(f"  {pat.description()}")
        return "\n".join(lines)

    # ── RPE internals ─────────────────────────────────────────

    def _try_extract_pattern(self, error_type, benchmark_name):
        same_type = [r for r in self.failure_log if r.error_type == error_type]
        if len(same_type) < self.tau:
            return None
        if self.mode == "intersection":
            return self._extract_intersection(same_type, error_type, benchmark_name)
        return self._extract_causal(same_type, error_type)

    def _extract_causal(self, same_type, error_type):
        fail_configs = [r.config for r in same_type]
        all_configs = fail_configs + self.success_log
        conditions = {}
        for d in self._param_keys:
            fail_vals = set(str(c.get(d)) for c in fail_configs if c.get(d) is not None)
            if len(fail_vals) != 1:
                continue
            all_vals = set(str(c.get(d)) for c in all_configs if c.get(d) is not None)
            if len(all_vals) > 1:
                conditions[d] = fail_configs[0].get(d)
        if not conditions:
            return None

        support = len(same_type)
        counter = sum(1 for c in self.success_log
                      if all(str(c.get(k)) == str(v) for k, v in conditions.items()))
        conf = support / (support + counter + 2)
        if conf < self.rpe_min_confidence:
            return None

        norm = {k: str(v) for k, v in conditions.items()}
        for pat in self.learned_patterns:
            ex = {k: str(v) for k, v in pat.conditions.items()}
            if pat.error_type == error_type and ex == norm:
                pat.count = support
                pat.confidence = conf
                return pat

        to_remove = []
        for i, pat in enumerate(self.learned_patterns):
            if pat.error_type != error_type:
                continue
            if set(norm.keys()) < set(pat.conditions.keys()):
                if all(str(pat.conditions.get(k)) == norm[k] for k in norm):
                    to_remove.append(i)
        for i in sorted(to_remove, reverse=True):
            self.learned_patterns.pop(i)

        new_pat = LearnedPattern(
            error_type=error_type, conditions=dict(conditions),
            count=support, confidence=conf,
            support_config_ids=[r.config_id for r in same_type],
        )
        self.learned_patterns.append(new_pat)
        return new_pat

    def _extract_intersection(self, same_type, error_type, benchmark_name):
        recent = same_type[-self.tau:]
        conditions = {}
        for d in self._param_keys:
            vals = [r.config.get(d) for r in recent]
            if all(v == vals[0] and v is not None for v in vals):
                conditions[d] = vals[0]
        if benchmark_name:
            conditions["benchmark"] = benchmark_name
        if not conditions:
            return None
        norm = {k: str(v) for k, v in conditions.items()}
        for pat in self.learned_patterns:
            ex = {k: str(v) for k, v in pat.conditions.items()}
            if pat.error_type == error_type and ex == norm:
                pat.count += 1
                return pat
        new_pat = LearnedPattern(
            error_type=error_type, conditions=conditions,
            count=self.tau,
            support_config_ids=[r.config_id for r in recent],
        )
        self.learned_patterns.append(new_pat)
        return new_pat


FailurePatternLearner = DynamicFailureRiskLearner
