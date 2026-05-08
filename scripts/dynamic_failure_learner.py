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

import math
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
                 min_support: int = 5, enable_pairwise: bool = False,
                 # ── Categorical-coverage extension (CC) ─────────────────
                 # beta_cov = 0 reproduces the original (vanilla) OFRS exactly.
                 # n_cov is the per-(dim,value) observation budget below which
                 # a coverage bonus is granted.
                 beta_cov: float = 0.0, n_cov: int = 2,
                 # ── QoR-saturation extension (QSE) ──────────────────────
                 # When a (dim, value) accumulates many successes but few
                 # unique (area, latency) outputs, it is QoR-saturated:
                 # further evaluations there cannot improve best_lat /
                 # best_area / UQoR. QSE adds a penalty to that (dim, value)
                 # in risk score, deferring further evaluation in favour of
                 # under-explored regions. gamma_qsat=0 reproduces vanilla.
                 gamma_qsat: float = 0.0, qsat_min_succ: int = 4,
                 # ── QoR-space Diversification (QSD) ─────────────────────
                 # Penalty for candidate configs whose *predicted* QoR is
                 # close to already-visited QoR points. delta_qsd=0 disables.
                 # Predicted QoR is the relevance-weighted mean per-dimension
                 # observed (area, latency) for c's parameter values.
                 delta_qsd: float = 0.0, qsd_min_succ: int = 3,
                 # ── QoR-Attractor extension (QAT) ───────────────────────
                 # Bonus (negative penalty) for candidate configs whose
                 # parameter values have historically produced LOW latency
                 # or area. This pulls the queue toward the lat-min/area-min
                 # cluster once it has been discovered, increasing the
                 # probability of finding the global lat-min within budget.
                 # alpha_attract = 0 disables. Activates only after at
                 # least qat_min_succ successes have been observed.
                 alpha_attract: float = 0.0, qat_min_succ: int = 4):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.min_support = min_support
        self.enable_pairwise = enable_pairwise
        self.beta_cov = beta_cov
        self.n_cov = n_cov
        self.gamma_qsat = gamma_qsat
        self.qsat_min_succ = qsat_min_succ
        self.delta_qsd = delta_qsd
        self.qsd_min_succ = qsd_min_succ
        self.alpha_attract = alpha_attract
        self.qat_min_succ = qat_min_succ

        self._counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: {"fail": 0, "success": 0})
        )
        self._pair_counts: Dict[Tuple[str,str], Dict[Tuple[str,str], Dict[str,int]]] = defaultdict(
            lambda: defaultdict(lambda: {"fail": 0, "success": 0})
        )
        # QoR points per (dim, value) — used to compute saturation
        # _qor_points[d][v] is a set of (area, latency) tuples observed
        self._qor_points: Dict[str, Dict[str, set]] = defaultdict(
            lambda: defaultdict(set)
        )
        # Per-(dim, value) running mean of (area, latency) for successes,
        # used to predict a candidate config's expected QoR.
        # _qor_sum[d][v] = (sum_area, sum_lat); _qor_n[d][v] = n_succ
        self._qor_sum: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(
            lambda: defaultdict(lambda: (0.0, 0.0))
        )
        self._qor_n: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        # Global cloud of all visited successful (area, latency) tuples
        # (with multiplicity = number of times this point was visited).
        self._qor_cloud: List[Tuple[float, float]] = []
        # QoR axis scales for normalization (computed lazily from the cloud)
        self._qor_scale_cache: Optional[Tuple[float, float]] = None
        # UQoR plateau tracking: history of |unique points| at each step
        # Used by the gate: if |unique| has not grown in last K steps,
        # QoR-aware extensions deactivate (the QoR space is exhausted).
        self._uqor_history: List[int] = []
        # Per-(dim, value) min observed latency and area (used by QAT
        # to attract candidates toward low-lat/low-area regions)
        self._qor_min_lat: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: float("inf"))
        )
        self._qor_min_area: Dict[str, Dict[str, float]] = defaultdict(
            lambda: defaultdict(lambda: float("inf"))
        )
        # Running global mins (across all visited successes)
        self._global_min_lat: float = float("inf")
        self._global_min_area: float = float("inf")
        self._total_fail = 0
        self._total_success = 0

    def update(self, config: Config, failed: bool,
               param_keys: List[str], pair_keys: List[Tuple[str, str]],
               qor: Optional[Tuple[float, float]] = None) -> None:
        """Update OFRS counts. qor=(area, latency) if known and success."""
        outcome = "fail" if failed else "success"
        if failed:
            self._total_fail += 1
        else:
            self._total_success += 1

        valid_qor = (not failed and qor is not None
                     and qor[0] is not None and qor[1] is not None)
        if valid_qor:
            qa, ql = float(qor[0]), float(qor[1])
            self._qor_cloud.append((qa, ql))
            self._qor_scale_cache = None  # invalidate
            if qa < self._global_min_area:
                self._global_min_area = qa
            if ql < self._global_min_lat:
                self._global_min_lat = ql

        for d in param_keys:
            v = str(config.get(d, ""))
            if v:
                self._counts[d][v][outcome] += 1
                if valid_qor:
                    self._qor_points[d][v].add((round(qa, 4), round(ql, 4)))
                    sa, sl = self._qor_sum[d][v]
                    self._qor_sum[d][v] = (sa + qa, sl + ql)
                    self._qor_n[d][v] += 1
                    if qa < self._qor_min_area[d][v]:
                        self._qor_min_area[d][v] = qa
                    if ql < self._qor_min_lat[d][v]:
                        self._qor_min_lat[d][v] = ql

        if self.enable_pairwise:
            for d1, d2 in pair_keys:
                v1 = str(config.get(d1, ""))
                v2 = str(config.get(d2, ""))
                if v1 and v2:
                    self._pair_counts[(d1, d2)][(v1, v2)][outcome] += 1

        # Update UQoR plateau history (count of distinct points in cloud)
        n_unique = len(set(self._qor_cloud)) if self._qor_cloud else 0
        self._uqor_history.append(n_unique)

    def _uqor_plateau(self, k: int = 5) -> bool:
        """Return True if |unique QoR points| has not grown in the last k
        observations. When True, QSE/QSD deactivate: the QoR space appears
        exhausted under the current exploration trajectory, and further
        diversification will only push toward fail-prone configs.
        """
        h = self._uqor_history
        if len(h) < k + 1:
            return False
        return h[-1] == h[-k - 1]

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

    def coverage_score(self, config: Config, param_keys: List[str]) -> float:
        """Categorical coverage bonus in [0, 1].

        For each parameter dimension d with value v in this config, compute
        under-exploration u(d,v) = max(0, 1 - obs(d,v)/n_cov). The dimension's
        contribution is u(d,v) weighted by relevance(d), focusing the bonus
        on dimensions OFRS has identified as discriminative. When no
        dimension has yet accumulated relevance (early in a run), fall back
        to an unweighted mean so CC can still operate during the first few
        evaluations — that is when categorical-coverage matters most,
        because OFRS has not yet had time to entrench preferences.

        A config whose every (dim, value) has been observed ≥ n_cov times
        scores 0; a config exercising a never-observed value scores close
        to 1.
        """
        if self.n_cov <= 0:
            return 0.0

        items = []
        for d in param_keys:
            v = str(config.get(d, ""))
            if not v:
                continue
            c = self._counts[d][v]
            obs = c["fail"] + c["success"]
            u = max(0.0, 1.0 - obs / self.n_cov)
            items.append((u, self.relevance(d)))

        if not items:
            return 0.0

        total_rel = sum(r for _, r in items)
        if total_rel > 1e-9:
            return sum(u * r for u, r in items) / total_rel
        return sum(u for u, _ in items) / len(items)

    def qor_saturation(self, config: Config, param_keys: List[str]) -> float:
        """QoR-saturation score in [0, 1] for this config.

        For each parameter dimension d with value v, the saturation s(d,v)
        measures how few unique QoR points have been observed relative to
        the number of successes. If (d=v) has many successes but only one
        unique (area, latency) point, that region is QoR-saturated and
        further evaluation cannot improve best_lat/best_area/UQoR.

        s(d, v) = 1 - (unique_qor / max(qsat_min_succ, n_success))
                 if n_success >= qsat_min_succ, else 0

        The config's saturation is the relevance-weighted average across
        its dimensions, mirroring how risk_score combines dimensions.

        gamma_qsat = 0 disables this entirely.
        """
        if self.gamma_qsat <= 0.0:
            return 0.0
        # UQoR plateau gate: if |unique QoR| has not grown in last 5 evals,
        # the QoR space appears exhausted; deactivate to prevent useless
        # diversification that pushes toward fail-prone regions.
        if self._uqor_plateau(k=5):
            return 0.0

        items = []
        for d in param_keys:
            v = str(config.get(d, ""))
            if not v:
                continue
            n_succ = self._counts[d][v]["success"]
            if n_succ < self.qsat_min_succ:
                continue
            n_unique = len(self._qor_points[d][v])
            if n_unique == 0:
                continue
            # Saturation: high if many successes but few unique points
            s = 1.0 - (n_unique / float(n_succ))
            s = max(0.0, min(1.0, s))
            items.append((s, self.relevance(d)))

        if not items:
            return 0.0
        total_rel = sum(r for _, r in items)
        if total_rel > 1e-9:
            return sum(s * r for s, r in items) / total_rel
        return sum(s for s, _ in items) / len(items)

    def _qor_scale(self) -> Tuple[float, float]:
        """Median absolute deviation of (area, latency) over the cloud,
        used to normalize distances. Computed lazily and cached."""
        if self._qor_scale_cache is not None:
            return self._qor_scale_cache
        if not self._qor_cloud:
            self._qor_scale_cache = (1.0, 1.0)
            return self._qor_scale_cache
        areas = [a for a, _ in self._qor_cloud]
        lats  = [l for _, l in self._qor_cloud]
        # Use range as scale; clamp to avoid div-by-zero
        sa = max(1e-6, max(areas) - min(areas))
        sl = max(1e-6, max(lats)  - min(lats))
        self._qor_scale_cache = (sa, sl)
        return self._qor_scale_cache

    def _predict_qor(self, config: Config, param_keys: List[str]
                     ) -> Optional[Tuple[float, float]]:
        """Predict (area, latency) of an unevaluated config c as the
        relevance-weighted average of per-(dim, value) running means.

        Returns None if too little data to predict. We require at least
        qsd_min_succ observations across the contributing dimensions.
        """
        num_a, num_l, denom = 0.0, 0.0, 0.0
        n_obs_total = 0
        for d in param_keys:
            v = str(config.get(d, ""))
            if not v:
                continue
            n = self._qor_n[d].get(v, 0)
            if n < 1:
                continue
            n_obs_total += n
            sa, sl = self._qor_sum[d][v]
            ma, ml = sa / n, sl / n
            w = self.relevance(d)
            if w <= 1e-9:
                continue
            num_a += w * ma
            num_l += w * ml
            denom += w
        if denom < 1e-9 or n_obs_total < self.qsd_min_succ:
            return None
        return (num_a / denom, num_l / denom)

    def qor_density_penalty(self, config: Config, param_keys: List[str]
                            ) -> float:
        """Parameter-space dispersion penalty in [0, 1].

        When the global QoR cloud is saturated (most successful evaluations
        produce few unique points relative to total successes), the algorithm
        is in a 'rut'. QSD adds a penalty proportional to how SIMILAR a
        candidate config's parameter values are to those of recently
        successful evaluations.

        Specifically:
        - Compute global saturation S = 1 - (|cloud|_unique / |cloud|)
        - For each dim d (with relevance weight w_d), compute the density
          of (d, c[d]) -- how many of the recent successes had this same
          (d, value).
        - Return S * (relevance-weighted mean of dim density).

        Effect: when not saturated globally, QSD is silent. Once the cloud
        gets stuck (many successes -> few unique QoR), QSD pushes the queue
        toward parameter values seen LESS often in recent successes,
        enabling exploration to break out of the rut.

        delta_qsd <= 0 disables this entirely.
        """
        if self.delta_qsd <= 0.0:
            return 0.0
        # UQoR plateau gate: same rationale as QSE. When |unique QoR| has
        # not grown in 5 evals, deactivate.
        if self._uqor_plateau(k=5):
            return 0.0
        if not self._qor_cloud or len(self._qor_cloud) < self.qsd_min_succ:
            return 0.0
        # Global saturation gate
        n_total = len(self._qor_cloud)
        n_unique = len(set(self._qor_cloud))
        global_sat = 1.0 - (n_unique / float(n_total))
        # Below 0.3 saturation = healthy diversity, no need to push
        if global_sat < 0.3:
            return 0.0

        # Per-dim density of c's values among recent successes
        items = []
        for d in param_keys:
            v = str(config.get(d, ""))
            if not v:
                continue
            n_succ_d_v = self._counts[d][v]["success"]
            n_succ_total = sum(self._counts[d][vv]["success"] for vv in self._counts[d])
            if n_succ_total < self.qsd_min_succ:
                continue
            density = n_succ_d_v / float(n_succ_total)
            items.append((density, self.relevance(d)))

        if not items:
            return 0.0
        total_rel = sum(r for _, r in items)
        if total_rel > 1e-9:
            avg_density = sum(d * r for d, r in items) / total_rel
        else:
            avg_density = sum(d for d, _ in items) / len(items)
        # Combine: scale density penalty by global saturation
        return global_sat * avg_density

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
                base = 0.5
            else:
                base = self._total_fail / total
        else:
            base = numerator / denominator

        # CC extension: subtract coverage bonus so under-explored configs
        # rank earlier. When beta_cov=0 this is a no-op (vanilla OFRS).
        if self.beta_cov > 0.0:
            base = base - self.beta_cov * self.coverage_score(config, param_keys)
        # QSE extension: ADD saturation penalty so QoR-saturated regions
        # get higher risk and are deferred. gamma_qsat=0 -> no-op.
        if self.gamma_qsat > 0.0:
            base = base + self.gamma_qsat * self.qor_saturation(config, param_keys)
        # QSD extension: ADD density penalty so configs predicted to land
        # in dense regions of QoR-space are deferred in favor of those
        # predicted to land in sparse / unexplored regions.
        # delta_qsd=0 -> no-op.
        if self.delta_qsd > 0.0:
            base = base + self.delta_qsd * self.qor_density_penalty(config, param_keys)
        # QAT extension: SUBTRACT attractor bonus so configs with novel
        # parameter values get advanced. ONLY applied when base risk is
        # already moderate-to-low (< 0.6) -- this prevents pulling
        # high-risk configs forward and protects SR.
        # alpha_attract=0 -> no-op.
        if self.alpha_attract > 0.0 and base < 0.6:
            base = base - self.alpha_attract * self.qor_attractor(config, param_keys)
        return base

    def qor_attractor(self, config: Config, param_keys: List[str]) -> float:
        """Dim-value-novelty bonus in [0, 1] for this config.

        For each dimension d with value v, novelty(d, v) measures how few
        successful evaluations this (d, v) pair has accumulated. Configs
        whose values are under-represented in the success log get a bonus.

        Concretely:
            novelty(d, v) = max(0, 1 - n_succ(d, v) / N_target)
        where N_target = qat_min_succ. So:
        - n_succ(d, v) = 0 → novelty = 1.0   (never seen successfully)
        - n_succ(d, v) = N_target → novelty = 0.0 (saturated)

        The config-level bonus is the relevance-weighted mean across
        dims. This pulls under-explored (d, v) values forward in the
        queue.

        IMPORTANT: this bonus is ONLY applied when the config's base risk
        is moderate-to-low (< 0.6). High-base-risk configs are not boosted
        regardless of novelty, since pulling them forward would drop SR.
        This keeps QAT inside the "low-risk + novel" regime, preserving
        SR while broadening QoR coverage.

        alpha_attract = 0 disables this entirely.
        """
        if self.alpha_attract <= 0.0:
            return 0.0
        if self._total_success < self.qat_min_succ:
            return 0.0
        # UQoR plateau gate: when |unique QoR| has not grown in last 5
        # evals, the QoR space appears exhausted; QAT deactivates so
        # the algorithm does not divert budget away from the established
        # best-lat / best-area cluster. Critical for benchmarks like
        # kernel_2mm with very small unique-QoR cardinality (~12 points).
        if self._uqor_plateau(k=5):
            return 0.0

        items = []
        for d in param_keys:
            v = str(config.get(d, ""))
            if not v:
                continue
            n_succ = self._counts[d][v]["success"]
            # Novelty: 1 if never seen successfully, decays linearly to 0
            # when n_succ reaches qat_min_succ.
            novelty = max(0.0, 1.0 - n_succ / float(self.qat_min_succ))
            items.append((novelty, self.relevance(d)))

        if not items:
            return 0.0
        total_rel = sum(r for _, r in items)
        if total_rel > 1e-9:
            return sum(s * r for s, r in items) / total_rel
        return sum(s for s, _ in items) / len(items)


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
        # CC extension passed through to OFRS:
        beta_cov: float = 0.0,
        n_cov: int = 2,
        # QSE extension passed through to OFRS:
        gamma_qsat: float = 0.0,
        qsat_min_succ: int = 4,
        # QSD extension passed through to OFRS:
        delta_qsd: float = 0.0,
        qsd_min_succ: int = 3,
        # QAT extension passed through to OFRS:
        alpha_attract: float = 0.0,
        qat_min_succ: int = 4,
    ) -> None:
        self.tau = tau or threshold
        self.rpe_min_confidence = rpe_min_confidence
        self.mode = mode

        self.scorer = OnlineFailureRiskScorer(
            min_support=min_support,
            enable_pairwise=enable_pairwise,
            beta_cov=beta_cov,
            n_cov=n_cov,
            gamma_qsat=gamma_qsat,
            qsat_min_succ=qsat_min_succ,
            delta_qsd=delta_qsd,
            qsd_min_succ=qsd_min_succ,
            alpha_attract=alpha_attract,
            qat_min_succ=qat_min_succ,
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
                    benchmark_name: Optional[str] = None,
                    qor: Optional[Tuple[float, float]] = None) -> None:
        """Record a success. qor=(area, latency) if known, used by QSE."""
        self._ensure_keys(config)
        self.success_log.append(config)
        self.scorer.update(config, failed=False,
                           param_keys=self._param_keys, pair_keys=self._pair_keys,
                           qor=qor)

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
