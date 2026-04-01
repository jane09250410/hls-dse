#!/usr/bin/env python3
"""
pattern_learner.py

Online failure-pattern learner ("Autophagy L1") for HLS DSE.
The learner extracts lightweight failure signatures and recurrent
parameter patterns from failed HLS runs.

This version is deliberately simple and explainable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


Config = Dict[str, Any]


def extract_error_type(output: str) -> str:
    text = (output or "").lower()

    if "phi" in text and "pipeline" in text:
        return "pipeline_phi_conflict"
    if "phi operations" in text:
        return "pipeline_phi_conflict"
    if "channel" in text and ("cannot" in text or "invalid" in text or "requires" in text):
        return "channel_incompatibility"
    if "timeout" in text:
        return "timeout"
    if "memory" in text and ("fail" in text or "cannot" in text):
        return "memory_failure"
    if "error" in text:
        return "generic_error"
    return "unknown_failure"


def config_signature_items(config: Config, benchmark_name: Optional[str]) -> Dict[str, Any]:
    return {
        "benchmark": benchmark_name,
        "clock_period": config.get("clock_period"),
        "pipeline": config.get("pipeline"),
        "pipeline_ii": config.get("pipeline_ii"),
        "memory_policy": config.get("memory_policy"),
        "channels_type": config.get("channels_type"),
        "channels_number": config.get("channels_number"),
    }


def common_pattern(configs: List[Config], benchmark_name: Optional[str]) -> Dict[str, Any]:
    """
    Find common parameter assignments across a set of failed configurations.
    """
    if not configs:
        return {}

    signatures = [config_signature_items(c, benchmark_name) for c in configs]
    keys = list(signatures[0].keys())
    common = {}

    for k in keys:
        vals = [sig.get(k) for sig in signatures]
        if all(v == vals[0] for v in vals):
            common[k] = vals[0]

    return common


@dataclass
class FailureRecord:
    config_id: int
    config: Config
    error_type: str
    runtime_s: float
    output_tail: str = ""


@dataclass
class LearnedPattern:
    error_type: str
    conditions: Dict[str, Any]
    count: int = 0
    support_config_ids: List[int] = field(default_factory=list)

    def matches(self, config: Config, benchmark_name: Optional[str]) -> bool:
        sig = config_signature_items(config, benchmark_name)
        for k, v in self.conditions.items():
            if sig.get(k) != v:
                return False
        return True

    def description(self) -> str:
        conds = ", ".join(f"{k}={v}" for k, v in self.conditions.items())
        return f"{self.error_type}: {conds}"


class FailurePatternLearner:
    """
    Simple recurrent-pattern learner:
    - groups failures by error type
    - computes common parameter assignments
    - triggers a learned pattern after threshold failures
    """

    def __init__(self, threshold: int = 2) -> None:
        self.threshold = threshold
        self.failure_log: List[FailureRecord] = []
        self.learned_patterns: List[LearnedPattern] = []

    def add_failure(
        self,
        config: Config,
        output: str,
        runtime_s: float,
        benchmark_name: Optional[str] = None
    ) -> Optional[LearnedPattern]:
        err = extract_error_type(output)
        rec = FailureRecord(
            config_id=int(config.get("id", -1)),
            config=config,
            error_type=err,
            runtime_s=float(runtime_s),
            output_tail="\n".join((output or "").splitlines()[-8:]),
        )
        self.failure_log.append(rec)

        same_type = [r for r in self.failure_log if r.error_type == err]
        if len(same_type) < self.threshold:
            return None

        recent = same_type[-self.threshold:]
        pattern_cfgs = [r.config for r in recent]
        conditions = common_pattern(pattern_cfgs, benchmark_name)

        if not conditions:
            return None

        for lp in self.learned_patterns:
            if lp.error_type == err and lp.conditions == conditions:
                lp.count += 1
                lp.support_config_ids.append(rec.config_id)
                return lp

        lp = LearnedPattern(
            error_type=err,
            conditions=conditions,
            count=self.threshold,
            support_config_ids=[r.config_id for r in recent],
        )
        self.learned_patterns.append(lp)
        return lp

    def get_patterns(self) -> List[LearnedPattern]:
        return self.learned_patterns
