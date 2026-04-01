#!/usr/bin/env python3
"""
feasibility_filter.py

Static feasibility filtering ("Phagocytosis") for HLS DSE.
Designed to work with the existing HLS-DSE codebase.

Main features
-------------
1. Built-in static compatibility rules.
2. Optional lightweight source-code analysis rules.
3. Rule logging for reporting and debugging.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import re


Config = Dict[str, Any]


@dataclass
class Rule:
    name: str
    reason: str
    source: str = "static"
    severity: str = "block"   # "block" or "suppress"
    matcher: Callable[[Config, Optional[str], Optional[str]], bool] = lambda c, s, t: False

    def matches(
        self,
        config: Config,
        source_code: Optional[str] = None,
        benchmark_name: Optional[str] = None
    ) -> bool:
        try:
            return bool(self.matcher(config, source_code, benchmark_name))
        except Exception:
            return False


def read_text(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8", errors="ignore")


def has_accumulation_in_loop(source_code: Optional[str]) -> bool:
    """
    Lightweight heuristic: detect += or x = x + ... inside a for-loop body.
    This is intentionally simple and conservative.
    """
    if not source_code:
        return False

    loop_blocks = re.findall(r'for\s*\([^)]*\)\s*\{(.*?)\}', source_code, flags=re.S)
    for block in loop_blocks:
        if re.search(r'\+=', block):
            return True
        if re.search(r'([A-Za-z_]\w*)\s*=\s*\1\s*\+', block):
            return True
    return False


def has_if_inside_loop(source_code: Optional[str]) -> bool:
    if not source_code:
        return False
    loop_blocks = re.findall(r'for\s*\([^)]*\)\s*\{(.*?)\}', source_code, flags=re.S)
    for block in loop_blocks:
        if re.search(r'\bif\s*\(', block):
            return True
    return False


def default_static_rules() -> List[Rule]:
    rules = []

    rules.append(Rule(
        name="mem_acc_11_multichannel_invalid",
        reason="MEM_ACC_11 is incompatible with channels_number > 1.",
        source="static",
        severity="block",
        matcher=lambda c, s, b: (
            c.get("channels_type") == "MEM_ACC_11" and
            int(c.get("channels_number", 0)) > 1
        )
    ))

    rules.append(Rule(
        name="mem_acc_nn_requires_two_channels",
        reason="MEM_ACC_NN requires at least 2 channels.",
        source="static",
        severity="block",
        matcher=lambda c, s, b: (
            c.get("channels_type") == "MEM_ACC_NN" and
            int(c.get("channels_number", 0)) < 2
        )
    ))

    # Optional code-aware rule: accumulation pattern often causes pipeline failure
    rules.append(Rule(
        name="pipeline_with_accumulation_pattern",
        reason="Pipeline enabled on code with accumulation pattern inside loop; high risk of loop-carried dependence failure.",
        source="code_analysis",
        severity="suppress",
        matcher=lambda c, s, b: (
            bool(c.get("pipeline")) and has_accumulation_in_loop(s)
        )
    ))

    # Optional code-aware rule: branch inside loop may reduce pipeline feasibility
    rules.append(Rule(
        name="pipeline_with_branch_inside_loop",
        reason="Pipeline enabled on code with branch inside loop; high risk of control-flow related pipeline failure.",
        source="code_analysis",
        severity="suppress",
        matcher=lambda c, s, b: (
            bool(c.get("pipeline")) and has_if_inside_loop(s)
        )
    ))

    return rules


def phagocytosis(
    configs: List[Config],
    rules: Optional[List[Rule]] = None,
    source_path: Optional[str] = None,
    benchmark_name: Optional[str] = None
) -> Tuple[List[Config], List[Config], List[Config], List[Dict[str, Any]]]:
    """
    Apply static rules to split configurations into:
    - active_configs
    - blocked_configs
    - suppressed_configs
    - event_log
    """
    if rules is None:
        rules = default_static_rules()

    source_code = read_text(source_path)

    active = []
    blocked = []
    suppressed = []
    event_log = []

    for cfg in configs:
        status = "active"

        for rule in rules:
            if rule.matches(cfg, source_code=source_code, benchmark_name=benchmark_name):
                event_log.append({
                    "config_id": cfg.get("id"),
                    "rule": rule.name,
                    "reason": rule.reason,
                    "source": rule.source,
                    "severity": rule.severity,
                })
                if rule.severity == "block":
                    status = "blocked"
                    blocked.append(cfg)
                    break
                elif rule.severity == "suppress":
                    status = "suppress"

        if status == "active":
            active.append(cfg)
        elif status == "suppress":
            suppressed.append(cfg)

    return active, blocked, suppressed, event_log
