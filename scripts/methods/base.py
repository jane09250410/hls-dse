#!/usr/bin/env python3
"""
base.py — Abstract base class for all DSE methods.

Responsibilities are clearly separated:
  initialize()    — build the initial evaluation queue
  apply_skips()   — remove configs (RPE hard skip, OFRS skip in ablation 8)
  apply_reorder() — sort remaining queue (OFRS ranking, RPE-reorder in ablation 7)
  select_next()   — pop the first config (already filtered and sorted)
  update()        — learn from evaluation result
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

Config = Dict[str, Any]


@dataclass
class SkipRecord:
    """One skipped configuration, for skip_log."""
    config_id: int
    skip_reason: str           # rpe_signature | ofrs_skip | failure_memo
    signature_id: Optional[int] = None
    signature_type: Optional[str] = None
    signature_cond: Optional[str] = None
    signature_conf: Optional[float] = None
    signature_n_fail: Optional[int] = None
    signature_n_counter: Optional[int] = None
    is_subsumed_product: bool = False


@dataclass
class SignatureEvent:
    """One signature lifecycle event, for signature_log."""
    event: str                 # created | activated | updated | subsumed | deactivated
    signature_id: int
    error_type: str
    conditions: str            # JSON string
    conf: float
    n_fail: int
    n_counter: int
    subsumed_signature_id: Optional[int] = None


class DSEMethod(ABC):
    """Abstract base for all DSE exploration methods."""

    def __init__(self, configs: List[Config], benchmark_name: str,
                 tool: str, budget: int, seed: Optional[int] = None,
                 **kwargs):
        self.configs = configs
        self.benchmark_name = benchmark_name
        self.tool = tool
        self.budget = budget
        self.seed = seed

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Unique strategy string for logs (e.g. 'Random', 'PA-DSE_Full')."""
        ...

    @abstractmethod
    def initialize(self) -> List[Config]:
        """Return initial evaluation queue. Called once per run."""
        ...

    def apply_skips(self, queue: List[Config], eval_step: int
                    ) -> Tuple[List[Config], List[SkipRecord]]:
        """Remove configs that should be permanently skipped.
        Returns (remaining_queue, skip_records).
        Default: no skips."""
        return queue, []

    def apply_reorder(self, queue: List[Config]) -> List[Config]:
        """Reorder queue by priority (lower risk first).
        Default: identity (no reorder)."""
        return queue

    def select_next(self, queue: List[Config]) -> Tuple[Config, str]:
        """Pop first config from queue. Returns (config, action).
        action is 'evaluate' or 'probe'.
        Default: pop(0) with action='evaluate'."""
        return queue.pop(0), "evaluate"

    @abstractmethod
    def update(self, config: Config, success: bool,
               output: str, synthesis_time: float):
        """Update internal state after one evaluation."""
        ...

    def get_signature_events(self) -> List[SignatureEvent]:
        """Return new signature events since last call. Default: empty."""
        return []

    def get_risk_score(self, config: Config) -> Optional[float]:
        """Return OFRS risk score for config. Default: None."""
        return None

    def get_active_signature_count(self) -> int:
        """Number of active RPE signatures. Default: 0."""
        return 0

    def get_overhead_ms(self) -> Dict[str, float]:
        """Per-component overhead. Keys: phago, rpe, ofrs."""
        return {"phago": 0.0, "rpe": 0.0, "ofrs": 0.0}
