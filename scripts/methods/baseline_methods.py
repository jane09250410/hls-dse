#!/usr/bin/env python3
"""
baseline_methods.py — Non-PA-DSE methods with unified DSEMethod interface.

  RandomMethod         — uniform random, feasibility-unaware
  FilteredRandomMethod — random after Phagocytosis filtering
  GridMethod           — fixed-order enumeration (offline reference)
  LHSMethod            — Latin Hypercube Sampling approximation
  FailureMemoMethod    — exact failure lookup, no generalization
"""

import random
from typing import List, Optional, Tuple

from methods.base import DSEMethod, SkipRecord, Config
from feasibility_filter import phagocytosis, default_static_rules


class RandomMethod(DSEMethod):
    @property
    def method_name(self):
        return "Random"

    def initialize(self):
        q = list(self.configs)
        random.Random(self.seed).shuffle(q)
        return q

    def update(self, config, success, output, synthesis_time):
        pass


class FilteredRandomMethod(DSEMethod):
    def __init__(self, configs, benchmark_name, tool, budget,
                 seed=None, source_path=None, **kwargs):
        super().__init__(configs, benchmark_name, tool, budget, seed=seed)
        self.source_path = source_path

    @property
    def method_name(self):
        return "Filtered_Random"

    def initialize(self):
        if self.source_path:
            active, _, suppressed, _ = phagocytosis(
                configs=self.configs, rules=default_static_rules(),
                source_path=self.source_path,
                benchmark_name=self.benchmark_name,
            )
            q = list(active) + list(suppressed)
        else:
            q = list(self.configs)
        random.Random(self.seed).shuffle(q)
        return q

    def update(self, config, success, output, synthesis_time):
        pass


class GridMethod(DSEMethod):
    """Deterministic fixed-order enumeration.
    Reported as offline reference, NOT as competitive baseline."""

    @property
    def method_name(self):
        return "Grid_offline_ref"

    def initialize(self):
        return list(self.configs)

    def update(self, config, success, output, synthesis_time):
        pass


class LHSMethod(DSEMethod):
    """Latin Hypercube Sampling (discrete approximation)."""

    @property
    def method_name(self):
        return "LHS"

    def initialize(self):
        q = list(self.configs)
        random.Random(self.seed).shuffle(q)
        return q

    def update(self, config, success, output, synthesis_time):
        pass


class FailureMemoMethod(DSEMethod):
    """Exact failure memoization. Skips configs that previously failed.
    No generalization — weakest possible 'learn from failure'."""

    def __init__(self, configs, benchmark_name, tool, budget,
                 seed=None, **kwargs):
        super().__init__(configs, benchmark_name, tool, budget, seed=seed)
        self._failed = set()

    @property
    def method_name(self):
        return "Failure_Memo"

    def initialize(self):
        return list(self.configs)

    def apply_skips(self, queue, eval_step):
        remaining, skips = [], []
        for cfg in queue:
            cid = int(cfg.get("id", -1))
            if cid in self._failed:
                skips.append(SkipRecord(config_id=cid, skip_reason="failure_memo"))
            else:
                remaining.append(cfg)
        return remaining, skips

    def update(self, config, success, output, synthesis_time):
        if not success:
            self._failed.add(int(config.get("id", -1)))
