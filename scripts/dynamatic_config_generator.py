#!/usr/bin/env python3
"""
dynamatic_config_generator.py
=============================
Generate design space configurations for Dynamatic HLS DSE.

Dynamatic parameters:
  - clock_period: target clock period in ns
  - buffer_algorithm: buffer placement strategy
  - sharing: credit-based resource sharing (on/off)
  - disable_lsq: force memory controllers instead of LSQs (on/off)
  - fast_token_delivery: fast token delivery strategy (on/off)

Usage:
    from dynamatic_config_generator import generate_dynamatic_configs
    configs = generate_dynamatic_configs()
"""

from itertools import product
from typing import Any, Dict, List

Config = Dict[str, Any]

# ============================================================
# Parameter Space Definition
# ============================================================

CLOCK_PERIODS = [2, 3, 4, 5, 6, 8, 10, 12]

BUFFER_ALGORITHMS = ["on-merges", "fpga20", "fpl22"]

BOOLEAN_PARAMS = {
    "sharing": [False, True],
    "disable_lsq": [False, True],
    "fast_token_delivery": [False, True],
}


def generate_dynamatic_configs(
    clock_periods: List[float] = None,
    buffer_algorithms: List[str] = None,
    include_sharing: bool = True,
    include_lsq: bool = True,
    include_ftd: bool = True,
) -> List[Config]:
    """
    Generate all Dynamatic configuration combinations.

    Returns list of dicts, each representing one design point.
    """
    if clock_periods is None:
        clock_periods = CLOCK_PERIODS
    if buffer_algorithms is None:
        buffer_algorithms = BUFFER_ALGORITHMS

    sharing_vals = [False, True] if include_sharing else [False]
    lsq_vals = [False, True] if include_lsq else [False]
    ftd_vals = [False, True] if include_ftd else [False]

    configs = []
    idx = 0
    for cp, buf, sh, lsq, ftd in product(
        clock_periods, buffer_algorithms, sharing_vals, lsq_vals, ftd_vals
    ):
        config = {
            "id": idx,
            "clock_period": cp,
            "buffer_algorithm": buf,
            "sharing": sh,
            "disable_lsq": lsq,
            "fast_token_delivery": ftd,
        }
        configs.append(config)
        idx += 1

    return configs


def config_to_dynamatic_script(
    config: Config,
    src_file: str,
    dynamatic_path: str,
    output_dir: str = None,
) -> str:
    """
    Convert a config dict to a Dynamatic frontend script (text).

    Returns the content of a .sh file that can be passed to:
        dynamatic --run=<script_file>
    """
    lines = []
    lines.append(f"set-dynamatic-path {dynamatic_path}")
    lines.append(f"set-src {src_file}")
    lines.append(f"set-clock-period {config['clock_period']}")

    if output_dir:
        lines.append(f"set-output-dir {output_dir}")

    # Build compile command with flags
    compile_parts = ["compile"]
    compile_parts.append(f"--buffer-algorithm {config['buffer_algorithm']}")

    if config.get("sharing", False):
        compile_parts.append("--sharing")
    if config.get("disable_lsq", False):
        compile_parts.append("--disable-lsq")
    if config.get("fast_token_delivery", False):
        compile_parts.append("--fast-token-delivery")

    lines.append(" ".join(compile_parts))
    lines.append("write-hdl")
    lines.append("exit")

    return "\n".join(lines)


def config_to_label(config: Config) -> str:
    """Human-readable label for a config."""
    parts = [
        f"cp={config['clock_period']}",
        f"buf={config['buffer_algorithm']}",
    ]
    if config.get("sharing"):
        parts.append("sharing")
    if config.get("disable_lsq"):
        parts.append("no-lsq")
    if config.get("fast_token_delivery"):
        parts.append("ftd")
    return "_".join(parts)


# ============================================================
# Static Rules for Dynamatic (Phagocytosis)
# ============================================================

def dynamatic_static_rules():
    """
    Return static feasibility rules for Dynamatic.
    
    These rules identify parameter combinations that are known
    to cause failures based on Dynamatic's tool-specific constraints.
    """
    rules = []

    # R1: disable_lsq + sharing may conflict
    # (sharing relies on LSQ for memory ordering)
    rules.append({
        "name": "R1_lsq_sharing_conflict",
        "severity": "suppress",
        "description": "disable_lsq with sharing may cause memory ordering issues",
        "condition": lambda c: c.get("disable_lsq") and c.get("sharing"),
    })

    # R2: Very tight clock with complex buffer algorithm may timeout
    rules.append({
        "name": "R2_tight_clock_complex_buffer",
        "severity": "suppress",
        "description": "Clock period <= 3ns with fpga20/fpl22 may cause MILP timeout",
        "condition": lambda c: c["clock_period"] <= 3 and c["buffer_algorithm"] in ("fpga20", "fpl22"),
    })

    return rules


def apply_static_rules(configs: List[Config], rules: List[dict], source_path: str = None):
    """
    Apply static rules to partition configs into active/blocked/suppressed.
    
    Returns: (active, blocked, suppressed, log)
    """
    active = []
    blocked = []
    suppressed = []
    log = []

    for cfg in configs:
        matched_rule = None
        for rule in rules:
            if rule["condition"](cfg):
                matched_rule = rule
                break

        if matched_rule is None:
            active.append(cfg)
        elif matched_rule["severity"] == "block":
            blocked.append(cfg)
            log.append({
                "config_id": cfg["id"],
                "rule": matched_rule["name"],
                "severity": "block",
                "description": matched_rule["description"],
            })
        else:  # suppress
            suppressed.append(cfg)
            log.append({
                "config_id": cfg["id"],
                "rule": matched_rule["name"],
                "severity": "suppress",
                "description": matched_rule["description"],
            })

    return active, blocked, suppressed, log


# ============================================================
# Code-Aware Analysis for Dynamatic
# ============================================================

def analyze_source_for_dynamatic(source_path: str) -> Dict[str, bool]:
    """
    Lightweight source code analysis for Dynamatic-specific patterns.
    
    Detects patterns that may interact with Dynamatic's dataflow scheduling:
    - while loops (dynamic iteration count)
    - nested loops (complex control flow)
    - data-dependent addressing (irregular memory access)
    - accumulation patterns (potential throughput bottleneck)
    """
    import re

    patterns = {
        "has_while_loop": False,
        "has_nested_loop": False,
        "has_data_dependent_addr": False,
        "has_accumulation": False,
        "has_conditional_branch": False,
        "loop_depth": 0,
    }

    try:
        with open(source_path, "r") as f:
            source = f.read()
    except FileNotFoundError:
        return patterns

    # Remove comments
    source_clean = re.sub(r"//.*", "", source)
    source_clean = re.sub(r"/\*.*?\*/", "", source_clean, flags=re.DOTALL)

    # While loops
    if re.search(r"\bwhile\s*\(", source_clean):
        patterns["has_while_loop"] = True

    # Count loop nesting
    for_count = len(re.findall(r"\bfor\s*\(", source_clean))
    while_count = len(re.findall(r"\bwhile\s*\(", source_clean))
    total_loops = for_count + while_count
    patterns["loop_depth"] = total_loops
    if total_loops >= 2:
        patterns["has_nested_loop"] = True

    # Accumulation: x += ..., x = x + ...
    if re.search(r"\w+\s*\+=", source_clean) or re.search(r"(\w+)\s*=\s*\1\s*[+\-*/]", source_clean):
        patterns["has_accumulation"] = True

    # Data-dependent addressing: arr[expr] where expr contains a variable
    if re.search(r"\w+\s*\[\s*\w+\s*[\[%*/+-]", source_clean):
        patterns["has_data_dependent_addr"] = True

    # Conditional branches inside loops
    if re.search(r"\bif\s*\(", source_clean):
        patterns["has_conditional_branch"] = True

    return patterns


if __name__ == "__main__":
    configs = generate_dynamatic_configs()
    print(f"Total Dynamatic configurations: {len(configs)}")
    print(f"\nParameter space:")
    print(f"  Clock periods: {CLOCK_PERIODS}")
    print(f"  Buffer algorithms: {BUFFER_ALGORITHMS}")
    print(f"  Boolean params: sharing, disable_lsq, fast_token_delivery")
    print(f"\nFirst 5 configs:")
    for c in configs[:5]:
        print(f"  {config_to_label(c)}")
    print(f"\nLast 5 configs:")
    for c in configs[-5:]:
        print(f"  {config_to_label(c)}")

    # Show static rule effects
    rules = dynamatic_static_rules()
    active, blocked, suppressed, log = apply_static_rules(configs, rules)
    print(f"\nStatic filtering:")
    print(f"  Active:     {len(active)}")
    print(f"  Blocked:    {len(blocked)}")
    print(f"  Suppressed: {len(suppressed)}")
