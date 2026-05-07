#!/usr/bin/env python3
"""
run_dynamatic_single.py
=======================
Run a single Dynamatic synthesis and extract metrics.

Metrics extracted:
  - success: whether compile + write-hdl succeeded
  - num_components: number of VHDL component files (area proxy)
  - num_buffers: number of buffer components (buffer overhead)
  - num_handshake_ops: count of handshake operations in MLIR
  - compile_time: wall-clock time for compilation
  - error_type: classification of failure if any

Usage:
    from run_dynamatic_single import run_dynamatic_single, extract_dynamatic_metrics
"""

import os
import re
import time
import shutil
import tempfile
import subprocess
from typing import Any, Dict, Optional, Tuple

from dynamatic_config_generator import Config, config_to_dynamatic_script, config_to_label


# ============================================================
# Dynamatic path configuration
# ============================================================

DYNAMATIC_PATH = os.path.expanduser("~/dynamatic")
DYNAMATIC_BIN = os.path.join(DYNAMATIC_PATH, "build", "bin", "dynamatic")

# Fallback: check if dynamatic is in PATH
if not os.path.isfile(DYNAMATIC_BIN):
    result = shutil.which("dynamatic")
    if result:
        DYNAMATIC_BIN = result


def run_dynamatic_single(
    config: Config,
    src_file: str,
    top_func: str,
    work_dir: str = None,
    timeout: int = 60,
) -> Tuple[bool, Dict[str, Any], str]:
    """
    Run Dynamatic for a single configuration.

    Args:
        config: parameter configuration dict
        src_file: absolute path to source .c file
        top_func: top function name (used for output dir naming)
        work_dir: working directory for output (temporary if None)
        timeout: max seconds for synthesis

    Returns:
        (success, metrics_dict, raw_output)
    """
    # Determine output directory
    # Dynamatic outputs to <src_dir>/out/ by default
    src_dir = os.path.dirname(os.path.abspath(src_file))
    out_dir = os.path.join(src_dir, "out")

    # Clean previous output
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)

    # Generate Dynamatic script
    script_content = config_to_dynamatic_script(
        config=config,
        src_file=os.path.abspath(src_file),
        dynamatic_path=DYNAMATIC_PATH,
    )

    # Write script to temp file
    script_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".sh", delete=False, prefix="dyn_"
    )
    script_file.write(script_content)
    script_file.close()

    # Run Dynamatic with process group kill on timeout
    start_time = time.time()
    try:
        import signal
        proc = subprocess.Popen(
            [DYNAMATIC_BIN, f"--run={script_file.name}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=src_dir,
            preexec_fn=os.setsid,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            elapsed = time.time() - start_time
            output = stdout + "\n" + stderr
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            # Kill entire process group (dynamatic + compile.sh + dynamatic-opt + cbc)
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            elapsed = time.time() - start_time
            output = "TIMEOUT"
            returncode = -1

    except Exception as e:
        elapsed = time.time() - start_time
        output = str(e)
        returncode = -2

    finally:
        os.unlink(script_file.name)

    # Check success
    success = (
        returncode == 0
        and "Compilation succeeded" in output
        and "HDL generation succeeded" in output
    )

    # Handle compile-only success (write-hdl failed)
    compile_ok = "Compilation succeeded" in output
    hdl_ok = "HDL generation succeeded" in output

    # Extract metrics
    metrics = extract_dynamatic_metrics(
        config=config,
        src_dir=src_dir,
        output=output,
        elapsed=elapsed,
        success=success,
        compile_ok=compile_ok,
        hdl_ok=hdl_ok,
    )

    return success, metrics, output


def extract_dynamatic_metrics(
    config: Config,
    src_dir: str,
    output: str,
    elapsed: float,
    success: bool,
    compile_ok: bool = False,
    hdl_ok: bool = False,
) -> Dict[str, Any]:
    """
    Extract metrics from Dynamatic output.

    Area proxy: count of VHDL component files
    Buffer count: number of buffer components
    Handshake ops: count from MLIR
    """
    out_dir = os.path.join(src_dir, "out")
    hdl_dir = os.path.join(out_dir, "hdl")
    comp_dir = os.path.join(out_dir, "comp")

    metrics = {
        "config_id": config["id"],
        "clock_period": config["clock_period"],
        "buffer_algorithm": config["buffer_algorithm"],
        "sharing": config.get("sharing", False),
        "disable_lsq": config.get("disable_lsq", False),
        "fast_token_delivery": config.get("fast_token_delivery", False),
        "success": success,
        "compile_ok": compile_ok,
        "hdl_ok": hdl_ok,
        "compile_time_s": round(elapsed, 2),
        "num_components": 0,
        "num_buffers": 0,
        "num_handshake_ops": 0,
        "error_type": "none",
    }

    if not success:
        metrics["error_type"] = classify_dynamatic_error(output)
        return metrics

    # Count VHDL components (area proxy)
    if os.path.isdir(hdl_dir):
        vhdl_files = [f for f in os.listdir(hdl_dir) if f.endswith(".vhd")]
        metrics["num_components"] = len(vhdl_files)
        metrics["num_buffers"] = sum(1 for f in vhdl_files if "buffer" in f.lower())

    # Count handshake operations from MLIR
    handshake_mlir = os.path.join(comp_dir, "handshake_buffered.mlir")
    if os.path.isfile(handshake_mlir):
        try:
            with open(handshake_mlir, "r") as f:
                content = f.read()
            metrics["num_handshake_ops"] = len(re.findall(r"handshake\.\w+", content))
        except Exception:
            pass

    return metrics


def classify_dynamatic_error(output: str) -> str:
    """
    Classify the type of Dynamatic failure.
    """
    output_lower = output.lower()

    if "timeout" in output_lower:
        return "timeout"
    if "cbc not installed" in output_lower:
        return "cbc_missing"
    if "milp" in output_lower and ("infeasible" in output_lower or "timeout" in output_lower):
        return "milp_infeasible"
    if "failed to place smart buffers" in output_lower:
        return "buffer_placement_failed"
    if "failed to compile" in output_lower:
        return "compile_failed"
    if "failed to export rtl" in output_lower:
        return "hdl_export_failed"
    if "segmentation fault" in output_lower or "core dumped" in output_lower:
        return "crash"
    if "error" in output_lower:
        return "generic_error"
    return "unknown"


if __name__ == "__main__":
    # Quick test
    from dynamatic_config_generator import generate_dynamatic_configs

    configs = generate_dynamatic_configs(
        clock_periods=[4],
        buffer_algorithms=["on-merges"],
        include_sharing=False,
        include_lsq=False,
        include_ftd=False,
    )

    test_src = os.path.join(DYNAMATIC_PATH, "integration-test", "fir", "fir.c")
    if os.path.isfile(test_src):
        print(f"Testing with: {test_src}")
        print(f"Config: {config_to_label(configs[0])}")
        success, metrics, output = run_dynamatic_single(configs[0], test_src, "fir")
        print(f"Success: {success}")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        print(f"Test source not found: {test_src}")
        print("Run this script from a machine with Dynamatic installed.")
