#!/usr/bin/env python3
"""
collect_bambu_gt.py
====================
Exhaustively evaluate all 420 Bambu configs × 8 benchmarks.
Output: results/bambu_ground_truth/run_summary.csv
Format: compatible with GroundTruthOracle (strategy="Grid_offline_ref")

~3360 synthesis calls × ~2.5s each ≈ 2.3 hours.
Supports resume: re-run safely after interruption.

Usage:
    cd ~/hls-dse
    nohup python3 collect_bambu_gt.py > bambu_gt.log 2>&1 &
    tail -f bambu_gt.log
"""

import csv, os, sys, time, subprocess
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "scripts"))

from config_generator import generate_bambu_configs, config_to_bambu_cmd

BENCHMARKS = ["matmul", "vadd", "fir", "histogram", "atax", "bicg", "gemm", "gesummv"]
GT_DIR = "results/bambu_ground_truth"
GT_CSV = os.path.join(GT_DIR, "run_summary.csv")

# Oracle-compatible columns
COLUMNS = ["strategy", "benchmark", "config_id", "success",
           "area", "latency", "error_type", "synthesis_time_s"]


def log(msg):
    print(f'[{datetime.now().strftime("%H:%M:%S")}] {msg}', flush=True)


def parse_bambu_output(output, returncode):
    """Extract success/area/latency/error_type from Bambu output.

    Bambu's report format has varied across versions: some lines use
    ': value', others 'value =' or '= value'. The regex accepts either.
    Latency is preferred from 'Average execution' or 'Total cycles'; if
    neither is present, falls back to 'Number of states'.
    """
    import re
    success = returncode == 0 and "Total estimated area" in output

    area, latency, error_type = None, None, ""

    if success:
        # Area
        m = re.search(r"Total\s+estimated\s+area\s*[=:]\s*([\d.]+)", output)
        if m:
            try:
                area = float(m.group(1))
            except ValueError:
                pass
        # Latency: prefer Average execution / Total cycles
        m = re.search(r"(?:Average\s+execution|Total\s+cycles)\s*[=:]\s*([\d.]+)", output)
        if m:
            try:
                latency = float(m.group(1))
            except ValueError:
                pass
        # Fallback: Number of states (matches simulator and run_single conventions)
        if latency is None:
            m = re.search(r"Number\s+of\s+states\s*[=:]\s*(\d+)", output)
            if m:
                try:
                    latency = float(m.group(1))
                except ValueError:
                    pass
    else:
        low = output.lower()
        if "timeout" in low:
            error_type = "timeout"
        elif "phi" in low and "pipeline" in low:
            error_type = "pipeline_phi_conflict"
        elif "incompatib" in low or "channels" in low:
            error_type = "param_incompatibility"
        else:
            error_type = "generic_error"

    return success, area, latency, error_type


def main():
    configs = generate_bambu_configs(enable_pipeline=True)
    total = len(configs) * len(BENCHMARKS)
    log(f"Bambu Ground Truth Collection")
    log(f"  Configs: {len(configs)}, Benchmarks: {len(BENCHMARKS)}, Total: {total}")

    os.makedirs(GT_DIR, exist_ok=True)

    # Resume: load already-done (benchmark, config_id) pairs
    existing = set()
    if os.path.exists(GT_CSV):
        import pandas as pd
        old = pd.read_csv(GT_CSV)
        for _, row in old.iterrows():
            existing.add((row['benchmark'], int(row['config_id'])))
        log(f"  Resuming: {len(existing)} done, {total - len(existing)} remaining")

    mode = "a" if existing else "w"
    with open(GT_CSV, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not existing:
            writer.writeheader()

        done = len(existing)
        t0 = time.time()

        for bi, bench in enumerate(BENCHMARKS):
            src = os.path.abspath(f"benchmarks/{bench}/{bench}.c")
            if not os.path.exists(src):
                log(f"  WARNING: {src} not found, skipping {bench}")
                continue

            bench_new = 0
            for ci, config in enumerate(configs):
                if (bench, ci) in existing:
                    continue

                cmd = config_to_bambu_cmd(config, src, bench)
                work_dir = os.path.join(GT_DIR, bench, f"cfg_{ci}")
                os.makedirs(work_dir, exist_ok=True)

                t1 = time.time()
                try:
                    result = subprocess.run(
                        cmd, shell=True, cwd=work_dir,
                        capture_output=True, text=True, timeout=300,
                    )
                    output = result.stdout + "\n" + result.stderr
                    elapsed = time.time() - t1
                    success, area, latency, error_type = parse_bambu_output(output, result.returncode)
                except subprocess.TimeoutExpired:
                    elapsed = time.time() - t1
                    success, area, latency, error_type = False, None, None, "timeout"
                except Exception as e:
                    elapsed = time.time() - t1
                    success, area, latency, error_type = False, None, None, "generic_error"

                writer.writerow({
                    "strategy": "Grid_offline_ref",
                    "benchmark": bench,
                    "config_id": ci,
                    "success": success,
                    "area": area if area is not None else "",
                    "latency": latency if latency is not None else "",
                    "error_type": error_type,
                    "synthesis_time_s": round(elapsed, 2),
                })
                f.flush()

                done += 1
                bench_new += 1

                if bench_new % 50 == 0:
                    elapsed_total = time.time() - t0
                    rate = max(done - len(existing), 1) / max(elapsed_total, 1)
                    remaining = (total - done) / max(rate, 0.001)
                    log(f"  [{done}/{total}] {bench} cfg {ci}/{len(configs)} "
                        f"{'OK' if success else 'FAIL'} {elapsed:.1f}s "
                        f"ETA {remaining/60:.0f}min")

            # Per-benchmark summary
            import pandas as pd
            tmp = pd.read_csv(GT_CSV)
            bsub = tmp[tmp['benchmark'] == bench]
            sr = bsub['success'].astype(str).str.lower().eq('true').mean() * 100
            log(f"  [{bi+1}/8] {bench}: {len(bsub)} configs, GT SR = {sr:.1f}%")

    elapsed_total = time.time() - t0
    log(f"\nGround truth complete in {elapsed_total/3600:.1f} hours")

    # Final summary
    import pandas as pd
    df = pd.read_csv(GT_CSV)
    log(f"Total rows: {len(df)}")
    for bench in BENCHMARKS:
        sub = df[df['benchmark'] == bench]
        sr = sub['success'].astype(str).str.lower().eq('true').mean() * 100
        log(f"  {bench}: n={len(sub)}, GT SR={sr:.1f}%")


if __name__ == "__main__":
    main()
