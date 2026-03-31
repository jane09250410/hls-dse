import subprocess
import re
import csv
import os
import json
import time
import shutil
from config_generator import generate_bambu_configs, config_to_bambu_cmd


def run_bambu_single(cmd, work_dir, timeout=300):
    os.makedirs(work_dir, exist_ok=True)
    start_time = time.time()
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                cwd=work_dir, timeout=timeout)
        elapsed = time.time() - start_time
        output = result.stdout + "\n" + result.stderr
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        elapsed = timeout
        output = "TIMEOUT"
        success = False
    except Exception as e:
        elapsed = time.time() - start_time
        output = str(e)
        success = False
    return output, elapsed, success


def extract_bambu_metrics(output):
    metrics = {}
    patterns = {
        'num_control_steps': r'Number of control steps:\s+(\d+)',
        'num_states': r'Number of states:\s+(\d+)',
        'max_freq_mhz': r'Estimated max frequency \(MHz\):\s+([\d.]+)',
        'min_slack': r'Minimum slack:\s+([\d.]+)',
        'total_area': r'Total estimated area:\s+(\d+)',
        'area_no_mux': r'Estimated resources area \(no Muxes.*?\):\s+(\d+)',
        'mux_area': r'Estimated area of MUX21:\s+(\d+)',
        'num_dsps': r'Estimated number of DSPs:\s+(\d+)',
        'num_ffs': r'Total number of flip-flops.*?:\s+(\d+)',
        'num_modules': r'Number of modules instantiated:\s+(\d+)',
        'num_registers': r'Register allocation.*?(\d+) registers',
        'num_muxes': r'Number of allocated multiplexers.*?:\s+(\d+)',
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        metrics[key] = float(match.group(1)) if match else None
    return metrics


def run_all_bambu(configs, src_file, top_func, results_dir="results/bambu",
                  max_configs=None):
    all_results = []
    total = len(configs) if max_configs is None else min(max_configs, len(configs))
    abs_src = os.path.abspath(src_file)
    print(f"\n{'='*60}")
    print(f"  Running {total} Bambu configurations")
    print(f"{'='*60}\n")
    for i, config in enumerate(configs[:total]):
        config_id = config['id']
        work_dir = os.path.join(results_dir, f"config_{config_id}")
        cmd = config_to_bambu_cmd(config, abs_src, top_func)
        print(f"[{i+1}/{total}] Config {config_id}: clock={config['clock_period']}ns, "
              f"pipe={config['pipeline']}, mem={config['memory_policy']}, "
              f"ch={config['channels_type']}")
        output, elapsed, success = run_bambu_single(cmd, work_dir)
        metrics = extract_bambu_metrics(output)
        result = {**config, **metrics, 'runtime_s': round(elapsed, 2), 'success': success}
        all_results.append(result)
        with open(os.path.join(work_dir, "bambu_output.log"), 'w') as f:
            f.write(output)
        if success:
            print(f"       area={metrics.get('total_area', '?')}, "
                  f"states={metrics.get('num_states', '?')}, "
                  f"freq={metrics.get('max_freq_mhz', '?')}MHz, "
                  f"time={elapsed:.1f}s")
        else:
            print(f"       FAILED ({elapsed:.1f}s)")
        hls_dir = os.path.join(work_dir, "HLS_output")
        if os.path.exists(hls_dir):
            shutil.rmtree(hls_dir, ignore_errors=True)
    return all_results


def save_results_csv(results, filepath):
    if not results:
        print("No results to save.")
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    keys = results[0].keys()
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {filepath}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()
    configs, _ = generate_bambu_configs()
    results = run_all_bambu(configs, src_file="benchmarks/matmul/matmul.c",
                            top_func="matmul", max_configs=args.max)
    save_results_csv(results, "results/bambu_results.csv")
