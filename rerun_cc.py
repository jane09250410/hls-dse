#!/usr/bin/env python3
"""
rerun_cc.py — Run PA-DSE-CC (Categorical Coverage extension) on top of
existing vanilla PA-DSE data.

Strategy: leave master/ data alone; produce only the +CC variant runs
needed to compare against the existing vanilla PA-DSE in the paper.

Output:
  results/cc/bambu_main/         — 8 Bambu benches × 10 perms × B=60, β=0.2
  results/cc/dynamatic_main/     — 4 Dynamatic benches × 10 perms × B=30, β=0.2
  results/cc/ablation/           — optional: 8 ablation configs × 4 benches × 3 perms × B=60

Resume capability: scans run_summary.csv at each sub-stage, skips completed cells.

Usage:
  cd ~/hls-dse
  nohup python3 rerun_cc.py > rerun_cc.log 2>&1 &
  tail -f rerun_cc.log

  # Selectively run only one stage:
  python3 rerun_cc.py --stage bambu_main
  python3 rerun_cc.py --stage dynamatic_main
  python3 rerun_cc.py --stage ablation

Hyperparameters (frozen, do not tune per-benchmark):
  beta_cov = 0.2
  n_cov    = 2
  Other PA-DSE params identical to vanilla (τ=2, θ=0.8, n_min=5, p_probe=0.05).
"""

import sys, os, csv, traceback, argparse
from datetime import datetime

sys.path.insert(0, 'scripts')

from config_generator import generate_bambu_configs
from dynamatic_config_generator import generate_dynamatic_configs
from feasibility_filter import phagocytosis, default_static_rules
from exp_logging.experiment_logger import ExperimentLogger
from runners.run_single import run_single
from runners.run_main_results import make_bambu_synth, make_dynamatic_synth
from methods.pa_dse_method import PADSEMethod

# ──────────────────────────────────────────────────────────
# Configuration — must match master_experiments.py
# ──────────────────────────────────────────────────────────

BAMBU_BENCH = ['matmul', 'vadd', 'fir', 'histogram',
               'bicg', 'atax', 'gemm', 'gesummv']

# 4 paper benchmarks (excluding fir/histogram which are SR=100% for all methods)
DYN_BENCH_PAPER = ['gcd', 'matching', 'binary_search', 'kernel_2mm']

DYN_PATH = os.path.expanduser('~/dynamatic/integration-test')

N_PERMS = 10
BAMBU_BUDGET = 60
DYN_BUDGET = 30

ABLATIONS = ['no-filter', 'SCF-only', 'SCF+RPE', 'SCF+OFRS',
             'SCF+DFRL', 'DFRL-only', 'SCF+RPE-reorder', 'SCF+OFRS-skip']

# ── CC hyperparameters (frozen) ───────────────────────────
BETA_COV = 0.2
N_COV    = 2


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


def load_done(csv_path):
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            done.add((r.get('strategy', ''), r.get('benchmark', ''),
                      r.get('budget', ''), r.get('seed', ''),
                      r.get('ablation_config', ''),
                      r.get('queue_permutation_id', '')))
    return done


def is_done(done, strategy, bench, budget, seed='', abl='N/A', perm=''):
    key = (strategy, bench, str(budget),
           str(seed) if seed != '' else '',
           abl, str(perm) if perm != '' else '')
    return key in done


def safe_run(method, synth, logger, tool, **kw):
    try:
        run_single(method, synth, logger, tool=tool, **kw)
        return True
    except Exception as e:
        log(f'  ERROR: {type(e).__name__}: {e}')
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────────────
# Stage A: Bambu PA-DSE+CC (10 perms × 8 benches × B=60)
# ──────────────────────────────────────────────────────────

def stage_bambu_main():
    log('=' * 70)
    log(f'STAGE: Bambu PA-DSE+CC (β={BETA_COV}, n_cov={N_COV})')
    log('=' * 70)

    configs_all = generate_bambu_configs(enable_pipeline=True)
    out_dir = 'results/cc/bambu_main'
    os.makedirs(out_dir, exist_ok=True)
    logger = ExperimentLogger(out_dir)
    done = load_done(f'{out_dir}/run_summary.csv')

    for b in BAMBU_BENCH:
        src = f'benchmarks/{b}/{b}.c'
        if not os.path.exists(src):
            log(f'SKIP {b}: missing .c file')
            continue
        synth = make_bambu_synth(src, b, f'{out_dir}/{b}')

        for pid in range(N_PERMS):
            if is_done(done, 'PA-DSE_SCF+DFRL+CC', b, BAMBU_BUDGET,
                       '', 'SCF+DFRL', str(pid)):
                continue
            log(f'{b} / PA-DSE+CC / perm={pid}')
            try:
                m = PADSEMethod(configs_all, b, 'bambu', BAMBU_BUDGET,
                                ablation_config='SCF+DFRL',
                                source_path=src,
                                queue_permutation_id=pid,
                                beta_cov=BETA_COV, n_cov=N_COV)
                safe_run(m, synth, logger, 'bambu',
                         ablation_config='SCF+DFRL',
                         queue_permutation_id=pid)
            except Exception as e:
                log(f'  ERROR init: {e}')


# ──────────────────────────────────────────────────────────
# Stage B: Dynamatic PA-DSE+CC (10 perms × 4 benches × B=30)
# ──────────────────────────────────────────────────────────

def stage_dynamatic_main():
    log('=' * 70)
    log(f'STAGE: Dynamatic PA-DSE+CC (β={BETA_COV}, n_cov={N_COV})')
    log('=' * 70)

    configs_all = generate_dynamatic_configs()
    out_dir = 'results/cc/dynamatic_main'
    os.makedirs(out_dir, exist_ok=True)
    logger = ExperimentLogger(out_dir)
    done = load_done(f'{out_dir}/run_summary.csv')

    for b in DYN_BENCH_PAPER:
        src = f'{DYN_PATH}/{b}/{b}.c'
        if not os.path.exists(src):
            log(f'SKIP {b}: missing .c file')
            continue
        synth = make_dynamatic_synth(src, b, f'{out_dir}/{b}')

        for pid in range(N_PERMS):
            if is_done(done, 'PA-DSE_SCF+DFRL+CC', b, DYN_BUDGET,
                       '', 'SCF+DFRL', str(pid)):
                continue
            log(f'{b} / PA-DSE+CC / perm={pid}')
            try:
                m = PADSEMethod(configs_all, b, 'dynamatic', DYN_BUDGET,
                                ablation_config='SCF+DFRL',
                                source_path=src,
                                queue_permutation_id=pid,
                                beta_cov=BETA_COV, n_cov=N_COV)
                safe_run(m, synth, logger, 'dynamatic',
                         ablation_config='SCF+DFRL',
                         queue_permutation_id=pid)
            except Exception as e:
                log(f'  ERROR init: {e}')


# ──────────────────────────────────────────────────────────
# Stage C: Ablation CC (optional, for completeness in Table IV)
# ──────────────────────────────────────────────────────────

def stage_ablation():
    log('=' * 70)
    log(f'STAGE: Ablation × CC (β={BETA_COV})')
    log('=' * 70)

    out_dir = 'results/cc/ablation'
    os.makedirs(out_dir, exist_ok=True)
    logger = ExperimentLogger(out_dir)
    done = load_done(f'{out_dir}/run_summary.csv')

    # Bambu ablation: full 8-bench × 8-config × 3-perm
    configs_b = generate_bambu_configs(enable_pipeline=True)
    for b in BAMBU_BENCH:
        src = f'benchmarks/{b}/{b}.c'
        if not os.path.exists(src):
            continue
        synth = make_bambu_synth(src, b, f'{out_dir}/{b}')
        for abl in ABLATIONS:
            for pid in range(3):
                if is_done(done, f'PA-DSE_{abl}+CC', b, BAMBU_BUDGET,
                           '', abl, str(pid)):
                    continue
                log(f'{b} / {abl}+CC / perm={pid}')
                try:
                    m = PADSEMethod(configs_b, b, 'bambu', BAMBU_BUDGET,
                                    ablation_config=abl,
                                    source_path=src,
                                    queue_permutation_id=pid,
                                    beta_cov=BETA_COV, n_cov=N_COV)
                    safe_run(m, synth, logger, 'bambu',
                             ablation_config=abl,
                             queue_permutation_id=pid)
                except Exception as e:
                    log(f'  ERROR: {e}')

    # Dynamatic ablation: 4 paper benches × 8 configs × 3 perm
    configs_d = generate_dynamatic_configs()
    for b in DYN_BENCH_PAPER:
        src = f'{DYN_PATH}/{b}/{b}.c'
        if not os.path.exists(src):
            continue
        synth = make_dynamatic_synth(src, b, f'{out_dir}/{b}')
        for abl in ABLATIONS:
            for pid in range(3):
                if is_done(done, f'PA-DSE_{abl}+CC', b, DYN_BUDGET,
                           '', abl, str(pid)):
                    continue
                log(f'{b} / {abl}+CC / perm={pid}')
                try:
                    m = PADSEMethod(configs_d, b, 'dynamatic', DYN_BUDGET,
                                    ablation_config=abl,
                                    source_path=src,
                                    queue_permutation_id=pid,
                                    beta_cov=BETA_COV, n_cov=N_COV)
                    safe_run(m, synth, logger, 'dynamatic',
                             ablation_config=abl,
                             queue_permutation_id=pid)
                except Exception as e:
                    log(f'  ERROR: {e}')


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

STAGES = {
    'bambu_main':     stage_bambu_main,
    'dynamatic_main': stage_dynamatic_main,
    'ablation':       stage_ablation,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=list(STAGES) + ['all'], default='all')
    args = p.parse_args()

    log(f'rerun_cc.py starting (β_cov={BETA_COV}, n_cov={N_COV})')
    log(f'cwd={os.getcwd()}')

    if args.stage == 'all':
        # Default order: main results first (most important), ablation last
        stage_bambu_main()
        stage_dynamatic_main()
        stage_ablation()
    else:
        STAGES[args.stage]()

    log('rerun_cc.py done')


if __name__ == '__main__':
    main()
