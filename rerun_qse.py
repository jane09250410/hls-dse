#!/usr/bin/env python3
"""
rerun_qse.py — PA-DSE+QSE (QoR-Saturation Extension) experiments.

QSE adds a saturation penalty to OFRS risk_score: when a (dim, value) has
many successes but few unique (area, latency) outputs, the region is
QoR-saturated and further evaluations cannot improve best_lat / best_area /
UQoR. The penalty defers further evaluation in saturated regions, exposing
under-explored QoR regions.

Default hyperparameters (frozen, no per-benchmark tuning):
  gamma_qsat   = 0.30
  qsat_min_succ = 4

Output:
  results/qse/bambu_main/         — 8 Bambu × 10 perms × B=60
  results/qse/dynamatic_main/     — 4 Dynamatic × 10 perms × B=30
  results/qse/ablation/           — 8 abl × 12 bench × 3 perms

Resume capability: scans run_summary.csv at each sub-stage, skips completed.

Usage:
  cd ~/hls-dse
  nohup python3 rerun_qse.py > rerun_qse.log 2>&1 &
  tail -f rerun_qse.log

  # Selective:
  python3 rerun_qse.py --stage dynamatic_main
  python3 rerun_qse.py --stage bambu_main
  python3 rerun_qse.py --stage ablation
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

DYN_BENCH_PAPER = ['gcd', 'matching', 'binary_search', 'kernel_2mm']

DYN_PATH = os.path.expanduser('~/dynamatic/integration-test')

N_PERMS = 10
BAMBU_BUDGET = 60
DYN_BUDGET = 30

ABLATIONS = ['no-filter', 'SCF-only', 'SCF+RPE', 'SCF+OFRS',
             'SCF+DFRL', 'DFRL-only', 'SCF+RPE-reorder', 'SCF+OFRS-skip']

# ── QSE hyperparameters (frozen) ─────────────────────────
GAMMA_QSAT = 0.30
QSAT_MIN_SUCC = 4


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
# Stage A: Bambu PA-DSE+QSE (10 perms × 8 benches × B=60)
# ──────────────────────────────────────────────────────────

def stage_bambu_main():
    log('=' * 70)
    log(f'STAGE: Bambu PA-DSE+QSE (γ={GAMMA_QSAT}, min_succ={QSAT_MIN_SUCC})')
    log('=' * 70)

    configs_all = generate_bambu_configs(enable_pipeline=True)
    out_dir = 'results/qse/bambu_main'
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
            if is_done(done, 'PA-DSE_SCF+DFRL+QSE', b, BAMBU_BUDGET,
                       '', 'SCF+DFRL', str(pid)):
                continue
            log(f'{b} / PA-DSE+QSE / perm={pid}')
            try:
                m = PADSEMethod(configs_all, b, 'bambu', BAMBU_BUDGET,
                                ablation_config='SCF+DFRL',
                                source_path=src,
                                queue_permutation_id=pid,
                                gamma_qsat=GAMMA_QSAT,
                                qsat_min_succ=QSAT_MIN_SUCC)
                safe_run(m, synth, logger, 'bambu',
                         ablation_config='SCF+DFRL',
                         queue_permutation_id=pid)
            except Exception as e:
                log(f'  ERROR init: {e}')


# ──────────────────────────────────────────────────────────
# Stage B: Dynamatic PA-DSE+QSE (10 perms × 4 benches × B=30)
# ──────────────────────────────────────────────────────────

def stage_dynamatic_main():
    log('=' * 70)
    log(f'STAGE: Dynamatic PA-DSE+QSE (γ={GAMMA_QSAT}, min_succ={QSAT_MIN_SUCC})')
    log('=' * 70)

    configs_all = generate_dynamatic_configs()
    out_dir = 'results/qse/dynamatic_main'
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
            if is_done(done, 'PA-DSE_SCF+DFRL+QSE', b, DYN_BUDGET,
                       '', 'SCF+DFRL', str(pid)):
                continue
            log(f'{b} / PA-DSE+QSE / perm={pid}')
            try:
                m = PADSEMethod(configs_all, b, 'dynamatic', DYN_BUDGET,
                                ablation_config='SCF+DFRL',
                                source_path=src,
                                queue_permutation_id=pid,
                                gamma_qsat=GAMMA_QSAT,
                                qsat_min_succ=QSAT_MIN_SUCC)
                safe_run(m, synth, logger, 'dynamatic',
                         ablation_config='SCF+DFRL',
                         queue_permutation_id=pid)
            except Exception as e:
                log(f'  ERROR init: {e}')


# ──────────────────────────────────────────────────────────
# Stage C: Ablation (optional)
# ──────────────────────────────────────────────────────────

def stage_ablation():
    log('=' * 70)
    log(f'STAGE: Ablation × QSE (γ={GAMMA_QSAT})')
    log('=' * 70)

    out_dir = 'results/qse/ablation'
    os.makedirs(out_dir, exist_ok=True)
    logger = ExperimentLogger(out_dir)
    done = load_done(f'{out_dir}/run_summary.csv')

    configs_b = generate_bambu_configs(enable_pipeline=True)
    for b in BAMBU_BENCH:
        src = f'benchmarks/{b}/{b}.c'
        if not os.path.exists(src):
            continue
        synth = make_bambu_synth(src, b, f'{out_dir}/{b}')
        for abl in ABLATIONS:
            for pid in range(3):
                if is_done(done, f'PA-DSE_{abl}+QSE', b, BAMBU_BUDGET,
                           '', abl, str(pid)):
                    continue
                log(f'{b} / {abl}+QSE / perm={pid}')
                try:
                    m = PADSEMethod(configs_b, b, 'bambu', BAMBU_BUDGET,
                                    ablation_config=abl,
                                    source_path=src,
                                    queue_permutation_id=pid,
                                    gamma_qsat=GAMMA_QSAT,
                                    qsat_min_succ=QSAT_MIN_SUCC)
                    safe_run(m, synth, logger, 'bambu',
                             ablation_config=abl,
                             queue_permutation_id=pid)
                except Exception as e:
                    log(f'  ERROR: {e}')

    configs_d = generate_dynamatic_configs()
    for b in DYN_BENCH_PAPER:
        src = f'{DYN_PATH}/{b}/{b}.c'
        if not os.path.exists(src):
            continue
        synth = make_dynamatic_synth(src, b, f'{out_dir}/{b}')
        for abl in ABLATIONS:
            for pid in range(3):
                if is_done(done, f'PA-DSE_{abl}+QSE', b, DYN_BUDGET,
                           '', abl, str(pid)):
                    continue
                log(f'{b} / {abl}+QSE / perm={pid}')
                try:
                    m = PADSEMethod(configs_d, b, 'dynamatic', DYN_BUDGET,
                                    ablation_config=abl,
                                    source_path=src,
                                    queue_permutation_id=pid,
                                    gamma_qsat=GAMMA_QSAT,
                                    qsat_min_succ=QSAT_MIN_SUCC)
                    safe_run(m, synth, logger, 'dynamatic',
                             ablation_config=abl,
                             queue_permutation_id=pid)
                except Exception as e:
                    log(f'  ERROR: {e}')


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

    log(f'rerun_qse.py starting (γ={GAMMA_QSAT}, min_succ={QSAT_MIN_SUCC})')
    log(f'cwd={os.getcwd()}')

    # Run Dynamatic first (where QSE actually moves numbers); Bambu second
    # (no-op-ish but confirms no regression); ablation last.
    if args.stage == 'all':
        stage_dynamatic_main()
        stage_bambu_main()
        stage_ablation()
    else:
        STAGES[args.stage]()

    log('rerun_qse.py done')


if __name__ == '__main__':
    main()
