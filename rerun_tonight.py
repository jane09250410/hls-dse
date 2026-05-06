#!/usr/bin/env python3
"""Rerun SA with restart + PA-DSE 10 perms on Bambu. Outputs to results/rerun/."""
import sys, os, csv, traceback
from datetime import datetime

sys.path.insert(0, 'scripts')

from config_generator import generate_bambu_configs
from feasibility_filter import phagocytosis, default_static_rules
from exp_logging.experiment_logger import ExperimentLogger
from runners.run_single import run_single
from runners.run_main_results import make_bambu_synth
from methods.pa_dse_method import PADSEMethod
from methods.advanced_baselines import SimulatedAnnealingMethod

BAMBU_BENCH = ['matmul', 'vadd', 'fir', 'histogram', 'bicg', 'atax', 'gemm', 'gesummv']
BUDGET = 60
N_SEEDS = 10
N_PERMS = 10

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)

def load_done(csv_path):
    done = set()
    if not os.path.exists(csv_path): return done
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            done.add((r.get('strategy',''), r.get('benchmark',''),
                      r.get('seed',''), r.get('queue_permutation_id','')))
    return done

def done_key(strategy, bench, seed='', perm=''):
    return (strategy, bench, str(seed) if seed != '' else '',
            str(perm) if perm != '' else '')

def safe_run(m, synth, logger, **kw):
    try:
        run_single(m, synth, logger, tool='bambu', **kw)
        return True
    except Exception as e:
        log(f'  ERROR: {type(e).__name__}: {e}')
        traceback.print_exc()
        return False

# ──────────────────────────────────────────────
# Part 1: Bambu SA with restart mechanism
# ──────────────────────────────────────────────
def part1_sa_restart():
    log('='*70)
    log('PART 1: Bambu SA with restart mechanism')
    log('='*70)
    configs_all = generate_bambu_configs(enable_pipeline=True)
    out_dir = 'results/rerun/bambu_sa_restart'
    os.makedirs(out_dir, exist_ok=True)
    logger = ExperimentLogger(out_dir)
    done = load_done(f'{out_dir}/run_summary.csv')

    for b in BAMBU_BENCH:
        src = f'benchmarks/{b}/{b}.c'
        if not os.path.exists(src):
            log(f'SKIP {b} (no .c)'); continue
        synth = make_bambu_synth(src, b, f'{out_dir}/{b}')
        # Filter with phagocytosis (same as original)
        active, _, _, _ = phagocytosis(configs_all, rules=default_static_rules(),
                                        source_path=src, benchmark_name=b)
        for seed in range(N_SEEDS):
            if done_key('SimulatedAnnealing', b, seed) in done:
                continue
            log(f'{b} / SA-restart / seed={seed}')
            try:
                m = SimulatedAnnealingMethod(active, b, 'bambu', BUDGET, seed=seed)
                safe_run(m, synth, logger,
                         ablation_config='N/A', tau=0, theta=0, n_min=0, p_probe=0)
            except Exception as e:
                log(f'  ERROR init: {e}')

# ──────────────────────────────────────────────
# Part 2: Bambu PA-DSE 10 permutations
# ──────────────────────────────────────────────
def part2_padse_perms():
    log('='*70)
    log('PART 2: Bambu PA-DSE 10 permutations')
    log('='*70)
    configs_all = generate_bambu_configs(enable_pipeline=True)
    out_dir = 'results/rerun/bambu_pa_dse_perms'
    os.makedirs(out_dir, exist_ok=True)
    logger = ExperimentLogger(out_dir)
    done = load_done(f'{out_dir}/run_summary.csv')

    for b in BAMBU_BENCH:
        src = f'benchmarks/{b}/{b}.c'
        if not os.path.exists(src):
            log(f'SKIP {b} (no .c)'); continue
        synth = make_bambu_synth(src, b, f'{out_dir}/{b}')
        for perm in range(N_PERMS):
            if done_key('PA-DSE_SCF+DFRL', b, '', perm) in done:
                continue
            log(f'{b} / PA-DSE / perm={perm}')
            try:
                m = PADSEMethod(configs_all, b, 'bambu', BUDGET,
                                ablation_config='SCF+DFRL',
                                source_path=src,
                                queue_permutation_id=perm)
                safe_run(m, synth, logger, ablation_config='SCF+DFRL')
            except Exception as e:
                log(f'  ERROR: {e}')

if __name__ == '__main__':
    part1_sa_restart()
    part2_padse_perms()
    log('='*70)
    log('RERUN DONE')
    log('='*70)
