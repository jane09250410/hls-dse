#!/usr/bin/env python3
"""Full Dynamatic rerun after Gurobi fix. Stages: GT -> Main -> PA-DSE perms."""
import sys, os, csv, traceback
from datetime import datetime

sys.path.insert(0, 'scripts')

from dynamatic_config_generator import generate_dynamatic_configs
from feasibility_filter import phagocytosis, default_static_rules
from exp_logging.experiment_logger import ExperimentLogger
from runners.run_single import run_single
from runners.run_main_results import make_dynamatic_synth
from methods.baseline_methods import RandomMethod, FilteredRandomMethod
from methods.pa_dse_method import PADSEMethod
from methods.advanced_baselines import (
    SimulatedAnnealingMethod, GeneticAlgorithmMethod,
    GPBayesOptMethod, RFClassifierMethod,
)

DYN_BENCH = ['gcd', 'matching', 'binary_search', 'fir', 'histogram', 'kernel_2mm']
DYN_PATH = os.path.expanduser('~/dynamatic/integration-test')
BUDGET = 30
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
                      r.get('budget',''), r.get('seed',''),
                      r.get('ablation_config',''), r.get('queue_permutation_id','')))
    return done

def is_done(done, strategy, bench, budget, seed='', abl='N/A', perm=''):
    key = (strategy, bench, str(budget),
           str(seed) if seed != '' else '',
           abl, str(perm) if perm != '' else '')
    return key in done

def safe_run(m, synth, logger, **kw):
    try:
        run_single(m, synth, logger, tool='dynamatic', **kw)
        return True
    except Exception as e:
        log(f'  ERROR: {type(e).__name__}: {e}')
        traceback.print_exc()
        return False

def stage_a_ground_truth():
    log('='*70)
    log('STAGE A: Dynamatic Ground Truth (6 bench x 192 configs)')
    log('='*70)
    configs = generate_dynamatic_configs()
    out_dir = 'results/master/dynamatic_ground_truth'
    os.makedirs(out_dir, exist_ok=True)
    logger = ExperimentLogger(out_dir)
    done = load_done(f'{out_dir}/run_summary.csv')

    for b in DYN_BENCH:
        src = f'{DYN_PATH}/{b}/{b}.c'
        if not os.path.exists(src):
            log(f'SKIP {b}: no .c'); continue
        if is_done(done, 'Grid_offline_ref', b, 192):
            log(f'SKIP {b} GT (already done)'); continue
        log(f'{b} ground truth (192 configs)')
        synth = make_dynamatic_synth(src, b, f'{out_dir}/{b}')
        try:
            from methods.baseline_methods import GridMethod
            m = GridMethod(configs, b, 'dynamatic', 192)
        except ImportError:
            m = RandomMethod(configs, b, 'dynamatic', 192, seed=0)
        safe_run(m, synth, logger,
                 ablation_config='N/A', tau=0, theta=0, n_min=0, p_probe=0)

def stage_b_main():
    log('='*70)
    log('STAGE B: Dynamatic Main (7 methods x 10 seeds x 6 bench x B=30)')
    log('='*70)
    configs_all = generate_dynamatic_configs()
    out_dir = 'results/master/dynamatic_main'
    os.makedirs(out_dir, exist_ok=True)
    logger = ExperimentLogger(out_dir)
    done = load_done(f'{out_dir}/run_summary.csv')

    for b in DYN_BENCH:
        src = f'{DYN_PATH}/{b}/{b}.c'
        if not os.path.exists(src):
            log(f'SKIP {b}: no .c'); continue

        active, blocked, _, _ = phagocytosis(configs_all,
                                              rules=default_static_rules(),
                                              source_path=src, benchmark_name=b)
        blocked_ids = {c['id'] for c in blocked}
        configs_filt = [c for c in configs_all if c['id'] not in blocked_ids]

        synth = make_dynamatic_synth(src, b, f'{out_dir}/{b}')

        for cls in [RandomMethod, FilteredRandomMethod,
                    SimulatedAnnealingMethod, GeneticAlgorithmMethod,
                    GPBayesOptMethod, RFClassifierMethod]:
            cls_name = cls.__name__.replace('Method', '')
            possible = [cls_name, cls_name.replace('Algorithm',''),
                        {'GPBayesOpt':'GP-BO','RFClassifier':'RF_Classifier'}.get(cls_name, cls_name)]
            for seed in range(N_SEEDS):
                if any(is_done(done, n, b, BUDGET, seed) for n in possible):
                    continue
                log(f'{b} / {cls.__name__} / seed={seed}')
                try:
                    if cls == RandomMethod:
                        m = cls(configs_all, b, 'dynamatic', BUDGET, seed=seed)
                    elif cls == FilteredRandomMethod:
                        m = cls(configs_all, b, 'dynamatic', BUDGET, seed=seed, source_path=src)
                    else:
                        m = cls(configs_filt, b, 'dynamatic', BUDGET, seed=seed)
                    safe_run(m, synth, logger,
                             ablation_config='N/A', tau=0, theta=0, n_min=0, p_probe=0)
                except Exception as e:
                    log(f'  ERROR init: {e}')

        if not is_done(done, 'PA-DSE_SCF+DFRL', b, BUDGET, 0, 'SCF+DFRL'):
            log(f'{b} / PA-DSE Full')
            try:
                m = PADSEMethod(configs_all, b, 'dynamatic', BUDGET,
                                ablation_config='SCF+DFRL', source_path=src)
                safe_run(m, synth, logger, ablation_config='SCF+DFRL')
            except Exception as e:
                log(f'  ERROR: {e}')

def stage_c_padse_perms():
    log('='*70)
    log('STAGE C: Dynamatic PA-DSE x 10 perms')
    log('='*70)
    configs_all = generate_dynamatic_configs()
    out_dir = 'results/rerun/dynamatic_pa_dse_perms'
    os.makedirs(out_dir, exist_ok=True)
    logger = ExperimentLogger(out_dir)
    done = load_done(f'{out_dir}/run_summary.csv')

    for b in DYN_BENCH:
        src = f'{DYN_PATH}/{b}/{b}.c'
        if not os.path.exists(src):
            log(f'SKIP {b}: no .c'); continue
        synth = make_dynamatic_synth(src, b, f'{out_dir}/{b}')
        for perm in range(N_PERMS):
            if is_done(done, 'PA-DSE_SCF+DFRL', b, BUDGET, '', 'SCF+DFRL', perm):
                continue
            log(f'{b} / PA-DSE / perm={perm}')
            try:
                m = PADSEMethod(configs_all, b, 'dynamatic', BUDGET,
                                ablation_config='SCF+DFRL',
                                source_path=src,
                                queue_permutation_id=perm)
                safe_run(m, synth, logger, ablation_config='SCF+DFRL')
            except Exception as e:
                log(f'  ERROR: {e}')

if __name__ == '__main__':
    stage_a_ground_truth()
    stage_b_main()
    stage_c_padse_perms()
    log('='*70)
    log('DYNAMATIC FULL RERUN DONE')
    log('='*70)
