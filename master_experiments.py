#!/usr/bin/env python3
"""
master_experiments.py — Unified experiment runner for PA-DSE paper.

Runs all experiments in priority order with resume capability.
Designed for unattended multi-day execution on Azure.

Stages (in order):
  1. Ground Truth     — 4 new Bambu benchmarks + kernel_2mm
  2. Bambu Main       — 8 bench × 8 methods × 10 seeds × B=60
  3. Dynamatic Main   — 6 bench × 8 methods × 10 seeds × B=30
  4. Ablation         — 8 configs × 4 bench × 3 seeds × B=60
  5. Budget Sweep     — 6 B values × 3 methods × 2 bench × 5 seeds
  6. Sensitivity      — θ, τ, n_min sensitivity
  7. Probe Sensitivity — p_probe sweep

Resume logic: scans run_summary.csv at each stage. Already-completed runs are skipped.
Safe to Ctrl+C and restart — picks up where it left off.

Usage:
  cd ~/hls-dse
  nohup python3 /path/to/master_experiments.py > master.log 2>&1 &

  # Monitor:
  tail -f master.log
  wc -l results/master/*/run_summary.csv
"""

import sys
import os
import csv
import traceback
from datetime import datetime

sys.path.insert(0, 'scripts')

from config_generator import generate_bambu_configs
from dynamatic_config_generator import generate_dynamatic_configs
from feasibility_filter import phagocytosis
from exp_logging.experiment_logger import ExperimentLogger
from runners.run_single import run_single
from runners.run_main_results import make_bambu_synth, make_dynamatic_synth
from methods.baseline_methods import RandomMethod, FilteredRandomMethod
from methods.pa_dse_method import PADSEMethod
from methods.advanced_baselines import (
    SimulatedAnnealingMethod, GeneticAlgorithmMethod,
    GPBayesOptMethod, RFClassifierMethod,
)

# ──────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────

BAMBU_BENCH = ['matmul', 'vadd', 'fir', 'histogram', 'bicg', 'atax', 'gemm', 'gesummv']
NEW_BAMBU_BENCH = ['bicg', 'atax', 'gemm', 'gesummv']  # need ground truth
DYN_BENCH = ['gcd', 'matching', 'binary_search', 'fir', 'histogram', 'kernel_2mm']

DYN_PATH = os.path.expanduser('~/dynamatic/integration-test')


# ──────────────────────────────────────────────────────────
# Resume logic
# ──────────────────────────────────────────────────────────

def load_completed(csv_path):
    """Load set of completed (strategy, benchmark, budget, seed, ablation_config) tuples."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row.get('strategy', ''),
                row.get('benchmark', ''),
                row.get('budget', ''),
                row.get('seed', ''),
                row.get('ablation_config', ''),
                row.get('queue_permutation_id', ''),
            )
            done.add(key)
    return done


def is_done(done_set, strategy, benchmark, budget, seed, ablation='N/A', perm=''):
    key = (strategy, benchmark, str(budget), str(seed) if seed is not None else '',
           ablation, str(perm) if perm else '')
    return key in done_set


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f'[{ts}] {msg}', flush=True)


# ──────────────────────────────────────────────────────────
# Helper: run a single method with exception handling
# ──────────────────────────────────────────────────────────

def safe_run(method, synth, logger, tool, **kwargs):
    try:
        run_single(method, synth, logger, tool=tool, **kwargs)
        return True
    except Exception as e:
        log(f'  ERROR: {type(e).__name__}: {e}')
        traceback.print_exc()
        return False


# ──────────────────────────────────────────────────────────
# Stage 1: Ground Truth (new Bambu + kernel_2mm)
# ──────────────────────────────────────────────────────────

def stage_1_ground_truth():
    log('=' * 70)
    log('STAGE 1: Ground Truth')
    log('=' * 70)

    # Import GridMethod only in this stage (in case it's optional)
    try:
        from methods.baseline_methods import GridMethod
    except ImportError:
        log('GridMethod not found, using RandomMethod with full budget instead')
        GridMethod = None

    # Bambu new benchmarks
    configs_b = generate_bambu_configs(enable_pipeline=True)
    logger = ExperimentLogger('results/master/ground_truth')
    done = load_completed('results/master/ground_truth/run_summary.csv')

    for b in NEW_BAMBU_BENCH:
        src = f'benchmarks/{b}/{b}.c'
        if not os.path.exists(src):
            log(f'SKIP {b}: missing .c file')
            continue
        if is_done(done, 'Grid' if GridMethod else 'Random', b, 420, 0):
            log(f'SKIP {b} ground truth (already done)')
            continue
        log(f'{b} ground truth (420 configs)')
        synth = make_bambu_synth(src, b, f'results/master/ground_truth/{b}')
        if GridMethod:
            m = GridMethod(configs_b, b, 'bambu', 420)
        else:
            m = RandomMethod(configs_b, b, 'bambu', 420, seed=0)
        safe_run(m, synth, logger, 'bambu',
                 ablation_config='N/A', tau=0, theta=0, n_min=0, p_probe=0)

    # Dynamatic kernel_2mm
    configs_d = generate_dynamatic_configs()
    src = f'{DYN_PATH}/kernel_2mm/kernel_2mm.c'
    if os.path.exists(src):
        if not is_done(done, 'Grid' if GridMethod else 'Random', 'kernel_2mm', 192, 0):
            log(f'kernel_2mm ground truth (192 configs)')
            synth = make_dynamatic_synth(src, 'kernel_2mm', f'results/master/ground_truth/kernel_2mm')
            if GridMethod:
                m = GridMethod(configs_d, 'kernel_2mm', 'dynamatic', 192)
            else:
                m = RandomMethod(configs_d, 'kernel_2mm', 'dynamatic', 192, seed=0)
            safe_run(m, synth, logger, 'dynamatic',
                     ablation_config='N/A', tau=0, theta=0, n_min=0, p_probe=0)


# ──────────────────────────────────────────────────────────
# Stage 2: Bambu Main
# ──────────────────────────────────────────────────────────

def stage_2_bambu_main():
    log('=' * 70)
    log('STAGE 2: Bambu Main (8 bench × 8 methods × 10 seeds × B=60)')
    log('=' * 70)

    configs_all = generate_bambu_configs(enable_pipeline=True)
    logger = ExperimentLogger('results/master/bambu_main')
    done = load_completed('results/master/bambu_main/run_summary.csv')

    N_SEEDS = 10
    BUDGET = 60

    for b in BAMBU_BENCH:
        src = f'benchmarks/{b}/{b}.c'
        if not os.path.exists(src):
            log(f'SKIP {b}: missing .c file')
            continue

        # Compute filtered configs once per benchmark
        active, blocked, suppressed, _ = phagocytosis(
            configs_all, source_path=src, benchmark_name=b)
        blocked_ids = {c['id'] for c in blocked}
        configs_filt = [c for c in configs_all if c['id'] not in blocked_ids]

        synth = make_bambu_synth(src, b, f'results/master/bambu_main/{b}')

        # Stochastic baselines (with seeds)
        for cls, use_filt in [
            (RandomMethod, False),
            (FilteredRandomMethod, True),  # uses phagocytosis internally
            (SimulatedAnnealingMethod, True),
            (GeneticAlgorithmMethod, True),
            (GPBayesOptMethod, True),
            (RFClassifierMethod, True),
        ]:
            cls_name = cls.__name__.replace('Method', '')
            if cls_name.endswith('Method'): cls_name = cls_name[:-6]
            # SimulatedAnnealingMethod → SimulatedAnnealing per run_summary.csv convention
            # Check alternatives
            possible_names = [
                cls_name,
                cls_name.replace('Algorithm', ''),
                {'GPBayesOpt': 'GP-BO', 'RFClassifier': 'RF_Classifier'}.get(cls_name, cls_name),
            ]

            for seed in range(N_SEEDS):
                # Check if done (try multiple strategy name variants)
                already = any(is_done(done, n, b, BUDGET, seed) for n in possible_names)
                if already:
                    continue

                log(f'{b} / {cls.__name__} / seed={seed}')
                try:
                    if cls == RandomMethod:
                        m = cls(configs_all, b, 'bambu', BUDGET, seed=seed)
                    elif cls == FilteredRandomMethod:
                        m = cls(configs_all, b, 'bambu', BUDGET, seed=seed, source_path=src)
                    else:
                        m = cls(configs_filt, b, 'bambu', BUDGET, seed=seed)
                    safe_run(m, synth, logger, 'bambu',
                             ablation_config='N/A', tau=0, theta=0, n_min=0, p_probe=0)
                except Exception as e:
                    log(f'  ERROR init: {e}')

        # PA-DSE L1 (intersection mode, phago only - no DFRL)
        if not is_done(done, 'PA-DSE_SCF+DFRL(L1)', b, BUDGET, 0, 'SCF+DFRL(L1)'):
            log(f'{b} / PA-DSE L1')
            try:
                m = PADSEMethod(configs_all, b, 'bambu', BUDGET,
                                ablation_config='SCF+DFRL',
                                dynamic_mode='intersection',
                                source_path=src)
                safe_run(m, synth, logger, 'bambu',
                         ablation_config='SCF+DFRL(L1)')
            except Exception as e:
                log(f'  ERROR: {e}')

        # PA-DSE Full
        if not is_done(done, 'PA-DSE_SCF+DFRL', b, BUDGET, 0, 'SCF+DFRL'):
            log(f'{b} / PA-DSE Full')
            try:
                m = PADSEMethod(configs_all, b, 'bambu', BUDGET,
                                ablation_config='SCF+DFRL',
                                source_path=src)
                safe_run(m, synth, logger, 'bambu')
            except Exception as e:
                log(f'  ERROR: {e}')


# ──────────────────────────────────────────────────────────
# Stage 3: Dynamatic Main
# ──────────────────────────────────────────────────────────

def stage_3_dynamatic_main():
    log('=' * 70)
    log('STAGE 3: Dynamatic Main (6 bench × 8 methods × 10 seeds × B=30)')
    log('=' * 70)

    configs_all = generate_dynamatic_configs()
    logger = ExperimentLogger('results/master/dynamatic_main')
    done = load_completed('results/master/dynamatic_main/run_summary.csv')

    N_SEEDS = 10
    BUDGET = 30

    for b in DYN_BENCH:
        src = f'{DYN_PATH}/{b}/{b}.c'
        if not os.path.exists(src):
            log(f'SKIP {b}: missing .c file')
            continue

        # Filter for advanced baselines
        active, blocked, suppressed, _ = phagocytosis(
            configs_all, source_path=src, benchmark_name=b)
        blocked_ids = {c['id'] for c in blocked}
        configs_filt = [c for c in configs_all if c['id'] not in blocked_ids]

        synth = make_dynamatic_synth(src, b, f'results/master/dynamatic_main/{b}')

        for cls in [RandomMethod, FilteredRandomMethod,
                    SimulatedAnnealingMethod, GeneticAlgorithmMethod,
                    GPBayesOptMethod, RFClassifierMethod]:
            cls_name = cls.__name__.replace('Method', '')
            possible_names = [
                cls_name,
                cls_name.replace('Algorithm', ''),
                {'GPBayesOpt': 'GP-BO', 'RFClassifier': 'RF_Classifier'}.get(cls_name, cls_name),
            ]

            for seed in range(N_SEEDS):
                already = any(is_done(done, n, b, BUDGET, seed) for n in possible_names)
                if already:
                    continue
                log(f'{b} / {cls.__name__} / seed={seed}')
                try:
                    if cls == RandomMethod:
                        m = cls(configs_all, b, 'dynamatic', BUDGET, seed=seed)
                    elif cls == FilteredRandomMethod:
                        m = cls(configs_all, b, 'dynamatic', BUDGET, seed=seed, source_path=src)
                    else:
                        m = cls(configs_filt, b, 'dynamatic', BUDGET, seed=seed)
                    safe_run(m, synth, logger, 'dynamatic',
                             ablation_config='N/A', tau=0, theta=0, n_min=0, p_probe=0)
                except Exception as e:
                    log(f'  ERROR init: {e}')

        # PA-DSE L1
        if not is_done(done, 'PA-DSE_SCF+DFRL(L1)', b, BUDGET, 0, 'SCF+DFRL(L1)'):
            log(f'{b} / PA-DSE L1')
            try:
                m = PADSEMethod(configs_all, b, 'dynamatic', BUDGET,
                                ablation_config='SCF+DFRL',
                                dynamic_mode='intersection',
                                source_path=src)
                safe_run(m, synth, logger, 'dynamatic',
                         ablation_config='SCF+DFRL(L1)')
            except Exception as e:
                log(f'  ERROR: {e}')

        # PA-DSE Full
        if not is_done(done, 'PA-DSE_SCF+DFRL', b, BUDGET, 0, 'SCF+DFRL'):
            log(f'{b} / PA-DSE Full')
            try:
                m = PADSEMethod(configs_all, b, 'dynamatic', BUDGET,
                                ablation_config='SCF+DFRL',
                                source_path=src)
                safe_run(m, synth, logger, 'dynamatic')
            except Exception as e:
                log(f'  ERROR: {e}')


# ──────────────────────────────────────────────────────────
# Stage 4: Ablation
# ──────────────────────────────────────────────────────────

def stage_4_ablation():
    log('=' * 70)
    log('STAGE 4: Ablation (8 configs × 4 bench × 3 seeds)')
    log('=' * 70)

    ABLATIONS = ['no-filter', 'SCF-only', 'SCF+RPE', 'SCF+OFRS',
                 'SCF+DFRL', 'DFRL-only', 'SCF+RPE-reorder', 'SCF+OFRS-skip']
    N_SEEDS = 3

    logger = ExperimentLogger('results/master/ablation')
    done = load_completed('results/master/ablation/run_summary.csv')

    # Bambu ablation: vadd + matmul, B=60
    configs_b = generate_bambu_configs(enable_pipeline=True)
    for b in BAMBU_BENCH:
        src = f'benchmarks/{b}/{b}.c'
        synth = make_bambu_synth(src, b, f'results/master/ablation/{b}')
        for abl in ABLATIONS:
            for seed in range(N_SEEDS):
                if is_done(done, f'PA-DSE_{abl}', b, 60, seed, abl, str(seed)):
                    continue
                log(f'{b} / {abl} / perm={seed}')
                try:
                    m = PADSEMethod(configs_b, b, 'bambu', 60,
                                    ablation_config=abl,
                                    source_path=src,
                                    queue_permutation_id=seed)
                    safe_run(m, synth, logger, 'bambu',
                             ablation_config=abl, queue_permutation_id=seed)
                except Exception as e:
                    log(f'  ERROR: {e}')

    # Dynamatic ablation: gcd + matching, B=60
    configs_d = generate_dynamatic_configs()
    for b in ['gcd', 'matching']:
        src = f'{DYN_PATH}/{b}/{b}.c'
        synth = make_dynamatic_synth(src, b, f'results/master/ablation/{b}')
        for abl in ABLATIONS:
            for seed in range(N_SEEDS):
                if is_done(done, f'PA-DSE_{abl}', b, 60, seed, abl, str(seed)):
                    continue
                log(f'{b} / {abl} / perm={seed}')
                try:
                    m = PADSEMethod(configs_d, b, 'dynamatic', 60,
                                    ablation_config=abl,
                                    source_path=src,
                                    queue_permutation_id=seed)
                    safe_run(m, synth, logger, 'dynamatic',
                             ablation_config=abl, queue_permutation_id=seed)
                except Exception as e:
                    log(f'  ERROR: {e}')


# ──────────────────────────────────────────────────────────
# Stage 5: Budget Sweep
# ──────────────────────────────────────────────────────────

def stage_5_budget_sweep():
    log('=' * 70)
    log('STAGE 5: Budget Sweep (6 B values × 3 methods × 2 bench × 5 seeds)')
    log('=' * 70)

    BUDGETS = [20, 40, 60, 80, 100, 120]
    N_SEEDS = 5
    CONFIGS_TO_TEST = ['SCF+RPE', 'SCF+OFRS', 'SCF+DFRL']

    logger = ExperimentLogger('results/master/budget_sweep')
    done = load_completed('results/master/budget_sweep/run_summary.csv')

    # Bambu: vadd
    configs_b = generate_bambu_configs(enable_pipeline=True)
    src = 'benchmarks/vadd/vadd.c'
    synth = make_bambu_synth(src, 'vadd', 'results/master/budget_sweep/vadd')
    for B in BUDGETS:
        for abl in CONFIGS_TO_TEST:
            for seed in range(N_SEEDS):
                if is_done(done, f'PA-DSE_{abl}', 'vadd', B, seed, abl, str(seed)):
                    continue
                log(f'vadd / B={B} / {abl} / perm={seed}')
                try:
                    m = PADSEMethod(configs_b, 'vadd', 'bambu', B,
                                    ablation_config=abl,
                                    source_path=src,
                                    queue_permutation_id=seed)
                    safe_run(m, synth, logger, 'bambu',
                             ablation_config=abl, queue_permutation_id=seed)
                except Exception as e:
                    log(f'  ERROR: {e}')

    # Dynamatic: gcd (budget up to config space size)
    configs_d = generate_dynamatic_configs()
    src = f'{DYN_PATH}/gcd/gcd.c'
    synth = make_dynamatic_synth(src, 'gcd', 'results/master/budget_sweep/gcd')
    for B in BUDGETS:
        if B > 120: continue  # Dynamatic space is 192 but keep symmetry
        for abl in CONFIGS_TO_TEST:
            for seed in range(N_SEEDS):
                if is_done(done, f'PA-DSE_{abl}', 'gcd', B, seed, abl, str(seed)):
                    continue
                log(f'gcd / B={B} / {abl} / perm={seed}')
                try:
                    m = PADSEMethod(configs_d, 'gcd', 'dynamatic', B,
                                    ablation_config=abl,
                                    source_path=src,
                                    queue_permutation_id=seed)
                    safe_run(m, synth, logger, 'dynamatic',
                             ablation_config=abl, queue_permutation_id=seed)
                except Exception as e:
                    log(f'  ERROR: {e}')


# ──────────────────────────────────────────────────────────
# Stage 6: Sensitivity (θ, τ, n_min)
# ──────────────────────────────────────────────────────────

def stage_6_sensitivity():
    log('=' * 70)
    log('STAGE 6: Sensitivity (θ, τ, n_min)')
    log('=' * 70)

    N_SEEDS = 3
    B = 60
    BENCH = [('vadd', 'bambu'), ('gcd', 'dynamatic')]

    SENS_CONFIGS = [
        # (param_name, param_value)
        ('theta', 0.7), ('theta', 0.8), ('theta', 0.9),
        ('tau', 2), ('tau', 3), ('tau', 4), ('tau', 5),
        ('n_min', 3), ('n_min', 5), ('n_min', 8),
    ]

    logger = ExperimentLogger('results/master/sensitivity')
    done = load_completed('results/master/sensitivity/run_summary.csv')

    configs_b = generate_bambu_configs(enable_pipeline=True)
    configs_d = generate_dynamatic_configs()

    for bench, tool in BENCH:
        if tool == 'bambu':
            src = f'benchmarks/{bench}/{bench}.c'
            configs = configs_b
            synth = make_bambu_synth(src, bench, f'results/master/sensitivity/{bench}')
        else:
            src = f'{DYN_PATH}/{bench}/{bench}.c'
            configs = configs_d
            synth = make_dynamatic_synth(src, bench, f'results/master/sensitivity/{bench}')

        for pname, pval in SENS_CONFIGS:
            for seed in range(N_SEEDS):
                # Use default ablation but varied param — check by run_id
                run_tag = f'PA-DSE_SCF+DFRL_{pname}_{pval}'
                if is_done(done, run_tag, bench, B, seed, 'SCF+DFRL', str(seed)):
                    continue
                log(f'{bench} / {pname}={pval} / perm={seed}')
                try:
                    kwargs = {
                        'configs': configs, 'benchmark_name': bench, 'tool': tool,
                        'budget': B, 'ablation_config': 'SCF+DFRL',
                        'source_path': src, 'queue_permutation_id': seed,
                    }
                    kwargs[pname] = pval
                    m = PADSEMethod(**kwargs)
                    safe_run(m, synth, logger, tool,
                             ablation_config='SCF+DFRL',
                             queue_permutation_id=seed,
                             **{pname: pval})
                except Exception as e:
                    log(f'  ERROR: {e}')


# ──────────────────────────────────────────────────────────
# Stage 7: Probe Sensitivity
# ──────────────────────────────────────────────────────────

def stage_7_probe():
    log('=' * 70)
    log('STAGE 7: Probe Sensitivity')
    log('=' * 70)

    PROBES = [0.0, 0.02, 0.05, 0.10, 0.20]
    N_SEEDS = 5
    B = 60

    logger = ExperimentLogger('results/master/probe')
    done = load_completed('results/master/probe/run_summary.csv')

    configs_b = generate_bambu_configs(enable_pipeline=True)
    configs_d = generate_dynamatic_configs()

    for bench, tool in [('vadd', 'bambu'), ('gcd', 'dynamatic')]:
        if tool == 'bambu':
            src = f'benchmarks/{bench}/{bench}.c'
            configs = configs_b
            synth = make_bambu_synth(src, bench, f'results/master/probe/{bench}')
        else:
            src = f'{DYN_PATH}/{bench}/{bench}.c'
            configs = configs_d
            synth = make_dynamatic_synth(src, bench, f'results/master/probe/{bench}')

        for p in PROBES:
            for seed in range(N_SEEDS):
                log(f'{bench} / p_probe={p} / perm={seed}')
                try:
                    m = PADSEMethod(configs, bench, tool, B,
                                    ablation_config='SCF+DFRL',
                                    source_path=src,
                                    queue_permutation_id=seed,
                                    p_probe=p)
                    safe_run(m, synth, logger, tool,
                             ablation_config='SCF+DFRL',
                             queue_permutation_id=seed,
                             p_probe=p)
                except Exception as e:
                    log(f'  ERROR: {e}')


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    stages = [
        ('ground_truth', stage_1_ground_truth),
        ('bambu_main', stage_2_bambu_main),
        ('dynamatic_main', stage_3_dynamatic_main),
        ('ablation', stage_4_ablation),
        ('budget_sweep', stage_5_budget_sweep),
        ('sensitivity', stage_6_sensitivity),
        ('probe', stage_7_probe),
    ]

    # Allow running a specific stage: python3 master_experiments.py bambu_main
    if len(sys.argv) > 1:
        target = sys.argv[1]
        stages_to_run = [(n, f) for n, f in stages if n == target]
        if not stages_to_run:
            print(f'Unknown stage: {target}')
            print(f'Available stages: {[n for n, _ in stages]}')
            sys.exit(1)
    else:
        stages_to_run = stages

    log('Starting master experiments')
    log(f'Stages: {[n for n, _ in stages_to_run]}')

    for name, func in stages_to_run:
        try:
            func()
            log(f'Stage {name}: COMPLETED')
        except KeyboardInterrupt:
            log(f'Stage {name}: INTERRUPTED')
            sys.exit(0)
        except Exception as e:
            log(f'Stage {name}: FAILED - {e}')
            traceback.print_exc()
            log('Continuing to next stage...')

    log('=' * 70)
    log('ALL STAGES COMPLETED')
    log('=' * 70)
