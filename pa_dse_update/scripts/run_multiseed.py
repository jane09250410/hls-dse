#!/usr/bin/env python3
"""
run_multiseed.py
================
Multi-seed PA-DSE experiment runner with comprehensive metrics.

Key features:
  - Multi-seed for Random/LHS (default 10 seeds) for mean +/- std
  - Budget sweep (e.g. B=20,40,60,80) to test with more configs
  - Extended config spaces (optional --extended-space)
  - Rich metrics: wall_clock, TTFF, wasted_time, hypervolume, budget_util,
    failure_breakdown, pareto_points, speedup, mean_success/failure_time

Usage (Bambu):
  python3 scripts/run_multiseed.py --tool bambu \
      --benchmarks matmul fir vadd histogram \
      --budgets 20 40 60 --num-seeds 10 --enable-pipeline

Usage (Dynamatic):
  python3 scripts/run_multiseed.py --tool dynamatic \
      --benchmarks gcd matching binary_search fir histogram \
      --budgets 30 60 90 --num-seeds 10

Usage (extended space + large budget):
  python3 scripts/run_multiseed.py --tool bambu --benchmarks matmul \
      --budgets 20 40 60 80 100 --num-seeds 10 --enable-pipeline --extended-space
"""

from __future__ import annotations

import argparse, csv, json, os, random, sys, time
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# ── existing codebase imports ───────────────────────────────
from config_generator import generate_bambu_configs, config_to_bambu_cmd
from run_exploration import run_bambu_single, extract_bambu_metrics
from analyze import find_pareto
from feasibility_filter import phagocytosis, default_static_rules
from pattern_learner import FailurePatternLearner, extract_error_type

try:
    from dynamatic_config_generator import (
        generate_dynamatic_configs, config_to_label,
        dynamatic_static_rules, apply_static_rules,
        analyze_source_for_dynamatic,
    )
    from run_dynamatic_single import run_dynamatic_single
    HAS_DYNAMATIC = True
except ImportError:
    HAS_DYNAMATIC = False

Config = Dict[str, Any]

# ════════════════════════════════════════════════════════════
# Hypervolume (2-D, both objectives minimized)
# ════════════════════════════════════════════════════════════

def compute_hypervolume_2d(points, ref):
    pts = sorted([(x, y) for x, y in points if x < ref[0] and y < ref[1]])
    if not pts:
        return 0.0
    pareto, best_y = [], float('inf')
    for x, y in pts:
        if y < best_y:
            pareto.append((x, y))
            best_y = y
    hv = 0.0
    for i, (x, y) in enumerate(pareto):
        x_next = pareto[i+1][0] if i+1 < len(pareto) else ref[0]
        hv += (x_next - x) * (ref[1] - y)
    return hv

# ════════════════════════════════════════════════════════════
# Extended config space
# ════════════════════════════════════════════════════════════

def generate_bambu_configs_extended(enable_pipeline=False):
    import itertools
    configs, cid = [], 0
    clocks = [3, 5, 8, 10, 12, 15, 20]
    mems   = ['ALL_BRAM', 'NO_BRAM']
    chtypes = ['MEM_ACC_11', 'MEM_ACC_N1', 'MEM_ACC_NN']
    chnums  = [1, 2]
    for cp, mem, ct, cn in itertools.product(clocks, mems, chtypes, chnums):
        configs.append({'id':cid,'tool':'bambu','clock_period':cp,
            'pipeline':False,'pipeline_ii':None,'memory_policy':mem,
            'channels_type':ct,'channels_number':cn})
        cid += 1
    if enable_pipeline:
        for cp, ii, mem, ct, cn in itertools.product(clocks,[1,2,3,4],mems,chtypes,chnums):
            configs.append({'id':cid,'tool':'bambu','clock_period':cp,
                'pipeline':True,'pipeline_ii':ii,'memory_policy':mem,
                'channels_type':ct,'channels_number':cn})
            cid += 1
    return configs

# ════════════════════════════════════════════════════════════
# Benchmark registries
# ════════════════════════════════════════════════════════════

BAMBU_BM = {
    "matmul":("benchmarks/matmul/matmul.c","matmul"),
    "fir":("benchmarks/fir/fir.c","fir"),
    "vadd":("benchmarks/vadd/vadd.c","vadd"),
    "histogram":("benchmarks/histogram/histogram.c","histogram"),
    "dotprod":("benchmarks/dotprod/dotprod.c","dotprod"),
    "stencil":("benchmarks/stencil/stencil.c","stencil"),
    "matvec":("benchmarks/matvec/matvec.c","matvec"),
    "atax":("benchmarks/atax/atax.c","atax"),
    "gemm":("benchmarks/gemm/gemm.c","gemm"),
    "spmv":("benchmarks/spmv/spmv.c","spmv"),
}
DYN_PATH = os.path.expanduser("~/dynamatic")
DYN_BM = {
    "gcd":(f"{DYN_PATH}/integration-test/gcd/gcd.c","gcd"),
    "matching":(f"{DYN_PATH}/integration-test/matching/matching.c","matching"),
    "binary_search":(f"{DYN_PATH}/integration-test/binary_search/binary_search.c","binary_search"),
    "fir":(f"{DYN_PATH}/integration-test/fir/fir.c","fir"),
    "histogram":(f"{DYN_PATH}/integration-test/histogram/histogram.c","histogram"),
}

# ════════════════════════════════════════════════════════════
# Sampling
# ════════════════════════════════════════════════════════════

def grid_sample(cfgs, B):
    step = max(1, len(cfgs)//B)
    return [cfgs[i] for i in range(0,len(cfgs),step)][:B]

def rand_sample(cfgs, B, seed):
    return random.Random(seed).sample(cfgs, min(B, len(cfgs)))

def lhs_sample(cfgs, B, seed):
    rng = random.Random(seed)
    n = len(cfgs)
    if B >= n: return list(cfgs)
    ss = n/B
    return [cfgs[rng.randint(int(i*ss), min(int((i+1)*ss)-1, n-1))] for i in range(B)]

# ════════════════════════════════════════════════════════════
# RunMetrics dataclass
# ════════════════════════════════════════════════════════════

@dataclass
class RunMetrics:
    strategy:str; benchmark:str; seed:int; tool:str; budget:int=0; total_config_space:int=0
    total_evaluated:int=0; successful:int=0; failed:int=0; success_rate_pct:float=0.0
    wall_clock_s:float=0.0; time_to_first_feasible_s:float=-1.0
    wasted_time_on_failures_s:float=0.0
    mean_eval_time_s:float=0.0; mean_success_time_s:float=0.0; mean_failure_time_s:float=0.0
    budget_utilization_pct:float=0.0
    best_area:Optional[float]=None; mean_area:Optional[float]=None
    best_latency:Optional[float]=None; mean_latency:Optional[float]=None
    unique_qor_points:int=0; pareto_points:int=0; hypervolume:float=0.0
    best_components:Optional[float]=None; mean_components:Optional[float]=None
    best_buffers:Optional[float]=None; mean_buffers:Optional[float]=None
    static_blocked:int=0; static_suppressed:int=0; autophagy_suppressed:int=0
    failure_breakdown:str="{}"

# ════════════════════════════════════════════════════════════
# Post-run QoR collection
# ════════════════════════════════════════════════════════════

def _bambu_qor(results, m, ref_a=1000.0, ref_l=50.0):
    import pandas as pd
    ok = [r for r in results if r['success']]
    areas = [r['total_area'] for r in ok if r.get('total_area') is not None]
    states = [r['num_states'] for r in ok if r.get('num_states') is not None]
    pairs = set()
    for r in ok:
        a,s = r.get('total_area'), r.get('num_states')
        if a is not None and s is not None:
            pairs.add((a,s))
    if areas: m.best_area=min(areas); m.mean_area=round(np.mean(areas),1)
    if states: m.best_latency=min(states); m.mean_latency=round(np.mean(states),1)
    m.unique_qor_points = len(pairs)
    if pairs:
        df_ok = pd.DataFrame([r for r in ok if r.get('total_area') is not None])
        if not df_ok.empty:
            pareto = find_pareto(df_ok, 'total_area', 'num_states')
            m.pareto_points = len(pareto)
        m.hypervolume = round(compute_hypervolume_2d(list(pairs), (ref_a, ref_l)), 1)

def _failure_bkdn(results, m):
    bd = {}
    for r in results:
        if not r['success']:
            et = extract_error_type(r.get('output',''))
            bd[et] = bd.get(et,0)+1
    m.failure_breakdown = json.dumps(bd)

def _timing(m, t0, first_t, stimes, ftimes):
    m.wall_clock_s = round(time.time()-t0, 2)
    m.time_to_first_feasible_s = round(first_t, 2) if first_t is not None else -1.0
    m.wasted_time_on_failures_s = round(m.wasted_time_on_failures_s, 2)
    m.mean_eval_time_s = round(m.wall_clock_s/max(1,m.total_evaluated), 2)
    m.mean_success_time_s = round(np.mean(stimes),2) if stimes else 0.0
    m.mean_failure_time_s = round(np.mean(ftimes),2) if ftimes else 0.0
    if m.total_evaluated > 0:
        m.success_rate_pct = round(100*m.successful/m.total_evaluated, 1)
    m.budget_utilization_pct = round(100*m.successful/max(1,m.budget), 1)

# ════════════════════════════════════════════════════════════
# Bambu runners
# ════════════════════════════════════════════════════════════

def _run_bambu_list(cfgs, src, top, bm, strat, seed, B, N, wk):
    absrc = os.path.abspath(src)
    m = RunMetrics(strategy=strat,benchmark=bm,seed=seed,tool="bambu",budget=B,total_config_space=N)
    results=[]; t0=time.time(); ft=None; st=[]; flt=[]
    for cfg in cfgs:
        wd = os.path.join(wk, f"config_{cfg['id']}")
        cmd = config_to_bambu_cmd(cfg, absrc, top)
        out, el, ok = run_bambu_single(cmd, wd)
        qor = extract_bambu_metrics(out)
        results.append({**cfg, **qor, 'runtime_s':el, 'success':ok, 'output':out})
        if ok:
            m.successful+=1; st.append(el)
            if ft is None: ft=time.time()-t0
        else:
            m.failed+=1; flt.append(el); m.wasted_time_on_failures_s+=el
        m.total_evaluated+=1
    _timing(m,t0,ft,st,flt); _bambu_qor(results,m); _failure_bkdn(results,m)
    return m

def _run_padse_bambu(cfgs, src, top, bm, B, N, wk, abl="full"):
    absrc = os.path.abspath(src)
    if abl=="no_filter":
        act,blk,sup,_=list(cfgs),[],[],[]
    else:
        act,blk,sup,_ = phagocytosis(configs=cfgs, rules=default_static_rules(),
                                       source_path=src, benchmark_name=bm)
    queue = list(act)+list(sup)
    learner = FailurePatternLearner(threshold=2)
    m = RunMetrics(strategy="PA-DSE",benchmark=bm,seed=0,tool="bambu",budget=B,total_config_space=N)
    m.static_blocked=len(blk); m.static_suppressed=len(sup)
    results=[]; t0=time.time(); ft=None; st=[]; flt=[]; n=0
    while queue and n < B:
        cfg = queue.pop(0); n+=1
        wd = os.path.join(wk, f"config_{cfg['id']}")
        cmd = config_to_bambu_cmd(cfg, absrc, top)
        out, el, ok = run_bambu_single(cmd, wd)
        qor = extract_bambu_metrics(out)
        results.append({**cfg, **qor, 'runtime_s':el, 'success':ok, 'output':out})
        if ok:
            m.successful+=1; st.append(el)
            if ft is None: ft=time.time()-t0
        else:
            m.failed+=1; flt.append(el); m.wasted_time_on_failures_s+=el
            if abl=="full":
                pat = learner.add_failure(config=cfg,output=out,runtime_s=el,benchmark_name=bm)
                if pat:
                    bf=len(queue)
                    queue=[c for c in queue if not pat.matches(c, bm)]
                    m.autophagy_suppressed+=(bf-len(queue))
        m.total_evaluated+=1
    _timing(m,t0,ft,st,flt); _bambu_qor(results,m); _failure_bkdn(results,m)
    return m

# ════════════════════════════════════════════════════════════
# Dynamatic runners
# ════════════════════════════════════════════════════════════

def _run_dyn_list(cfgs, src, top, bm, strat, seed, B, N, wk):
    m = RunMetrics(strategy=strat,benchmark=bm,seed=seed,tool="dynamatic",budget=B,total_config_space=N)
    t0=time.time(); ft=None; cl=[]; bl=[]; st=[]; flt=[]; ftypes={}
    for cfg in cfgs:
        ok, met, out = run_dynamatic_single(config=cfg, src_file=src, top_func=top)
        ct = met.get('compile_time_s',0)
        if ok:
            m.successful+=1; st.append(ct)
            cl.append(met.get('num_components',0)); bl.append(met.get('num_buffers',0))
            if ft is None: ft=time.time()-t0
        else:
            m.failed+=1; flt.append(ct); m.wasted_time_on_failures_s+=ct
            et=met.get('error_type','unknown'); ftypes[et]=ftypes.get(et,0)+1
        m.total_evaluated+=1
    _timing(m,t0,ft,st,flt)
    if cl: m.best_components=min(cl); m.mean_components=round(np.mean(cl),1)
    if bl: m.best_buffers=min(bl); m.mean_buffers=round(np.mean(bl),1)
    m.failure_breakdown=json.dumps(ftypes)
    return m

def _run_padse_dyn(cfgs, src, top, bm, B, N, wk, abl="full"):
    from run_dynamatic_dse import DynamaticPatternLearner
    rules = dynamatic_static_rules()
    ci = analyze_source_for_dynamatic(src)
    if ci.get("has_while_loop"):
        rules.append({"name":"R_CA1","severity":"suppress",
            "description":"while+MILP","condition":lambda c:c["buffer_algorithm"] in ("fpga20","fpl22")})
    if abl=="no_filter":
        act,blk,sup=list(cfgs),[],[]
    else:
        act,blk,sup,_=apply_static_rules(cfgs,rules)
    queue=list(act)+list(sup)
    learner=DynamaticPatternLearner(threshold=2)
    m = RunMetrics(strategy="PA-DSE",benchmark=bm,seed=0,tool="dynamatic",budget=B,total_config_space=N)
    m.static_blocked=len(blk); m.static_suppressed=len(sup)
    t0=time.time(); ft=None; cl=[]; bl=[]; st=[]; flt=[]; ftypes={}; n=0
    for cfg in queue:
        if n>=B: break
        if abl=="full" and learner.should_suppress(cfg):
            m.autophagy_suppressed+=1; continue
        n+=1
        ok,met,out=run_dynamatic_single(config=cfg,src_file=src,top_func=top)
        ct=met.get('compile_time_s',0)
        if ok:
            m.successful+=1; st.append(ct)
            cl.append(met.get('num_components',0)); bl.append(met.get('num_buffers',0))
            if ft is None: ft=time.time()-t0
        else:
            m.failed+=1; flt.append(ct); m.wasted_time_on_failures_s+=ct
            et=met.get("error_type","unknown"); ftypes[et]=ftypes.get(et,0)+1
            if abl=="full": learner.add_failure(cfg, et)
        m.total_evaluated+=1
    _timing(m,t0,ft,st,flt)
    if cl: m.best_components=min(cl); m.mean_components=round(np.mean(cl),1)
    if bl: m.best_buffers=min(bl); m.mean_buffers=round(np.mean(bl),1)
    m.failure_breakdown=json.dumps(ftypes)
    return m

# ════════════════════════════════════════════════════════════
# Orchestrator
# ════════════════════════════════════════════════════════════

def run_multiseed_experiment(tool, benchmarks, budgets, num_seeds,
                              enable_pipeline, extended_space, results_root, ablation):
    all_raw = []
    seeds = list(range(num_seeds))
    bm_map = BAMBU_BM if tool=="bambu" else DYN_BM

    for bm in benchmarks:
        src, top = bm_map[bm]
        if tool=="bambu":
            cfgs = generate_bambu_configs_extended(enable_pipeline) if extended_space \
                   else generate_bambu_configs(enable_pipeline)
        else:
            cfgs = generate_dynamatic_configs()
        N = len(cfgs)

        for B in budgets:
            print(f"\n{'#'*70}\n# {bm} / {tool} / B={B} / space={N}\n{'#'*70}")

            # Grid (1 run)
            print(f"\n  >> Grid (B={B})")
            s = grid_sample(cfgs, B)
            wk = os.path.join(results_root, bm, f"B{B}", "Grid")
            r = (_run_bambu_list if tool=="bambu" else _run_dyn_list)(s,src,top,bm,"Grid",0,B,N,wk)
            all_raw.append(asdict(r))
            print(f"     SR={r.success_rate_pct}% time={r.wall_clock_s}s TTFF={r.time_to_first_feasible_s}s")

            # Random (N seeds)
            for sd in seeds:
                print(f"\n  >> Random seed={sd} (B={B})")
                s = rand_sample(cfgs, B, sd)
                wk = os.path.join(results_root, bm, f"B{B}", f"Rand_s{sd}")
                r = (_run_bambu_list if tool=="bambu" else _run_dyn_list)(s,src,top,bm,"Random",sd,B,N,wk)
                all_raw.append(asdict(r))
                print(f"     SR={r.success_rate_pct}% time={r.wall_clock_s}s")

            # LHS (N seeds)
            for sd in seeds:
                print(f"\n  >> LHS seed={sd} (B={B})")
                s = lhs_sample(cfgs, B, sd)
                wk = os.path.join(results_root, bm, f"B{B}", f"LHS_s{sd}")
                r = (_run_bambu_list if tool=="bambu" else _run_dyn_list)(s,src,top,bm,"LHS",sd,B,N,wk)
                all_raw.append(asdict(r))
                print(f"     SR={r.success_rate_pct}% time={r.wall_clock_s}s")

            # PA-DSE (1 run)
            print(f"\n  >> PA-DSE (B={B})")
            wk = os.path.join(results_root, bm, f"B{B}", "PADSE")
            r = (_run_padse_bambu if tool=="bambu" else _run_padse_dyn)(cfgs,src,top,bm,B,N,wk,ablation)
            all_raw.append(asdict(r))
            print(f"     SR={r.success_rate_pct}% time={r.wall_clock_s}s "
                  f"blk={r.static_blocked} sup={r.static_suppressed} auto={r.autophagy_suppressed}")

    # Save
    raw_path = os.path.join(results_root, "raw_results.csv")
    os.makedirs(results_root, exist_ok=True)
    if all_raw:
        with open(raw_path,"w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=all_raw[0].keys()); w.writeheader(); w.writerows(all_raw)
    print(f"\n{'='*70}\nSaved {raw_path} ({len(all_raw)} runs)")
    print(f"Next: python3 scripts/aggregate_results.py {raw_path} --latex")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tool",required=True,choices=["bambu","dynamatic"])
    p.add_argument("--benchmarks",nargs="+",required=True)
    p.add_argument("--budgets",nargs="+",type=int,default=[20])
    p.add_argument("--num-seeds",type=int,default=10)
    p.add_argument("--enable-pipeline",action="store_true")
    p.add_argument("--extended-space",action="store_true")
    p.add_argument("--results-root",type=str,default="results/multiseed")
    p.add_argument("--ablation",default="full",choices=["no_filter","static_only","full"])
    a = p.parse_args()
    run_multiseed_experiment(a.tool,a.benchmarks,a.budgets,a.num_seeds,
                             a.enable_pipeline,a.extended_space,a.results_root,a.ablation)

if __name__=="__main__":
    main()
