#!/usr/bin/env python3
"""
aggregate_results.py
====================
Reads raw_results.csv and produces:
  1. Aggregated CSV with mean ± std across seeds
  2. Paper-ready LaTeX tables (Bambu Table IV / Dynamatic Table V style)
  3. Time-metric table (wall clock, TTFF, wasted time)
  4. Budget sweep summary

Usage:
  python3 scripts/aggregate_results.py results/multiseed/raw_results.csv
  python3 scripts/aggregate_results.py results/multiseed/raw_results.csv --latex
"""

from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd


def fmt(vals, f=".1f"):
    if len(vals)==0: return "—"
    m = np.mean(vals)
    if len(vals)==1: return f"{m:{f}}"
    s = np.std(vals, ddof=1)
    return f"{m:{f}}$\\pm${s:{f}}" if s > 0.05 else f"{m:{f}}"

def fmt_plain(vals, f=".1f"):
    if len(vals)==0: return "—"
    m = np.mean(vals)
    if len(vals)==1: return f"{m:{f}}"
    s = np.std(vals, ddof=1)
    return f"{m:{f}} ± {s:{f}}" if s > 0.05 else f"{m:{f}}"


def aggregate(df):
    rows = []
    for (bm, strat, B), g in df.groupby(["benchmark","strategy","budget"]):
        r = {"benchmark":bm, "strategy":strat, "budget":int(B), "n_seeds":len(g)}

        for col, key in [
            ("success_rate_pct","SR"), ("wall_clock_s","time"),
            ("time_to_first_feasible_s","TTFF"),
            ("wasted_time_on_failures_s","wasted"),
            ("mean_eval_time_s","mean_eval"),
            ("mean_success_time_s","mean_succ_t"),
            ("mean_failure_time_s","mean_fail_t"),
            ("budget_utilization_pct","util"),
            ("unique_qor_points","UQoR"),
            ("pareto_points","pareto"),
            ("hypervolume","HV"),
        ]:
            v = g[col].dropna().values
            r[f"{key}_mean"] = round(np.mean(v),1) if len(v)>0 else None
            r[f"{key}_std"] = round(np.std(v,ddof=1),1) if len(v)>1 else 0.0
            r[f"{key}_str"] = fmt_plain(v)
            r[f"{key}_tex"] = fmt(v)

        for col, key in [
            ("best_area","bA"), ("mean_area","mA"),
            ("best_latency","bL"), ("mean_latency","mL"),
            ("best_components","bC"), ("mean_components","mC"),
            ("best_buffers","bB"), ("mean_buffers","mB"),
        ]:
            v = g[col].dropna().values
            r[f"{key}_str"] = fmt_plain(v)
            r[f"{key}_tex"] = fmt(v)

        # Filtering stats (only for PA-DSE)
        for col in ["static_blocked","static_suppressed","autophagy_suppressed"]:
            v = g[col].dropna().values
            r[col] = int(np.mean(v)) if len(v)>0 else 0

        rows.append(r)
    return pd.DataFrame(rows)


def print_summary(agg):
    for B in sorted(agg["budget"].unique()):
        sub = agg[agg["budget"]==B]
        print(f"\n{'='*100}")
        print(f"  BUDGET = {B}")
        print(f"{'='*100}")
        for bm in sub["benchmark"].unique():
            bsub = sub[sub["benchmark"]==bm]
            print(f"\n  {bm}:")
            print(f"  {'Strategy':>10s} {'seeds':>5s} {'SR%':>18s} {'Time(s)':>18s} "
                  f"{'TTFF(s)':>18s} {'Wasted(s)':>18s} {'Util%':>12s} {'UQoR':>10s} {'HV':>12s}")
            print(f"  {'-'*10} {'-'*5} {'-'*18} {'-'*18} {'-'*18} {'-'*18} {'-'*12} {'-'*10} {'-'*12}")
            for _,r in bsub.iterrows():
                print(f"  {r['strategy']:>10s} {r['n_seeds']:>5d} {r['SR_str']:>18s} "
                      f"{r['time_str']:>18s} {r['TTFF_str']:>18s} {r['wasted_str']:>18s} "
                      f"{r['util_str']:>12s} {r['UQoR_str']:>10s} {r['HV_str']:>12s}")


def latex_bambu_table(agg):
    """Generate Bambu results table (like paper Table IV but with mean±std and time)."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{PandA-Bambu Results with Multi-Seed Statistics}",
        r"\label{tab:bambu_multiseed}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll r rrrr rr}",
        r"\toprule",
        r"Bench. & Strategy & $B$ & SR\% & Best A. & Best L. & UQoR & Time (s) & TTFF (s) \\",
        r"\midrule",
    ]
    strats = ["Grid","Random","LHS","PA-DSE"]
    bms = agg["benchmark"].unique()
    for bm in bms:
        for B in sorted(agg[agg["benchmark"]==bm]["budget"].unique()):
            for i,s in enumerate(strats):
                row = agg[(agg["benchmark"]==bm)&(agg["strategy"]==s)&(agg["budget"]==B)]
                if row.empty: continue
                r = row.iloc[0]
                bl = bm if i==0 else ""
                Bl = str(B) if i==0 else ""
                lines.append(f"  {bl} & {s} & {Bl} & {r['SR_tex']} & {r['bA_tex']} & "
                             f"{r['bL_tex']} & {r['UQoR_tex']} & {r['time_tex']} & {r['TTFF_tex']} \\\\")
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def latex_dynamatic_table(agg):
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Dynamatic Results with Multi-Seed Statistics}",
        r"\label{tab:dyn_multiseed}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll r rrr rr}",
        r"\toprule",
        r"Bench. & Strategy & $B$ & SR\% & Best Comp. & Mean Comp. & Time (s) & TTFF (s) \\",
        r"\midrule",
    ]
    strats = ["Grid","Random","LHS","PA-DSE"]
    for bm in agg["benchmark"].unique():
        for B in sorted(agg[agg["benchmark"]==bm]["budget"].unique()):
            for i,s in enumerate(strats):
                row = agg[(agg["benchmark"]==bm)&(agg["strategy"]==s)&(agg["budget"]==B)]
                if row.empty: continue
                r = row.iloc[0]
                bl = bm if i==0 else ""
                Bl = str(B) if i==0 else ""
                lines.append(f"  {bl} & {s} & {Bl} & {r['SR_tex']} & {r['bC_tex']} & "
                             f"{r['mC_tex']} & {r['time_tex']} & {r['TTFF_tex']} \\\\")
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def latex_time_table(agg):
    """Dedicated time-metrics table for both tools."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Time Analysis: Wall Clock, TTFF, and Wasted Time}",
        r"\label{tab:time_analysis}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{ll r rrrr r}",
        r"\toprule",
        r"Bench. & Strategy & $B$ & Wall Clock (s) & TTFF (s) & Wasted (s) & Avg Fail (s) & Util\% \\",
        r"\midrule",
    ]
    strats = ["Grid","Random","LHS","PA-DSE"]
    for bm in agg["benchmark"].unique():
        for B in sorted(agg[agg["benchmark"]==bm]["budget"].unique()):
            for i,s in enumerate(strats):
                row = agg[(agg["benchmark"]==bm)&(agg["strategy"]==s)&(agg["budget"]==B)]
                if row.empty: continue
                r = row.iloc[0]
                bl = bm if i==0 else ""
                Bl = str(B) if i==0 else ""
                lines.append(f"  {bl} & {s} & {Bl} & {r['time_tex']} & {r['TTFF_tex']} & "
                             f"{r['wasted_tex']} & {r['mean_fail_t_tex']} & {r['util_tex']} \\\\")
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    p.add_argument("--latex", action="store_true")
    p.add_argument("--output-dir", default=None)
    a = p.parse_args()

    df = pd.read_csv(a.csv_path)
    out = a.output_dir or os.path.dirname(a.csv_path)
    os.makedirs(out, exist_ok=True)
    tool = df["tool"].iloc[0] if "tool" in df.columns else "bambu"

    agg = aggregate(df)
    agg.to_csv(os.path.join(out,"aggregated_results.csv"), index=False)
    print(f"Saved aggregated_results.csv")

    print_summary(agg)

    if a.latex:
        if tool == "bambu":
            tex = latex_bambu_table(agg)
        else:
            tex = latex_dynamatic_table(agg)
        with open(os.path.join(out,f"{tool}_table.tex"),"w") as f: f.write(tex)
        print(f"\nSaved {tool}_table.tex")

        tex2 = latex_time_table(agg)
        with open(os.path.join(out,"time_table.tex"),"w") as f: f.write(tex2)
        print(f"Saved time_table.tex")

        print("\n" + tex + "\n\n" + tex2)


if __name__=="__main__":
    main()
