"""Full diagnosis of Bambu data on local Mac.

Reports file locations, row counts, unique values, and date stamps
so we know exactly what data we have before writing paper figures.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path("/Users/zhangxinyu/Desktop/hls/results")

print("=" * 70)
print("BAMBU DATA DIAGNOSIS")
print("=" * 70)


def file_info(path):
    p = Path(path)
    if not p.exists():
        return "MISSING"
    size = p.stat().st_size
    mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return f"{size:>10,} B | {mtime}"


# ==================== 1. MAIN ====================
print("\n### 1. MAIN EXPERIMENT DATA ###")

main_dir = ROOT / "master" / "bambu_main"
print(f"\n[{main_dir}]")
for f in ["run_summary.csv", "eval_log.csv", "signature_log.csv"]:
    print(f"  {f:<20s} {file_info(main_dir / f)}")

perms_dir = ROOT / "rerun" / "bambu_pa_dse_perms"
print(f"\n[{perms_dir}]")
for f in ["run_summary.csv", "eval_log.csv"]:
    print(f"  {f:<20s} {file_info(perms_dir / f)}")


# ==================== 2. MAIN run_summary content ====================
print("\n### 2. bambu_main/run_summary.csv CONTENT ###")
m = pd.read_csv(main_dir / "run_summary.csv")
print(f"Total rows: {len(m)}")
print(f"\nStrategy value counts:")
print(m["strategy"].value_counts())
print(f"\nBenchmarks: {sorted(m['benchmark'].unique())}")
print(f"Budgets: {sorted(m['budget'].unique())}")
print(f"Seeds: {sorted(m['seed'].dropna().unique())}")
if "queue_permutation_id" in m.columns:
    print(f"Perms: {sorted(m['queue_permutation_id'].dropna().unique())}")

print(f"\nPer (strategy, benchmark) seed/perm count:")
grp = m.groupby(["strategy", "benchmark"]).size().unstack(fill_value=0)
print(grp)


# ==================== 3. PA-DSE perms content ====================
print("\n### 3. bambu_pa_dse_perms/run_summary.csv CONTENT ###")
p = pd.read_csv(perms_dir / "run_summary.csv")
print(f"Total rows: {len(p)}")
print(f"\nStrategy value counts:")
print(p["strategy"].value_counts())
print(f"Benchmarks: {sorted(p['benchmark'].unique())}")
print(f"Perms: {sorted(p['queue_permutation_id'].dropna().unique())}")

print(f"\nPer benchmark perm count:")
print(p.groupby("benchmark").size())

print(f"\nPer benchmark SR distribution:")
print(p.groupby("benchmark")["sr_pct"].agg(["mean", "std", "min", "max", "count"]).round(2))


# ==================== 4. Compare PA-DSE in main vs perms ====================
print("\n### 4. PA-DSE in bambu_main vs bambu_pa_dse_perms ###")
padse_in_main = m[m["strategy"].str.contains("PA-DSE", na=False)]
print(f"\nPA-DSE rows in bambu_main/: {len(padse_in_main)}")
if len(padse_in_main) > 0:
    print("First few rows:")
    print(padse_in_main[["strategy", "benchmark", "seed", "queue_permutation_id",
                         "sr_pct", "successful_evals", "wasted_calls"]].head(15).to_string())

print(f"\nPA-DSE rows in bambu_pa_dse_perms/: {len(p)}")
print("First few rows:")
print(p[["strategy", "benchmark", "queue_permutation_id",
         "sr_pct", "successful_evals", "wasted_calls"]].head(15).to_string())


# ==================== 5. Other support datasets ====================
print("\n### 5. ABLATION / BUDGET / SENSITIVITY ###")
for subdir in ["ablation", "budget_sweep", "sensitivity", "probe", "ground_truth"]:
    d = ROOT / "master" / subdir
    f = d / "run_summary.csv"
    print(f"\n[{d.name}]")
    print(f"  run_summary.csv: {file_info(f)}")
    if f.exists():
        try:
            df = pd.read_csv(f)
            print(f"  rows: {len(df)}")
            print(f"  benchmarks: {sorted(df['benchmark'].unique())}")
            if "ablation_config" in df.columns and df["ablation_config"].notna().any():
                print(f"  ablation_config: {df['ablation_config'].value_counts().to_dict()}")
            if "strategy" in df.columns:
                print(f"  strategies: {df['strategy'].value_counts().to_dict()}")
        except Exception as e:
            print(f"  READ ERROR: {e}")