"""Diagnose all data after Dynamatic rerun + Bambu ablation extension.

Covers:
  1. Dynamatic main (7 methods × 6 bench × 10 seeds)
  2. Dynamatic PA-DSE perms (10 perms × 6 bench)
  3. Dynamatic GT
  4. Bambu ablation extended (new, overnight rerun)
  5. Role reversal check (Bambu vs Dynamatic)
"""

import pandas as pd
from pathlib import Path

ROOT = Path("/Users/zhangxinyu/Desktop/hls/results")

print("=" * 70)
print("DATA DIAGNOSIS")
print("=" * 70)


def check(path, label):
    p = Path(path)
    print(f"\n### {label} ###")
    if not p.exists():
        print(f"  ❌ MISSING: {p}")
        return None
    df = pd.read_csv(p)
    print(f"  File: {p.relative_to(ROOT)}")
    print(f"  Rows: {len(df)}")
    if "benchmark" in df.columns:
        print(f"  Benchmarks: {sorted(df['benchmark'].unique())}")
    if "strategy" in df.columns:
        print(f"  Strategies:")
        for s, n in df['strategy'].value_counts().items():
            print(f"    {s:<30s} {n}")
    return df


# ======== Load all data ========
dyn_main  = check(ROOT / "master/dynamatic_main/run_summary.csv",   "1. DYNAMATIC MAIN")
dyn_perms = check(ROOT / "rerun/dynamatic_pa_dse_perms/run_summary.csv", "2. DYNAMATIC PA-DSE PERMS")
dyn_gt    = check(ROOT / "master/dynamatic_ground_truth/run_summary.csv", "3. DYNAMATIC GT")
bam_abl   = check(ROOT / "rerun/bambu_ablation_extended/run_summary.csv", "4. BAMBU ABLATION EXTENDED")

# Bambu main data for comparison
bam_main  = check(ROOT / "master/bambu_main/run_summary.csv",       "5. BAMBU MAIN (existing)")
bam_perms = check(ROOT / "rerun/bambu_pa_dse_perms/run_summary.csv", "6. BAMBU PA-DSE PERMS (existing)")


# ======== Dynamatic main: SR analysis ========
if dyn_main is not None:
    print("\n" + "=" * 70)
    print("DYNAMATIC MAIN — Mean SR per strategy × benchmark")
    print("=" * 70)
    # Exclude the single PA-DSE_phago+Full runs in bambu_main style (keep for ref)
    pivot = dyn_main.pivot_table(
        index="strategy", columns="benchmark",
        values="sr_pct", aggfunc="mean"
    )
    print(pivot.round(2))

    print("\nDynamatic MAIN — Overall SR per strategy (across benchmarks):")
    summary = dyn_main.groupby("strategy")["sr_pct"].agg(
        ["count", "mean", "std"]
    ).round(2).sort_values("mean", ascending=False)
    print(summary)


# ======== Dynamatic PA-DSE perms ========
if dyn_perms is not None:
    print("\n" + "=" * 70)
    print("DYNAMATIC PA-DSE PERMS — SR stats per benchmark")
    print("=" * 70)
    per_bench = dyn_perms.groupby("benchmark")["sr_pct"].agg(
        ["count", "mean", "std", "min", "max"]
    ).round(2)
    print(per_bench)
    print(f"\n  Overall: mean = {dyn_perms['sr_pct'].mean():.2f}%, "
          f"std = {dyn_perms['sr_pct'].std():.2f}%, "
          f"n = {len(dyn_perms)}")


# ======== Bambu ablation extended (new) ========
if bam_abl is not None:
    print("\n" + "=" * 70)
    print("BAMBU ABLATION EXTENDED — config × benchmark counts")
    print("=" * 70)
    cnt = bam_abl.pivot_table(
        index="ablation_config", columns="benchmark",
        values="sr_pct", aggfunc="count", fill_value=0
    ).astype(int)
    print(cnt)

    print("\nBAMBU ABLATION EXTENDED — Mean SR per config × benchmark:")
    mean = bam_abl.pivot_table(
        index="ablation_config", columns="benchmark",
        values="sr_pct", aggfunc="mean"
    )
    print(mean.round(2))

    print("\nBAMBU ABLATION EXTENDED — Config overall stats:")
    summary = bam_abl.groupby("ablation_config")["sr_pct"].agg(
        ["count", "mean", "std"]
    ).round(2).sort_values("mean", ascending=False)
    print(summary)


# ======== ROLE REVERSAL: combine Bambu + Dynamatic PA-DSE SR ========
print("\n" + "=" * 70)
print("ROLE REVERSAL CHECK: PA-DSE SR on Bambu vs Dynamatic")
print("=" * 70)

if bam_perms is not None and dyn_perms is not None:
    bam_sr = bam_perms['sr_pct'].mean()
    bam_sr_std = bam_perms['sr_pct'].std()
    dyn_sr = dyn_perms['sr_pct'].mean()
    dyn_sr_std = dyn_perms['sr_pct'].std()
    print(f"  Bambu     PA-DSE SR: {bam_sr:.2f} ± {bam_sr_std:.2f}%  (n={len(bam_perms)})")
    print(f"  Dynamatic PA-DSE SR: {dyn_sr:.2f} ± {dyn_sr_std:.2f}%  (n={len(dyn_perms)})")


# ======== Expected role reversal: need ablation data on both tools ========
print("\n" + "=" * 70)
print("Suggested next steps based on data availability")
print("=" * 70)
print("  - Bambu main (baselines + PA-DSE): ✓ available")
print("  - Bambu ablation extended: ✓ available (new overnight rerun)")
print("  - Dynamatic main: ✓ available")
print("  - Dynamatic PA-DSE perms: ✓ available")
print("  - Dynamatic ablation: ❌ still missing (need to decide whether to rerun)")
print("  - Dynamatic budget sweep: ❌ still missing")
print("  - Bambu budget sweep: ⚠️ only vadd (need to extend)")
