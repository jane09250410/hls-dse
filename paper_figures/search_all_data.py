"""Search ALL run_summary*.csv and related files under results/ to find
any data we might have missed.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path("/Users/zhangxinyu/Desktop/hls/results")

print("=" * 80)
print("SEARCHING ALL CSV FILES UNDER", ROOT)
print("=" * 80)

# ==================== 1. All run_summary*.csv files ====================
print("\n### 1. ALL run_summary*.csv files ###")
for p in sorted(ROOT.rglob("run_summary*.csv")):
    mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    size = p.stat().st_size
    rel = p.relative_to(ROOT)
    try:
        df = pd.read_csv(p)
        nrows = len(df)
        strategies = df['strategy'].value_counts().to_dict() if 'strategy' in df.columns else {}
        benchmarks = sorted(df['benchmark'].unique()) if 'benchmark' in df.columns else []
        print(f"\n  📄 {rel}")
        print(f"     {size:>10,} B | {mtime} | {nrows} rows")
        print(f"     benchmarks ({len(benchmarks)}): {benchmarks}")
        print(f"     strategies: {strategies}")
    except Exception as e:
        print(f"  ❌ {rel}: {e}")

# ==================== 2. All directories with CSVs ====================
print("\n### 2. ALL CSV files under results/ ###")
all_csvs = sorted(ROOT.rglob("*.csv"))
print(f"Total CSV files: {len(all_csvs)}")
print("\nDirectory summary (count, total size):")
dir_stats = {}
for p in all_csvs:
    d = str(p.parent.relative_to(ROOT))
    if d not in dir_stats:
        dir_stats[d] = {"count": 0, "size": 0, "files": []}
    dir_stats[d]["count"] += 1
    dir_stats[d]["size"] += p.stat().st_size
    if "run_summary" in p.name or "summary" in p.name.lower():
        dir_stats[d]["files"].append(p.name)

for d, s in sorted(dir_stats.items()):
    print(f"  {d:<50s} {s['count']:>4d} files, {s['size']:>12,} B")
    for f in s["files"]:
        print(f"     -> {f}")

# ==================== 3. Look for other PA-DSE data ====================
print("\n### 3. Look for anything with 'pa_dse' or 'padse' or 'perm' in path/name ###")
patterns = ["*pa_dse*", "*padse*", "*perm*", "*PA-DSE*"]
found = set()
for pat in patterns:
    for p in ROOT.rglob(pat):
        if p.is_file():
            found.add(p)

for p in sorted(found):
    mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    size = p.stat().st_size
    rel = p.relative_to(ROOT)
    print(f"  {rel}")
    print(f"     {size:>10,} B | {mtime}")

# ==================== 4. Check experiments/ and b120_compare/ ====================
print("\n### 4. Other results subdirs ###")
for subdir in ["experiments", "b120_compare", "ground_truth"]:
    d = ROOT / subdir
    if d.exists():
        print(f"\n[{subdir}]")
        # Top-level files
        for p in sorted(d.iterdir())[:20]:
            if p.is_file():
                mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                print(f"  {p.name:<40s} {p.stat().st_size:>10,} B | {mtime}")
            elif p.is_dir():
                nfiles = sum(1 for _ in p.rglob("*") if _.is_file())
                print(f"  {p.name}/  ({nfiles} files inside)")

# ==================== 5. List ALL subdirs under master/ and rerun/ ====================
print("\n### 5. Subdirs under master/ and rerun/ ###")
for parent in ["master", "rerun"]:
    d = ROOT / parent
    if d.exists():
        print(f"\n[{parent}/]")
        for p in sorted(d.iterdir()):
            if p.is_dir():
                summary = p / "run_summary.csv"
                if summary.exists():
                    mtime = datetime.fromtimestamp(summary.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                    try:
                        df = pd.read_csv(summary)
                        print(f"  {p.name:<30s} run_summary.csv: {len(df):>4d} rows | {mtime}")
                    except:
                        print(f"  {p.name:<30s} run_summary.csv: UNREADABLE")
                else:
                    nfiles = sum(1 for _ in p.rglob("*.csv"))
                    print(f"  {p.name:<30s} ({nfiles} csv files, no run_summary.csv)")
