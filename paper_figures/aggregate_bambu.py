"""Aggregate all Bambu experiment results into clean summary CSVs.

Reads from ~/Desktop/hls/results/:
  master/bambu_main/        -> 7 baselines × 8 benches × 10 seeds
  rerun/bambu_pa_dse_perms/ -> PA-DSE × 8 benches × 10 perms
  master/ablation/          -> 8 configs × benches
  master/budget_sweep/      -> methods × budgets
  master/sensitivity/       -> hparam sweeps

Writes to:
  results/analysis/bambu_main_summary.csv
  results/analysis/bambu_ablation_summary.csv
  results/analysis/bambu_budget_summary.csv
  results/analysis/bambu_sensitivity_summary.csv

Run once. Figure scripts then consume these summaries.
"""

import pandas as pd
from pathlib import Path

# ================= Config =================
ROOT = Path("/Users/zhangxinyu/Desktop/hls/results")
OUT = ROOT / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

# Normalize strategy names: CSV 'strategy' column -> paper display name
STRATEGY_MAP = {
    "Random":                "Random",
    "Filtered_Random":       "Filtered_Random",
    "FilteredRandom":        "Filtered_Random",
    "SA":                    "SA",
    "SimulatedAnnealing":    "SA",
    "GA":                    "GA",
    "GeneticAlgorithm":      "GA",
    "GP-BO":                 "GP-BO",
    "GPBO":                  "GP-BO",
    "GP_BO":                 "GP-BO",
    "RF":                    "RF",
    "RF_Classifier":         "RF",         # <-- 新增
    "RFClassifier":          "RF",
    "RandomForest":          "RF",
    "PA-DSE":                "PA-DSE",
    "PADSE":                 "PA-DSE",
    "phago+Full":            "PA-DSE",
    "PA-DSE Full":           "PA-DSE",
    "PA-DSE_phago+Full":     "PA-DSE",     # <-- 新增
    "PA-DSE_L1":             "__DROP__",   # <-- 新增，后面过滤
}


def find_run_summaries(root_dir):
    root = Path(root_dir)
    if not root.exists():
        print(f"  [WARN] {root} missing")
        return []
    return sorted(root.rglob("run_summary*.csv"))


def load_all_runs(root_dir, label):
    """Concatenate all run_summary*.csv under root_dir."""
    paths = find_run_summaries(root_dir)
    if not paths:
        print(f"[{label}] no run_summary*.csv found under {root_dir}")
        return pd.DataFrame()

    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__source_path"] = str(p.relative_to(ROOT))
            dfs.append(df)
        except Exception as e:
            print(f"  [ERR] {p}: {e}")
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    print(f"[{label}] {len(paths)} files, {len(out)} rows")
    return out


def normalize_methods(df):
    if "strategy" in df.columns:
        df["method"] = df["strategy"].map(STRATEGY_MAP).fillna(df["strategy"])
    return df


# ================= 1. MAIN =================
def aggregate_main():
    print("\n=== MAIN ===")
    # Baselines (seeds 0-9)
    df_base = load_all_runs(ROOT / "master" / "bambu_main", "main_baseline")
    # PA-DSE perms
    df_pad = load_all_runs(ROOT / "rerun" / "bambu_pa_dse_perms", "main_padse")

    df = pd.concat([df_base, df_pad], ignore_index=True) if not df_pad.empty else df_base
    if df.empty:
        print("  [ERROR] no main data")
        return None

    df = normalize_methods(df)
    df = df[df["method"] != "__DROP__"].copy()  # <-- 新增，移除 L1
    # Unified run identifier (seed for baselines, perm for PA-DSE)
    if "queue_permutation_id" in df.columns:
        df["run_id"] = df["seed"].fillna(df["queue_permutation_id"]).astype("Int64")
    else:
        df["run_id"] = df["seed"].astype("Int64")

    keep = ["benchmark", "method", "run_id", "tool", "budget",
            "sr_pct", "successful_evals", "wasted_calls", "total_evals",
            "best_area", "best_latency", "uqor", "ttff_s", "total_wall_clock_s",
            "total_skipped", "false_skips_pending", "false_skips_verified",
            "signatures_learned", "probes_triggered", "probes_succeeded",
            "overhead_phago_ms", "overhead_rpe_ms", "overhead_ofrs_ms"]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    path = OUT / "bambu_main_summary.csv"
    out.to_csv(path, index=False)
    print(f"  → {path} ({len(out)} rows)")

    # Sanity: per method × bench counts and SR
    print("\n  Run counts (method × benchmark):")
    print(out.pivot_table(index="method", columns="benchmark",
                          values="sr_pct", aggfunc="count").fillna(0).astype(int))
    print("\n  Mean SR_pct by method (across all runs):")
    print(out.groupby("method")["sr_pct"].agg(["mean", "std", "count"]).round(2))
    return out


# ================= 2. ABLATION =================
def aggregate_ablation():
    print("\n=== ABLATION ===")
    df = load_all_runs(ROOT / "master" / "ablation", "ablation")
    if df.empty:
        return None
    df = normalize_methods(df)
    path = OUT / "bambu_ablation_summary.csv"
    df.to_csv(path, index=False)
    print(f"  → {path} ({len(df)} rows)")
    if "ablation_config" in df.columns:
        print("\n  Ablation configs:")
        print(df["ablation_config"].value_counts())
    return df


# ================= 3. BUDGET SWEEP =================
def aggregate_budget():
    print("\n=== BUDGET SWEEP ===")
    df = load_all_runs(ROOT / "master" / "budget_sweep", "budget")
    if df.empty:
        return None
    df = normalize_methods(df)
    path = OUT / "bambu_budget_summary.csv"
    df.to_csv(path, index=False)
    print(f"  → {path} ({len(df)} rows)")
    if "budget" in df.columns:
        print("\n  Budget values sampled:")
        print(sorted(df["budget"].unique()))
    return df


# ================= 4. SENSITIVITY =================
def aggregate_sensitivity():
    print("\n=== SENSITIVITY ===")
    df = load_all_runs(ROOT / "master" / "sensitivity", "sensitivity")
    if df.empty:
        return None
    df = normalize_methods(df)
    path = OUT / "bambu_sensitivity_summary.csv"
    df.to_csv(path, index=False)
    print(f"  → {path} ({len(df)} rows)")
    for col in ["tau", "theta", "n_min", "p_probe"]:
        if col in df.columns:
            vals = sorted(df[col].dropna().unique())
            print(f"  {col} sampled: {vals}")
    return df


# ================= RUN =================
if __name__ == "__main__":
    print(f"Output → {OUT}\n")
    df_main = aggregate_main()
    df_abl = aggregate_ablation()
    df_bud = aggregate_budget()
    df_sens = aggregate_sensitivity()

    print("\n=== DONE ===")
    for name, df in [("main", df_main), ("ablation", df_abl),
                     ("budget", df_bud), ("sensitivity", df_sens)]:
        n = len(df) if df is not None else 0
        print(f"  {name:14s} → {n:6d} rows")
