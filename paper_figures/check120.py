import pandas as pd
from pathlib import Path

df = pd.read_csv(Path("/Users/zhangxinyu/Desktop/hls/results/master/budget_sweep/run_summary.csv"))

print("=== Full/OFRS/RPE 在 gcd B=60 的详细字段 ===")
for cfg in ["phago+Full", "phago+OFRS", "phago+RPE"]:
    sub = df[(df["benchmark"] == "gcd") & (df["ablation_config"] == cfg) & (df["budget"] == 60)]
    print(f"\n--- {cfg} ---")
    print(sub[["sr_pct", "total_skipped", "signatures_learned", "probes_triggered", "queue_permutation_id"]].to_string())

print("\n=== Full 在 gcd B=120 的详细字段 ===")
sub = df[(df["benchmark"] == "gcd") & (df["ablation_config"] == "phago+Full") & (df["budget"] == 120)]
print(sub[["sr_pct", "total_evals", "successful_evals", "total_skipped", "signatures_learned", "probes_triggered", "queue_permutation_id"]].to_string())

print("\n=== Full 在 vadd B=60 的详细字段 (Bambu, DFRL 应该正常) ===")
sub = df[(df["benchmark"] == "vadd") & (df["ablation_config"] == "phago+Full") & (df["budget"] == 60)]
print(sub[["sr_pct", "total_skipped", "signatures_learned", "probes_triggered", "queue_permutation_id"]].to_string())