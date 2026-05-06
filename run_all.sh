#!/usr/bin/env bash
# run_all.sh — Reproduce every table and figure in the paper.
#
# Run order (each step writes to its own subdirectory under results/):
#   1. main_bambu          — Random/FilteredRandom/SA/GA/GP-BO/RF/PA-DSE-CC × 8 benches
#   2. main_dynamatic      — same baselines × 4 benches
#   3. ablation_bambu      — 8 ablation configs × 8 benches
#   4. ablation_dynamatic  — 8 ablation configs × 4 benches
#   5. ground_truth        — full grid sweep (offline reference, optional)
#   6. compute_paper_tables — aggregate run_summary.csv into table_*.csv
#   7. paper figures       — regenerate every figure in paper_figures/out/
#
# Hardware target: Azure Standard_D4s_v5 (4 vCPU, 15 GB RAM, Ubuntu 24.04).
# Wall-clock estimate: ≈ 8–12 hours total.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

LOG_DIR="${REPO_ROOT}/results/run_all_logs"
mkdir -p "$LOG_DIR"

run_step () {
    local name="$1"; shift
    local logfile="${LOG_DIR}/${name}.log"
    echo
    echo "=========================================================="
    echo "STEP: $name"
    echo "  log: $logfile"
    echo "=========================================================="
    if "$@" 2>&1 | tee "$logfile"; then
        echo "  [ok] $name"
    else
        echo "  [FAIL] $name — see $logfile"
        exit 1
    fi
}

# ────────────────────────────────────────────────────────────────
# 1. Main results — Bambu (Table II)
# ────────────────────────────────────────────────────────────────
run_step "main_bambu" \
    python3 scripts/runners/run_main_results.py \
        --tool bambu --variant cc \
        --output-dir results/main_bambu

# ────────────────────────────────────────────────────────────────
# 2. Main results — Dynamatic (Table III)
# ────────────────────────────────────────────────────────────────
run_step "main_dynamatic" \
    python3 scripts/runners/run_main_results.py \
        --tool dynamatic --variant cc \
        --output-dir results/main_dynamatic

# ────────────────────────────────────────────────────────────────
# 3. Ablation — Bambu (Table IV, top half)
# ────────────────────────────────────────────────────────────────
run_step "ablation_bambu" \
    python3 scripts/runners/run_ablation.py \
        --tool bambu --beta-cov 0.2 \
        --output-dir results/ablation_bambu

# ────────────────────────────────────────────────────────────────
# 4. Ablation — Dynamatic (Table IV, bottom half)
# ────────────────────────────────────────────────────────────────
run_step "ablation_dynamatic" \
    python3 scripts/runners/run_ablation.py \
        --tool dynamatic --beta-cov 0.2 \
        --output-dir results/ablation_dynamatic

# ────────────────────────────────────────────────────────────────
# 5. Ground truth / Grid offline reference (optional)
# ────────────────────────────────────────────────────────────────
run_step "ground_truth" \
    python3 scripts/runners/run_main_results.py \
        --tool both --variant cc --include-appendix \
        --no-advanced-baselines \
        --output-dir results/ground_truth

# ────────────────────────────────────────────────────────────────
# 6. Compute aggregate tables
# ────────────────────────────────────────────────────────────────
run_step "compute_tables" \
    python3 paper_figures/compute_paper_tables.py

# ────────────────────────────────────────────────────────────────
# 7. Regenerate all figures
# ────────────────────────────────────────────────────────────────
mkdir -p paper_figures/out
for fig in \
    fig1_main_results_v2.py \
    fig_dynamatic_main_v2.py \
    fig_ablation_b60.py \
    fig_ablation_bar.py \
    fig_perbench_heatmap.py \
    fig_convergence.py \
    fig_qor.py \
    fig_cost.py \
    fig_overhead_v3.py \
    fig_sensitivity.py \
    fig_budget_sweep.py \
    fig_probe.py
do
    run_step "fig_${fig%.py}" python3 "paper_figures/${fig}"
done

echo
echo "=========================================================="
echo "ALL DONE."
echo "Tables: paper_figures/out/table_*.csv"
echo "Figures: paper_figures/out/*.pdf"
echo "Raw runs: results/main_bambu, results/main_dynamatic, …"
echo "=========================================================="
