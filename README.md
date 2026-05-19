# PA-DSE: Feasibility-Aware Design Space Exploration for High-Level Synthesis

Implementation and experimental data for **PA-DSE**, a feasibility-aware HLS DSE policy built on the principle that *action strength must not exceed evidence strength*.

📄 **Paper**: [`pa_dse_paper.pdf`](pa_dse_paper.pdf) — *PA-DSE: Feasibility-Aware Design Space Exploration for High-Level Synthesis via Hierarchical Evidence-Bounded Pruning*

## Overview

PA-DSE has two layers:

- **SCF (Static Constraint Filter)** — permanently removes configurations matching tool-documented incompatibility rules.
- **DFRL (Dynamic Failure Risk Learner)** — accumulates evidence online during a single run, with two sub-components:
  - **RPE (Recurrent Pattern Extractor)** — hard-skips configurations matching learned failure signatures.
  - **OFRS (Online Failure Risk Scorer)** — reorders the queue by smoothed per-dimension risk score.

## Headline Results

Evaluated on 8 shared benchmarks (matmul, vadd, fir, histogram, atax, bicg, gemm, gesummv) on two HLS tools:

| Tool | Budget | PA-DSE SR | Best Baseline | Wasted Reduction |
|---|---|---|---|---|
| Bambu (v0.9.8) | 60 | **92.5%** | RF 85.9% | 1.9× fewer |
| Dynamatic (v2.0) | 30 | **91.3%** | GP-BO 90.2% | 2× faster TTFF |
| Bambu | 120 | **83.0%** | SCF+OFRS 46.7% | RPE adds +36.3pp |

Algorithmic overhead: 5.3 ms/iter on Bambu, 2.0 ms/iter on Dynamatic (< 0.25% of synthesis cost).

## Repository Structure

```
hls-dse/
├── pa_dse_paper.tex      # Paper source
├── pa_dse_paper.pdf      # Compiled paper
├── references.bib        # Bibliography
├── benchmarks/           # C source for 8 benchmarks
├── scripts/              # PA-DSE implementation
│   ├── config_generator.py
│   ├── dynamatic_config_generator.py
│   ├── feasibility_filter.py       # SCF
│   ├── pattern_learner.py          # RPE
│   ├── dynamic_failure_learner.py  # OFRS
│   ├── methods/                    # PA-DSE + baselines
│   └── runners/
├── offline_sim/          # Offline simulator (ground-truth oracle)
├── paper_figures/        # Plotting scripts + generated PDFs
└── results/              # Aggregated experimental data
    ├── master/bambu_main/                # Bambu B=60 main comparison
    ├── master/dynamatic_main/            # Dynamatic B=30 main comparison
    ├── b30/ablation_bambu/               # 8-way ablation (Bambu)
    ├── b30/ablation_dynamatic/           # 8-way ablation (Dynamatic)
    ├── b120/bambu_main/                  # B=120 experiment (RPE active)
    ├── bambu_ground_truth/               # 420 configs × 8 benchmarks
    ├── dynamatic_ground_truth/           # 192 configs × 8 benchmarks
    ├── theta_sweep_b120/                 # θ sensitivity sweep
    └── rerun/                            # PA-DSE permutation runs (overhead)
```

## Reproducibility

This artifact supports reproducibility at three tiers, ordered from strongest to weakest.

### Tier 1: Paper tables from checked-in CSVs (authoritative)

```bash
python3 paper_figures/compute_paper_tables.py
```

This regenerates every numerical value reported in Tables I-VIII (main success-rate comparisons, ablation breakdowns, B=120 RPE analysis, theta sensitivity sweep, per-iteration overhead decomposition). All numbers in this path match the paper byte-for-byte.

### Tier 2: PA-DSE algorithm from source via the offline simulator

The `offline_sim/simulator.py` module replays any DSE method against the released ground-truth tables (`results/{bambu,dynamatic}_ground_truth/run_summary.csv`) without invoking the real HLS toolchain. We have verified that this path reproduces PA-DSE main-table numbers within statistical noise:

- Bambu B=60, n=40: PA-DSE SR = 92.33 ± 1.35 (paper: 92.5 ± 1.1, Δ = -0.17 pp)
- Dynamatic B=30, n=50: PA-DSE SR = 91.33 ± 3.93 (paper: 91.33 ± 3.93, exact match)

This validates that the released PA-DSE source code implements the algorithm described in the paper. Random and Filtered Random baselines also reproduce closely via this path (Δ ≤ 1 pp on Bambu).

### Tier 3: ML/stochastic baselines via the offline simulator

The baseline results reported in the paper (`results/master/bambu_main/run_summary.csv`) were collected on the experiment VM with the real Bambu/Dynamatic toolchain (≈ 150 s per run). Re-running SA, GA, GP-BO, and RF against the offline simulator with the source code in this artifact does **not** reproduce the absolute SR numbers in the paper; in our replays we observed:

| Method | Released CSV | Replay (sklearn 1.8 / Python 3.12) | Δ |
|--------|-------------:|----------------------------------:|---:|
| Random         | 15.0 % | 14.0 % | -1.0 |
| Filtered Random | 18.5 % | 18.3 % | -0.2 |
| SA             | 18.4 % | 12.6 % | -5.8 |
| GA             | 50.6 % | 28.0 % | -22.6 |
| GP-BO          | 71.3 % | 65.5 % | -5.8 |
| RF             | 85.9 % | 71.0 % | -14.9 |
| PA-DSE         | 92.5 % | 92.3 % | -0.2 |

The released CSVs are the authoritative source for all paper-reported baseline numbers. The offline simulator is suitable for validating PA-DSE end-to-end and for sanity-checking baseline ordering (PA-DSE > RF > GP-BO > GA > Filtered Random > SA > Random holds in both columns), but absolute baseline SR figures should be read from the CSVs rather than recomputed via the simulator. Environment drift (NumPy / SciPy / scikit-learn versions and Python random-stream behaviour) is the most likely source of the gap; we do not currently pin a reproducibility environment.

### Regenerate figures

**Note**: `fig_convergence.py` and `fig_qor.py` require per-evaluation logs
(`eval_log.csv`, ~100MB each). These traces were produced on the experiment
VM and are not included in this artifact due to size. The pre-generated
convergence and QoR figures are included under `paper_figures/out/`.
All other figures only need the aggregated `run_summary.csv` files, which ARE
in the repository.


```bash
cd paper_figures
python3 fig1_main_results_v2.py
python3 fig_dynamatic_main_v2.py
python3 fig_cost.py
python3 fig_perbench_heatmap.py
python3 fig_ablation_bar.py
python3 fig_overhead_v3.py
python3 fig_pareto.py
python3 fig_b120_bambu.py

# The two scripts below require eval_log.csv (~100MB each), which are NOT
# included in the repository. Pre-generated PDF/PNG outputs are in
# paper_figures/out/. Uncomment only if you have regenerated the eval logs.
# python3 fig_convergence.py
# python3 fig_qor.py
```

### Note on experiment launchers

The included `run_summary.csv` files are the authoritative data source for all paper tables.
The aggregation script `paper_figures/compute_paper_tables.py` regenerates the reported table values from these CSV files.

`scripts/runners/run_main_results.py` is a legacy stub kept only to preserve git history.
It does not reproduce the main tables. The complete main-experiment results were produced by batch jobs on the experiment VM and are released here as `results/master/*/run_summary.csv` and `results/rerun/*/run_summary.csv`.

### Rerun experiments

Requires Bambu 0.9.8, Dynamatic 2.0 (Gurobi 12.0), Python 3.12, sklearn, scipy, pandas, matplotlib.

```bash
# Collect Bambu ground truth (~2 hours)
python3 collect_bambu_gt.py

# B=120 main comparison (offline, ~10 minutes after GT is ready)
python3 run_b120_bambu.py

# Dynamatic ablation (5 permutations per benchmark; n=25 rows/config after filtering to 5 non-trivial benchmarks)
python3 rerun_ablation_n5.py
```

## Citation

```bibtex
@article{padse2026,
  title={PA-DSE: Feasibility-Aware Design Space Exploration for High-Level Synthesis
         via Hierarchical Evidence-Bounded Pruning},
  author={Zhang, Xinyu and Pilato, Christian},
  year={2026},
  note={Under submission}
}
```

## Tools

- **PandA-Bambu** v0.9.8 — Static HLS, Politecnico di Milano
- **Dynamatic** v2.0 — Dynamic dataflow HLS with MILP buffer placement, EPFL (Gurobi 12.0)

## Author

Xinyu Zhang — Politecnico di Milano, DEIB
Supervisor: Prof. Christian Pilato
