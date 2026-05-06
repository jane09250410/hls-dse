# PA-DSE: Feasibility-Aware Design-Space Exploration for HLS

Implementation and reproduction package for the paper **"PA-DSE: Feasibility-Aware
Design Space Exploration for High-Level Synthesis via Hierarchical
Evidence-Bounded Pruning"**.

## What's in this repo

| Path | Contents |
|---|---|
| `pa_dse_paper.tex` | LaTeX source of the paper |
| `references.bib` | Bibliography |
| `paper_figures/` | Figure-generation scripts and aggregated tables |
| `scripts/` | DSE algorithms, baselines, runners, logging |
| `benchmarks/` | Bambu C kernels (8 benchmarks) |
| `run_all.sh` | One-shot reproduction of every table and figure |

Dynamatic benchmarks (`gcd`, `matching`, `binary_search`, `kernel_2mm`) are read from
`~/dynamatic/integration-test/<name>/<name>.c` — you must have the Dynamatic toolchain
installed and the path set up (see paper §IV-A).

## Algorithm

PA-DSE has two layers organized around the *evidence hierarchy* principle —
action strength must not exceed evidence strength.

**Layer 1 — SCF (Static Constraint Filter).** Tool-documented incompatibility rules
permanently remove configurations before exploration begins.

**Layer 2 — DFRL (Dynamic Failure Risk Learning).** Online learning during a
single run.
  - **RPE (Recurrent Pattern Extractor).** After repeated same-type failures,
    extracts a typed signature; matching configs are hard-skipped (with a probe
    rate keeping signatures honest).
  - **OFRS (Online Failure Risk Scorer).** Per-dimension risk model that reorders
    the queue, never skips.
  - **OFRS-CC (Categorical-Coverage extension).** A coverage bonus subtracted
    from OFRS's risk score that prioritises (dim, value) pairs OFRS has
    observed fewer than `n_cov` times. This addresses a deficiency observed
    when SCF is unavailable: a single early failure can cause OFRS to abandon
    an entire categorical sub-space. CC is controlled by `β_cov` (default 0.2);
    `β_cov = 0` reproduces the original PA-DSE.

The full algorithm is `scripts/methods/pa_dse_method.py`; OFRS lives in
`scripts/dynamic_failure_learner.py`.

## Reproducing the paper

```bash
# One command, runs everything (~8–12 h on Azure D4s_v5)
bash run_all.sh
```

Or step-by-step:

```bash
# Main results (Tables II / III)
python3 scripts/runners/run_main_results.py --tool both --variant cc

# Component ablation (Table IV)
python3 scripts/runners/run_ablation.py --tool both --beta-cov 0.2

# Aggregate raw run_summary.csv → paper tables
python3 paper_figures/compute_paper_tables.py

# Regenerate figures
python3 paper_figures/fig1_main_results_v2.py
python3 paper_figures/fig_dynamatic_main_v2.py
# …etc, one script per figure
```

### Variants

`--variant` selects which PA-DSE configuration to run:

| Flag | Meaning |
|---|---|
| `vanilla` | β_cov = 0 (original PA-DSE) |
| `cc` | β_cov = 0.2, n_cov = 2 (PA-DSE-CC; paper default) |
| `both` | run both back-to-back |

To sweep `β_cov`, override the default:
```bash
python3 scripts/runners/run_main_results.py --tool dynamatic \
        --variant cc --beta-cov 0.3 --n-cov 2
```

## Hyperparameters

The paper freezes one set of hyperparameters across all (tool, benchmark) cells:

| Parameter | Value | Where set |
|---|---|---|
| τ (RPE min failure support) | 2 | `PADSEMethod(tau=2)` |
| θ (RPE confidence threshold) | 0.8 | `PADSEMethod(theta=0.8)` |
| n_min (OFRS cold-start) | 5 | `PADSEMethod(n_min=5)` |
| p_probe (probe rate) | 0.05 | `PADSEMethod(p_probe=0.05)` |
| β_cov (CC weight) | 0.2 | `PADSEMethod(beta_cov=0.2)` |
| n_cov (CC observation budget) | 2 | `PADSEMethod(n_cov=2)` |

Defaults are not tuned per-benchmark or per-tool. Sensitivity analyses for τ,
θ, β_cov are reported in §V of the paper.

## Hardware / Software

- Azure Standard D4s_v5 (4 vCPUs, 15 GB RAM, Ubuntu 24.04)
- Python 3.12, pandas, numpy, scikit-learn (for the RF baseline), GPy (for GP-BO)
- PandA-Bambu 0.9.8
- Dynamatic 2.0 with Gurobi 12.0 (academic license)

## Layout details

```
scripts/
├── dynamic_failure_learner.py  — RPE + OFRS (with CC extension)
├── pattern_learner.py          — error-type extraction
├── feasibility_filter.py       — SCF (Layer 1)
├── config_generator.py         — Bambu config space (420 points)
├── dynamatic_config_generator.py — Dynamatic config space (192 points)
├── methods/
│   ├── base.py                 — DSEMethod base interface
│   ├── pa_dse_method.py        — PA-DSE (8 ablation configs + CC switch)
│   ├── baseline_methods.py     — Random, Filtered, Grid, LHS
│   └── advanced_baselines.py   — SA, GA, GP-BO, RF
├── runners/
│   ├── run_single.py           — atomic (method, benchmark, budget) loop
│   ├── run_main_results.py     — drives Tables II / III
│   ├── run_ablation.py         — drives Table IV
│   └── run_experiments.py      — utility wrapper
└── exp_logging/
    └── experiment_logger.py    — CSV log writers
```

## Citation

```bibtex
@article{padse2026,
  title  = {PA-DSE: Feasibility-Aware Design Space Exploration for
            High-Level Synthesis via Hierarchical Evidence-Bounded Pruning},
  author = {Anonymous},
  year   = {2026},
}
```
