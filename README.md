# PA-DSE: Feasibility-Aware Design-Space Exploration for HLS

Implementation and reproduction package for the paper **"PA-DSE: Feasibility-Aware
Design Space Exploration for High-Level Synthesis via Hierarchical
Evidence-Bounded Pruning"**.

## What's in this repo

| Path | Contents |
|---|---|
| `pa_dse_paper.tex` / `references.bib` | Paper source |
| `paper_figures/` | Figure-generation scripts and aggregated tables |
| `scripts/` | DSE algorithms, baselines, runners, logging |
| `benchmarks/` | Bambu C kernels (8 benchmarks) |
| `master_experiments.py` | Main experiment driver (vanilla PA-DSE + baselines) |
| `rerun_tonight.py` | Bambu PA-DSE 10-permutation completion |
| `rerun_dynamatic_full.py` | Dynamatic full rerun after Gurobi fix |
| `rerun_cc.py` | **NEW**: PA-DSE-CC (Categorical Coverage extension) runs |
| `offline_sim/` | Optional ground-truth-based offline simulator |

Dynamatic benchmarks (`gcd`, `matching`, `binary_search`, `kernel_2mm`,
`fir`, `histogram`) are read from `~/dynamatic/integration-test/<name>/<name>.c`.
You must have the Dynamatic toolchain installed (paper §IV-A).

## Algorithm

PA-DSE has two layers organized around the *evidence hierarchy* principle —
action strength must not exceed evidence strength.

**Layer 1 — SCF (Static Constraint Filter).** Tool-documented incompatibility
rules permanently remove configurations before exploration begins.

**Layer 2 — DFRL (Dynamic Failure Risk Learning).**
  - **RPE (Recurrent Pattern Extractor).** After repeated same-type failures,
    extracts a typed signature; matching configs are hard-skipped.
  - **OFRS (Online Failure Risk Scorer).** Per-dimension risk model that
    reorders the queue, never skips.
  - **OFRS-CC (Categorical-Coverage extension).** A coverage bonus subtracted
    from OFRS's risk score that prioritises (dim, value) pairs OFRS has
    observed fewer than `n_cov` times. Addresses a deficiency observed
    when SCF is unavailable: a single early failure can cause OFRS to abandon
    an entire categorical sub-space. Controlled by `β_cov` (default 0.2);
    `β_cov = 0` reproduces the original PA-DSE.

The 8 ablation configurations are: `no-filter`, `SCF-only`, `SCF+RPE`,
`SCF+OFRS`, `SCF+DFRL` (recommended), `DFRL-only`, `SCF+RPE-reorder`,
`SCF+OFRS-skip` (hierarchy stress test).

Implementation files:
- `scripts/dynamic_failure_learner.py` — RPE + OFRS (with CC extension)
- `scripts/methods/pa_dse_method.py` — PADSEMethod (8 ablation configs + CC switch)
- `scripts/feasibility_filter.py` — SCF (Layer 1)

## Reproducing the paper

### Vanilla PA-DSE (paper Tables II / III / IV)

```bash
# Multi-day unattended runner with resume capability
nohup python3 master_experiments.py > master.log 2>&1 &
tail -f master.log

# Then add 10 PA-DSE permutations on Bambu
nohup python3 rerun_tonight.py > rerun_tonight.log 2>&1 &

# And on Dynamatic
nohup python3 rerun_dynamatic_full.py > rerun_dyn.log 2>&1 &
```

### PA-DSE-CC (categorical coverage extension)

```bash
# Run all CC stages (main + ablation) — 8-12 hours
nohup python3 rerun_cc.py > rerun_cc.log 2>&1 &

# Or one stage at a time
python3 rerun_cc.py --stage bambu_main
python3 rerun_cc.py --stage dynamatic_main
python3 rerun_cc.py --stage ablation
```

CC outputs go to `results/cc/`, kept separate from `results/master/`
(vanilla data) so the comparison is straightforward.

### Aggregating tables and figures

```bash
python3 paper_figures/compute_paper_tables.py
python3 paper_figures/fig1_main_results_v2.py
python3 paper_figures/fig_dynamatic_main_v2.py
# …etc
```

## Hyperparameters (frozen, no per-benchmark tuning)

| Parameter | Value | Notes |
|---|---|---|
| τ (RPE min failure support) | 2 | |
| θ (RPE confidence threshold) | 0.8 | |
| n_min (OFRS cold-start) | 5 | |
| p_probe (probe rate) | 0.05 | |
| β_cov (CC weight) | 0.2 | β_cov=0 → vanilla |
| n_cov (CC observation budget) | 2 | |

Sensitivity analyses for τ, θ, β_cov are reported in the paper §V.

## Hardware / Software

- Azure Standard D4s_v5 (4 vCPUs, 15 GB RAM, Ubuntu 24.04)
- Python 3.12, pandas, numpy, scikit-learn, GPy
- PandA-Bambu 0.9.8
- Dynamatic 2.0 with Gurobi 12.0 (academic license)

## Citation

```bibtex
@article{padse2026,
  title  = {PA-DSE: Feasibility-Aware Design Space Exploration for
            High-Level Synthesis via Hierarchical Evidence-Bounded Pruning},
  author = {Anonymous},
  year   = {2026},
}
```
