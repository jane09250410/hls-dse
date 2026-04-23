# PA-DSE: Feasibility-Aware HLS Design Space Exploration

A feasibility-aware Design Space Exploration (DSE) framework for High-Level
Synthesis, built around a single design principle: **action strength must
not exceed evidence strength**. PA-DSE targets two structurally different
HLS backends — PandA-Bambu (statically-scheduled) and Dynamatic
(dynamically-scheduled) — and consistently achieves higher success rate
with fewer wasted synthesis calls than classical optimisers or learned
surrogate baselines.

## Paper

The full manuscript is available in this repository as
[`pa_dse_paper.pdf`](pa_dse_paper.pdf) (single-file LaTeX source:
[`pa_dse_paper.tex`](pa_dse_paper.tex) with bibliography
[`references.bib`](references.bib)).

To rebuild the PDF:

```bash
pdflatex pa_dse_paper
bibtex   pa_dse_paper
pdflatex pa_dse_paper
pdflatex pa_dse_paper
```

## Method overview

PA-DSE has two layers:

- **Layer 1 — Static Constraint Filter (SCF)**: permanently removes
  configurations matching tool-documented incompatibility rules (e.g.\
  Bambu's `MEM_ACC_11` with `channels_number > 1`). Zero false-prune.
- **Layer 2 — Dynamic Failure Risk Learner (DFRL)**: learns online during
  a single run. Two sub-components:
  - **RPE (Recurrent Pattern Extractor)** — extracts typed failure
    signatures from observed errors and hard-skips matching configs.
  - **OFRS (Online Failure Risk Scorer)** — reorders the remaining queue
    by a smoothed per-dimension risk score.

Each component's action class (permanent block / hard skip / reorder) is
strictly bounded by the strength of the evidence it has access to.

## Tools & benchmarks

- **PandA-Bambu v0.9.8** — 420-config design space, 8 benchmarks:
  `matmul`, `vadd`, `fir`, `histogram`, `atax`, `bicg`, `gemm`, `gesummv`
- **Dynamatic v2.0.0** — 192-config design space, 4 benchmarks:
  `gcd`, `matching`, `binary_search`, `kernel_2mm`

## Baselines compared

Random, Filtered-Random, Simulated Annealing (SA), Genetic Algorithm
(GA), Gaussian-Process Bayesian Optimization (GP-BO), Random-Forest
feasibility classifier (RF, modelled after AutoScaleDSE).

## Project structure

```
hls-dse/
├── pa_dse_paper.tex          # Single-file manuscript (double-column)
├── pa_dse_paper.pdf          # Compiled paper
├── references.bib            # Bibliography (13 entries)
├── benchmarks/               # HLS benchmark source code (C)
│   ├── matmul/, vadd/, fir/, histogram/
│   └── atax/, bicg/, gemm/, gesummv/
├── scripts/                  # Experiment automation
│   ├── methods/              #   PA-DSE method, baselines
│   ├── runners/              #   Bambu / Dynamatic runners
│   ├── exp_logging/          #   Per-run CSV logging
│   └── analysis/             #   Post-processing
├── results/                  # Raw run summaries (per experiment)
├── tests/                    # Unit and integration tests
├── paper_figures/            # Figure-generation scripts
│   ├── compute_paper_tables.py         # Produces Tables 2–5 CSVs
│   ├── fig1_main_results_v2.py         # Fig 2: Bambu main comparison
│   ├── fig_dynamatic_main_v2.py        # Fig 3: Dynamatic main
│   ├── fig_cost.py                     # Fig 4: wasted calls + TTFF
│   ├── fig_perbench_heatmap.py         # Fig 5: per-benchmark SR
│   ├── fig_convergence.py              # Fig 6: cumulative feasibles
│   ├── fig_qor.py                      # Fig 7: QoR coverage
│   ├── fig_ablation_bar.py             # Fig 8: 8-way ablation
│   ├── fig_overhead_v3.py              # Fig 9: overhead breakdown
│   └── out/                            # Rendered PDFs / PNGs / tables
└── diagnose.sh               # Environment check
```

## Reproducing the paper figures

```bash
cd paper_figures
python3 compute_paper_tables.py   # Generates table_*.csv
python3 fig1_main_results_v2.py
python3 fig_dynamatic_main_v2.py
python3 fig_cost.py
python3 fig_perbench_heatmap.py
python3 fig_convergence.py
python3 fig_qor.py
python3 fig_ablation_bar.py
python3 fig_overhead_v3.py
```

All figures land in `paper_figures/out/` as both `.pdf` (for LaTeX) and
`.png` (for quick inspection). See individual scripts for the expected
input paths under `results/`.

## Requirements

- Python 3.10+ with `numpy`, `pandas`, `matplotlib`, `scipy`,
  `scikit-learn`
- Gurobi (for GP-BO and RF baseline ILP solves; licence not included)
- PandA-Bambu v0.9.8 and/or Dynamatic v2.0.0 for re-running experiments
- TeX Live 2025+ with the `IEEEtran` class for rebuilding the paper

## Author

**Xinyu Zhang** (Cindy) — Politecnico di Milano, DEIB
