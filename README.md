# HLS Design Space Exploration (DSE) Framework

A framework to automatically explore different configurations of High-Level Synthesis (HLS) designs using **Bambu** and **Dynamatic**.

## Tools
- **PandA-Bambu** (v0.9.8) — Static HLS tool from Politecnico di Milano
- **Dynamatic** (MLIR-based) — Dynamic HLS tool from EPFL

## Benchmark
- Matrix Multiplication (8x8 integer matrices)

## Project Structure
```
hls-dse/
├── benchmarks/matmul/   # Benchmark source code
├── scripts/             # Automation scripts
├── results/             # Collected QoR results
└── docs/                # Reports and documentation
```

## Author
Cindy — Politecnico di Milano
