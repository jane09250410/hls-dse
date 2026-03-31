#!/usr/bin/env python3
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from config_generator import generate_bambu_configs, save_configs
from run_exploration import run_all_bambu, save_results_csv
from analyze import run_analysis
from compare_strategies import run_comparison


def main():
    parser = argparse.ArgumentParser(description="HLS Design Space Exploration Framework")
    parser.add_argument("--mode", choices=["test", "bambu", "compare", "full", "analyze"],
                        default="test", help="Execution mode")
    parser.add_argument("--max", type=int, default=None, help="Max configs to run")
    parser.add_argument("--enable-pipeline", action="store_true",
                        help="Include pipeline configurations in search space")
    args = parser.parse_args()

    src_file = "benchmarks/matmul/matmul.c"
    top_func = "matmul"

    # Results go to different dirs based on pipeline setting
    if args.enable_pipeline:
        results_dir = "results/with_pipeline"
        space_label = "WITH pipeline"
    else:
        results_dir = "results/no_pipeline"
        space_label = "NO pipeline"

    print("=" * 60)
    print("  HLS Design Space Exploration Framework")
    print("  Politecnico di Milano")
    print("=" * 60)
    print(f"  Mode: {args.mode}")
    print(f"  Search space: {space_label}")
    print(f"  Benchmark: {src_file}")
    print(f"  Results dir: {results_dir}")
    print("=" * 60)

    if args.mode == "test":
        max_configs = args.max or 5
        configs = generate_bambu_configs(enable_pipeline=args.enable_pipeline)
        save_configs(configs[:max_configs], f"{results_dir}/bambu_configs.json")
        results = run_all_bambu(configs, src_file, top_func, max_configs=max_configs,
                               results_dir=f"{results_dir}/bambu")
        save_results_csv(results, f"{results_dir}/bambu_results.csv")
        run_analysis(f"{results_dir}/bambu_results.csv", results_dir)

    elif args.mode == "bambu":
        configs = generate_bambu_configs(enable_pipeline=args.enable_pipeline)
        save_configs(configs, f"{results_dir}/bambu_configs.json")
        results = run_all_bambu(configs, src_file, top_func, max_configs=args.max,
                               results_dir=f"{results_dir}/bambu")
        save_results_csv(results, f"{results_dir}/bambu_results.csv")
        run_analysis(f"{results_dir}/bambu_results.csv", results_dir)

    elif args.mode == "compare":
        run_comparison(src_file, top_func, results_dir,
                      max_per_strategy=args.max,
                      enable_pipeline=args.enable_pipeline)

    elif args.mode == "full":
        configs = generate_bambu_configs(enable_pipeline=args.enable_pipeline)
        save_configs(configs, f"{results_dir}/bambu_configs.json")
        results = run_all_bambu(configs, src_file, top_func, max_configs=args.max,
                               results_dir=f"{results_dir}/bambu")
        save_results_csv(results, f"{results_dir}/bambu_results.csv")
        run_analysis(f"{results_dir}/bambu_results.csv", results_dir)
        run_comparison(src_file, top_func, results_dir,
                      max_per_strategy=args.max,
                      enable_pipeline=args.enable_pipeline)

    elif args.mode == "analyze":
        csv_path = f"{results_dir}/bambu_results.csv"
        if not os.path.exists(csv_path):
            print(f"Error: {csv_path} not found. Run bambu mode first.")
            sys.exit(1)
        run_analysis(csv_path, results_dir)

    print("\n" + "=" * 60)
    print(f"  Done! Check {results_dir}/ for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
