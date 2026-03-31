import os
import time
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from config_generator import generate_bambu_configs, config_to_bambu_cmd
from run_exploration import run_bambu_single, extract_bambu_metrics, save_results_csv
from heuristic import grid_search, random_search, latin_hypercube_search
from analyze import find_pareto


def run_strategy(strategy_name, configs, src_file, top_func, results_dir):
    print(f"\n{'='*60}")
    print(f"  Strategy: {strategy_name} ({len(configs)} configs)")
    print(f"{'='*60}")
    abs_src = os.path.abspath(src_file)
    all_results = []
    start_time = time.time()
    for i, config in enumerate(configs):
        work_dir = os.path.join(results_dir, strategy_name.replace(' ', '_'), f"config_{config['id']}")
        cmd = config_to_bambu_cmd(config, abs_src, top_func)
        pipe_str = f"ii={config['pipeline_ii']}" if config['pipeline'] else "off"
        print(f"  [{i+1}/{len(configs)}] Config {config['id']}: "
              f"clock={config['clock_period']}ns, pipe={pipe_str}, "
              f"mem={config['memory_policy']}, ch={config['channels_type']}, "
              f"ch_num={config['channels_number']}")
        output, elapsed, success = run_bambu_single(cmd, work_dir)
        metrics = extract_bambu_metrics(output)
        result = {**config, **metrics, 'runtime_s': round(elapsed, 2),
                  'success': success, 'strategy': strategy_name}
        all_results.append(result)
        if success:
            print(f"       -> area={metrics.get('total_area', '?')}, "
                  f"states={metrics.get('num_states', '?')}, "
                  f"freq={metrics.get('max_freq_mhz', '?')}MHz")
        else:
            print(f"       -> FAILED ({elapsed:.1f}s)")
    total_time = time.time() - start_time
    df = pd.DataFrame(all_results)
    df_ok = df[df['success'] == True]
    stats = {
        'strategy': strategy_name,
        'total_configs': len(configs),
        'successful': len(df_ok),
        'failed': len(configs) - len(df_ok),
        'success_rate_pct': round(len(df_ok) / len(configs) * 100, 1) if configs else 0,
        'total_time_s': round(total_time, 1),
    }
    if not df_ok.empty and df_ok['total_area'].notna().any():
        stats['best_area'] = df_ok['total_area'].min()
        stats['mean_area'] = round(df_ok['total_area'].mean(), 1)
        stats['best_latency'] = df_ok['num_states'].min()
        stats['mean_latency'] = round(df_ok['num_states'].mean(), 1)
        pareto = find_pareto(df_ok, 'total_area', 'num_states')
        stats['pareto_points'] = len(pareto)
        unique = df_ok[['total_area', 'num_states']].drop_duplicates()
        stats['unique_qor_points'] = len(unique)
    else:
        stats.update({'best_area': None, 'mean_area': None, 'best_latency': None,
                      'mean_latency': None, 'pareto_points': 0, 'unique_qor_points': 0})
    return all_results, stats


def plot_strategy_comparison(all_stats, output_dir):
    df = pd.DataFrame(all_stats)
    strategies = df['strategy'].tolist()
    x = range(len(strategies))
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes[0, 0].bar(x, df['total_configs'], color='steelblue', alpha=0.8)
    axes[0, 0].set_title('Configs Explored')
    axes[0, 0].set_xticks(x); axes[0, 0].set_xticklabels(strategies, rotation=20)
    axes[0, 1].bar(x, df['total_time_s'], color='coral', alpha=0.8)
    axes[0, 1].set_title('Total Runtime (s)')
    axes[0, 1].set_xticks(x); axes[0, 1].set_xticklabels(strategies, rotation=20)
    axes[0, 2].bar(x, df['best_area'].fillna(0), color='mediumpurple', alpha=0.8)
    axes[0, 2].set_title('Best Area Found')
    axes[0, 2].set_xticks(x); axes[0, 2].set_xticklabels(strategies, rotation=20)
    axes[1, 0].bar(x, df['best_latency'].fillna(0), color='seagreen', alpha=0.8)
    axes[1, 0].set_title('Best Latency Found')
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(strategies, rotation=20)
    axes[1, 1].bar(x, df['pareto_points'], color='orange', alpha=0.8)
    axes[1, 1].set_title('Pareto Optimal Points')
    axes[1, 1].set_xticks(x); axes[1, 1].set_xticklabels(strategies, rotation=20)
    success_rates = df['success_rate_pct'].fillna(0)
    axes[1, 2].bar(x, success_rates, color='goldenrod', alpha=0.8)
    axes[1, 2].set_title('Success Rate (%)')
    axes[1, 2].set_xticks(x); axes[1, 2].set_xticklabels(strategies, rotation=20)
    axes[1, 2].set_ylim(0, 105)
    plt.suptitle('Exploration Strategy Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strategy_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved strategy_comparison.png")


def plot_strategy_pareto_overlay(all_results_dict, output_dir):
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['steelblue', 'coral', 'mediumpurple']
    markers = ['o', '^', 's']
    for i, (strategy, results) in enumerate(all_results_dict.items()):
        df = pd.DataFrame(results)
        df_ok = df[(df['success'] == True) & df['total_area'].notna() & df['num_states'].notna()]
        if df_ok.empty:
            continue
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        ax.scatter(df_ok['total_area'], df_ok['num_states'], alpha=0.3, color=color, marker=marker, s=30)
        pareto = find_pareto(df_ok, 'total_area', 'num_states')
        if not pareto.empty:
            pareto_sorted = pareto.sort_values('total_area')
            ax.plot(pareto_sorted['total_area'], pareto_sorted['num_states'],
                   'o-', color=color, markersize=8, linewidth=2,
                   label=f'{strategy} Pareto ({len(pareto)} pts)')
    ax.set_xlabel('Estimated Area', fontsize=12)
    ax.set_ylabel('Number of States (Latency)', fontsize=12)
    ax.set_title('Pareto Fronts by Strategy', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pareto_overlay.png'), dpi=150)
    plt.close()
    print(f"  Saved pareto_overlay.png")


def run_comparison(src_file, top_func, results_dir="results", max_per_strategy=None,
                   enable_pipeline=False):
    configs = generate_bambu_configs(enable_pipeline=enable_pipeline)
    param_keys = ['clock_period', 'memory_policy', 'channels_type', 'channels_number']
    if enable_pipeline:
        param_keys.append('pipeline')
    strategies = {
        'Grid Search': grid_search(configs),
        'Random 30%': random_search(configs, sample_ratio=0.3),
        'LHS 20%': latin_hypercube_search(configs, param_keys, sample_ratio=0.2),
    }
    if max_per_strategy:
        strategies = {k: v[:max_per_strategy] for k, v in strategies.items()}
    all_stats = []
    all_results_dict = {}
    for strategy_name, strategy_configs in strategies.items():
        results, stats = run_strategy(strategy_name, strategy_configs, src_file, top_func, results_dir)
        all_stats.append(stats)
        all_results_dict[strategy_name] = results
        safe_name = strategy_name.replace(' ', '_').replace('%', 'pct')
        save_results_csv(results, os.path.join(results_dir, f'{safe_name}_results.csv'))
    stats_df = pd.DataFrame(all_stats)
    stats_df.to_csv(os.path.join(results_dir, 'strategy_stats.csv'), index=False)
    print(f"\n{stats_df.to_string(index=False)}")
    plot_strategy_comparison(all_stats, results_dir)
    plot_strategy_pareto_overlay(all_results_dict, results_dir)
    return all_stats, all_results_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--enable-pipeline", action="store_true")
    args = parser.parse_args()
    run_comparison(src_file="benchmarks/matmul/matmul.c", top_func="matmul",
                   max_per_strategy=args.max, enable_pipeline=args.enable_pipeline)
