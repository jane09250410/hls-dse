import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os


def load_results(filepath):
    df = pd.read_csv(filepath)
    df_ok = df[df['success'] == True].copy()
    print(f"Loaded {len(df)} configs, {len(df_ok)} successful")
    return df, df_ok


def find_pareto(df, col_x, col_y):
    df_clean = df.dropna(subset=[col_x, col_y]).copy()
    pareto_mask = []
    for i, row in df_clean.iterrows():
        dominated = False
        for j, other in df_clean.iterrows():
            if i == j:
                continue
            if (other[col_x] <= row[col_x] and other[col_y] <= row[col_y] and
                (other[col_x] < row[col_x] or other[col_y] < row[col_y])):
                dominated = True
                break
        pareto_mask.append(not dominated)
    return df_clean[pareto_mask]


def plot_pareto_front(df, output_dir):
    df_clean = df.dropna(subset=['total_area', 'num_states'])
    if df_clean.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    for mem, marker, color in [('ALL_BRAM', 'o', 'steelblue'), ('NO_BRAM', '^', 'coral')]:
        subset = df_clean[df_clean['memory_policy'] == mem]
        if not subset.empty:
            ax.scatter(subset['total_area'], subset['num_states'],
                      alpha=0.5, marker=marker, s=60, color=color, label=mem)
    pareto = find_pareto(df_clean, 'total_area', 'num_states')
    if not pareto.empty:
        pareto_sorted = pareto.sort_values('total_area')
        ax.scatter(pareto_sorted['total_area'], pareto_sorted['num_states'],
                  color='red', s=150, zorder=5, edgecolors='black', linewidth=2,
                  label='Pareto Optimal')
        ax.plot(pareto_sorted['total_area'], pareto_sorted['num_states'],
               'r--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Estimated Area', fontsize=12)
    ax.set_ylabel('Number of States (Latency)', fontsize=12)
    ax.set_title('Design Space Exploration: Area vs Latency', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pareto_front.png'), dpi=150)
    plt.close()
    print(f"  Saved pareto_front.png ({len(pareto)} Pareto points)")


def plot_parameter_impact(df, output_dir):
    df_clean = df.dropna(subset=['total_area', 'num_states'])
    if df_clean.empty:
        return
    params = ['clock_period', 'memory_policy', 'channels_type', 'channels_number']
    available_params = [p for p in params if p in df_clean.columns]
    fig, axes = plt.subplots(2, len(available_params),
                             figsize=(5 * len(available_params), 10))
    if len(available_params) == 1:
        axes = axes.reshape(-1, 1)
    for col_idx, param in enumerate(available_params):
        groups = df_clean.groupby(param)['total_area']
        means = groups.mean()
        stds = groups.std().fillna(0)
        x = range(len(means))
        axes[0, col_idx].bar(x, means, yerr=stds, alpha=0.7, capsize=5, color='steelblue')
        axes[0, col_idx].set_xticks(x)
        axes[0, col_idx].set_xticklabels([str(v) for v in means.index], rotation=45, ha='right')
        axes[0, col_idx].set_title(f'Area by {param}', fontsize=11)
        axes[0, col_idx].set_ylabel('Estimated Area')
        groups = df_clean.groupby(param)['num_states']
        means = groups.mean()
        stds = groups.std().fillna(0)
        axes[1, col_idx].bar(x, means, yerr=stds, alpha=0.7, capsize=5, color='coral')
        axes[1, col_idx].set_xticks(x)
        axes[1, col_idx].set_xticklabels([str(v) for v in means.index], rotation=45, ha='right')
        axes[1, col_idx].set_title(f'Latency by {param}', fontsize=11)
        axes[1, col_idx].set_ylabel('Number of States')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_impact.png'), dpi=150)
    plt.close()
    print(f"  Saved parameter_impact.png")


def plot_clock_vs_metrics(df, output_dir):
    df_clean = df.dropna(subset=['total_area', 'num_states', 'clock_period'])
    if df_clean.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for mem, color, label in [('ALL_BRAM', 'steelblue', 'ALL_BRAM'),
                               ('NO_BRAM', 'coral', 'NO_BRAM')]:
        subset = df_clean[df_clean['memory_policy'] == mem]
        if subset.empty:
            continue
        grouped = subset.groupby('clock_period')
        means = grouped['total_area'].mean()
        axes[0].plot(means.index, means.values, 'o-', color=color, label=label)
        axes[0].set_title('Clock Period vs Area')
        axes[0].set_xlabel('Clock Period (ns)')
        axes[0].set_ylabel('Estimated Area')
        means = grouped['num_states'].mean()
        axes[1].plot(means.index, means.values, 'o-', color=color, label=label)
        axes[1].set_title('Clock Period vs Latency')
        axes[1].set_xlabel('Clock Period (ns)')
        axes[1].set_ylabel('Number of States')
        if 'max_freq_mhz' in df_clean.columns:
            means = grouped['max_freq_mhz'].mean()
            axes[2].plot(means.index, means.values, 'o-', color=color, label=label)
            axes[2].set_title('Clock Period vs Max Frequency')
            axes[2].set_xlabel('Clock Period (ns)')
            axes[2].set_ylabel('Max Frequency (MHz)')
    for ax in axes:
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clock_vs_metrics.png'), dpi=150)
    plt.close()
    print(f"  Saved clock_vs_metrics.png")


def plot_resource_breakdown(df, output_dir):
    df_clean = df.dropna(subset=['num_dsps', 'num_ffs', 'num_registers'])
    if df_clean.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].hist(df_clean['num_dsps'], bins=10, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_title('DSP Usage Distribution')
    axes[0].set_xlabel('Number of DSPs')
    axes[1].hist(df_clean['num_ffs'], bins=15, alpha=0.7, color='coral', edgecolor='black')
    axes[1].set_title('Flip-Flop Distribution')
    axes[1].set_xlabel('Number of FFs')
    axes[2].hist(df_clean['num_registers'], bins=10, alpha=0.7, color='mediumpurple', edgecolor='black')
    axes[2].set_title('Register Distribution')
    axes[2].set_xlabel('Number of Registers')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'resource_breakdown.png'), dpi=150)
    plt.close()
    print(f"  Saved resource_breakdown.png")


def plot_heatmap(df, output_dir):
    df_clean = df.dropna(subset=['total_area', 'clock_period', 'channels_type'])
    if df_clean.empty:
        return
    pivot = df_clean.pivot_table(values='total_area', index='clock_period',
                                  columns='channels_type', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel('Channel Type')
    ax.set_ylabel('Clock Period (ns)')
    ax.set_title('Average Area: Clock Period x Channel Type')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=9)
    plt.colorbar(im, ax=ax, label='Estimated Area')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heatmap_area.png'), dpi=150)
    plt.close()
    print(f"  Saved heatmap_area.png")


def generate_summary(df, output_dir):
    df_ok = df[df['success'] == True]
    summary = []
    summary.append("=" * 60)
    summary.append("  DESIGN SPACE EXPLORATION - SUMMARY REPORT")
    summary.append("=" * 60)
    summary.append(f"\nTotal configurations: {len(df)}")
    summary.append(f"Successful: {len(df_ok)}")
    summary.append(f"Failed: {len(df) - len(df_ok)}")
    summary.append(f"Success rate: {len(df_ok)/len(df)*100:.1f}%")
    has_pipeline = df_ok['pipeline'].any() if 'pipeline' in df_ok.columns else False
    summary.append(f"Pipeline explored: {'Yes' if has_pipeline else 'No'}")
    if not df_ok.empty:
        unique = df_ok[['total_area', 'num_states']].dropna().drop_duplicates()
        summary.append(f"Unique QoR points: {len(unique)}")
        summary.append(f"\n--- Best Results ---")
        best_area_idx = df_ok['total_area'].idxmin()
        best_lat_idx = df_ok['num_states'].idxmin()
        if pd.notna(best_area_idx):
            b = df_ok.loc[best_area_idx]
            summary.append(f"\nSmallest area (config {int(b['id'])}):")
            summary.append(f"  Area={b['total_area']}, States={b['num_states']}, "
                         f"Clock={b['clock_period']}ns, Mem={b['memory_policy']}, "
                         f"Ch={b['channels_type']}")
        if pd.notna(best_lat_idx):
            b = df_ok.loc[best_lat_idx]
            summary.append(f"\nLowest latency (config {int(b['id'])}):")
            summary.append(f"  Area={b['total_area']}, States={b['num_states']}, "
                         f"Clock={b['clock_period']}ns, Mem={b['memory_policy']}, "
                         f"Ch={b['channels_type']}")
        summary.append(f"\n--- Statistics ---")
        for col in ['total_area', 'num_states', 'max_freq_mhz', 'num_dsps', 'num_ffs']:
            if col in df_ok.columns and df_ok[col].notna().any():
                summary.append(f"  {col}: min={df_ok[col].min():.1f}, "
                             f"max={df_ok[col].max():.1f}, mean={df_ok[col].mean():.1f}")
    summary_text = "\n".join(summary)
    print(summary_text)
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(summary_text)


def run_analysis(csv_path, output_dir="results"):
    print(f"\n{'='*60}")
    print(f"  Analyzing results from {csv_path}")
    print(f"{'='*60}\n")
    os.makedirs(output_dir, exist_ok=True)
    df_all, df_ok = load_results(csv_path)
    if df_ok.empty:
        print("No successful results to analyze!")
        return
    print("\nGenerating plots...")
    plot_pareto_front(df_ok, output_dir)
    plot_parameter_impact(df_ok, output_dir)
    plot_clock_vs_metrics(df_ok, output_dir)
    plot_resource_breakdown(df_ok, output_dir)
    plot_heatmap(df_ok, output_dir)
    print("\nGenerating summary...")
    generate_summary(df_all, output_dir)
    print(f"\n  Analysis complete! Check {output_dir}/ for results")


if __name__ == "__main__":
    run_analysis("results/bambu_results.csv", "results")
