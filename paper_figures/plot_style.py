"""Shared plot style for PA-DSE TRETS paper figures.

Import at the top of every figure script:
    from plot_style import setup, COLORS, METHOD_ORDER, save_fig
    setup()

Benchmark filtering:
    fir and histogram are EXCLUDED from Dynamatic (SR=100% for all methods,
    no discriminative signal). The paper (§IV.A) reports Dynamatic on 4
    benchmarks only. Any script aggregating Dynamatic data MUST filter to
    DYNAMATIC_BENCHMARKS or call filter_dynamatic(df).
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ======================= Method ordering & colors =======================
METHOD_ORDER = ["Random", "Filtered_Random", "SA", "GA", "GP-BO", "RF", "PA-DSE"]

# Colorblind-safe, PA-DSE = red for emphasis
COLORS = {
    "Random":           "#B0B0B0",  # gray
    "Filtered_Random":  "#808080",  # darker gray
    "SA":               "#F4A261",  # sand
    "GA":               "#E9C46A",  # muted yellow
    "GP-BO":            "#2A9D8F",  # teal
    "RF":               "#264653",  # dark slate
    "PA-DSE":           "#C1272D",  # red (highlight)
}

# ======================= Benchmark whitelists =======================
# Bambu: 8 benchmarks, all retained.
BAMBU_BENCHMARKS = ["matmul", "vadd", "fir", "histogram",
                    "atax", "bicg", "gemm", "gesummv"]

# Dynamatic: 4 benchmarks. fir, histogram EXCLUDED (SR=100% for all methods,
# no discriminative signal). Paper (§IV.A) reports on these 4 only.
DYNAMATIC_BENCHMARKS = ["gcd", "matching", "binary_search", "kernel_2mm"]

# Backward-compatibility alias: old scripts used BENCHMARK_ORDER for Bambu.
BENCHMARK_ORDER = BAMBU_BENCHMARKS


# ======================= Filter helpers =======================
def filter_bambu(df, col="benchmark", verbose=True):
    """Keep only Bambu benchmarks (noop in practice; present for symmetry)."""
    before = len(df)
    df = df[df[col].isin(BAMBU_BENCHMARKS)].copy()
    if verbose and before != len(df):
        print(f"  [filter_bambu] {before} -> {len(df)} rows")
    return df


def filter_dynamatic(df, col="benchmark", verbose=True):
    """Keep only the 4 discriminative Dynamatic benchmarks.

    Applies the paper's §IV.A filter: drops fir and histogram, which produce
    SR=100% for every method on Dynamatic. Call this on every Dynamatic
    aggregation so numbers match Table III.
    """
    before = len(df)
    df = df[df[col].isin(DYNAMATIC_BENCHMARKS)].copy()
    if verbose:
        print(f"  [filter_dynamatic] {before} -> {len(df)} rows "
              f"(kept: {DYNAMATIC_BENCHMARKS})")
    return df


# ======================= Matplotlib setup =======================
def setup():
    """Apply global style. Call once at the top of each figure script."""
    mpl.rcParams.update({
        # Fonts
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,

        # Lines & markers
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "axes.linewidth": 0.8,

        # Grid
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "grid.color": "#888888",
        "axes.axisbelow": True,

        # Ticks
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#888888",
        "legend.fancybox": False,

        # Saving
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,     # editable text in PDF
        "ps.fonttype": 42,
    })


# ======================= Paths =======================
RESULTS_ROOT = Path("/Users/zhangxinyu/Desktop/hls/results")
ANALYSIS_DIR = RESULTS_ROOT / "analysis"
FIGURES_DIR = Path("/Users/zhangxinyu/Desktop/hls/paper_figures/out")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ======================= Save helper =======================
def save_fig(fig, name):
    """Save figure as both PDF (paper) and PNG (preview) at 300dpi."""
    pdf_path = FIGURES_DIR / f"{name}.pdf"
    png_path = FIGURES_DIR / f"{name}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)  # lower dpi for PNG preview
    print(f"  [SAVED] {pdf_path}")
    print(f"  [SAVED] {png_path}")