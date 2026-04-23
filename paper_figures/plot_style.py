"""Shared plot style for PA-DSE TRETS paper figures.

Import at the top of every figure script:
    from plot_style import setup, COLORS, METHOD_ORDER, save_fig
    setup()
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

BENCHMARK_ORDER = ["matmul", "vadd", "fir", "histogram",
                   "atax", "bicg", "gemm", "gesummv"]


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
