"""fig_architecture.py — FA-DSE architecture diagram.

Renders into paper_figures/out/fig_architecture.{pdf,png}, which is the
path referenced by pa_dse_paper.tex via
  \\includegraphics[width=\\textwidth]{paper_figures/out/fig_architecture}

Run from anywhere:
    python3 paper_figures/fig_architecture.py
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch, Circle

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

W, H = 16.72, 9.41
red = "#cf1f16"
dash = (0, (5, 3))


def rounded(ax, x, y, w, h, fc="white", ec="black", lw=1.3, r=0.12,
            linestyle="solid", z=2):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.035,rounding_size={r}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        linestyle=linestyle,
        zorder=z
    )
    ax.add_patch(box)
    return box


def arrow(ax, xy1, xy2, color="black", lw=1.25, ls="solid",
          ms=16, z=4, shrinkA=0, shrinkB=0):
    arr = FancyArrowPatch(
        xy1, xy2,
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        linestyle=ls,
        shrinkA=shrinkA,
        shrinkB=shrinkB,
        zorder=z
    )
    ax.add_patch(arr)
    return arr


fig, ax = plt.subplots(figsize=(W, H), dpi=100)
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# =========================
# DFRL outer dashed region
# =========================
ax.add_patch(Rectangle(
    (7.00, 1.88), 6.78, 6.96,
    fill=False,
    edgecolor=red,
    linewidth=1.3,
    linestyle=dash,
    zorder=0
))

ax.text(
    10.38, 9.10,
    "DFRL (Layer 2): Dynamic Failure Risk Learning",
    ha="center",
    va="bottom",
    fontsize=16,
    color=red,
    fontweight="bold"
)

# =========================
# Config space
# =========================
rounded(ax, 0.15, 4.40, 1.95, 1.62, fc="white", r=0.06)

ax.text(
    1.125, 5.58,
    r"Config space $C$",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold"
)

ax.text(
    1.125, 4.88,
    "N=420 for Bambu,\n192 for Dynamatic",
    ha="center",
    va="center",
    fontsize=11.5,
    linespacing=1.25
)

# =========================
# SCF box
# =========================
rounded(ax, 3.20, 3.54, 3.12, 3.26, fc="#e9f2fb", r=0.13)

ax.text(
    4.76, 6.43,
    "SCF (Layer 1):\nStatic Constraint Filter",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    linespacing=1.15
)

ax.text(
    4.76, 5.78,
    "Rule-based, zero false-prune",
    ha="center",
    va="center",
    fontsize=12.3,
    style="italic",
    color="#4b4b4b"
)

# Partition box
ax.add_patch(FancyBboxPatch(
    (3.32, 3.72), 2.88, 1.53,
    boxstyle="round,pad=0.025,rounding_size=0.12",
    fill=False,
    edgecolor="#555555",
    linewidth=1.0,
    linestyle="--",
    zorder=3
))

ax.text(
    4.76, 5.06,
    r"Partition of $C$",
    ha="center",
    va="center",
    fontsize=12
)

# Inner partition rectangles
x0, y0, bw, bh, gap = 3.42, 3.88, 0.88, 1.02, 0.08

ax.add_patch(Rectangle(
    (x0, y0), bw, bh,
    facecolor="#3b3b3b",
    edgecolor="black",
    linewidth=1.0,
    zorder=3
))
ax.text(x0 + bw / 2, y0 + 0.58, r"$C_{blocked}$",
        ha="center", va="center", fontsize=11, color="white", fontweight="bold")
ax.text(x0 + bw / 2, y0 + 0.31, "(removed)",
        ha="center", va="center", fontsize=9.5, color="white")

x1 = x0 + bw + gap
ax.add_patch(Rectangle(
    (x1, y0), bw, bh,
    facecolor="#cfcfcf",
    edgecolor="black",
    linewidth=1.0,
    zorder=3
))
ax.text(x1 + bw / 2, y0 + 0.58, r"$C_{suppressed}$",
        ha="center", va="center", fontsize=9.4, color="black", fontweight="bold")
ax.text(x1 + bw / 2, y0 + 0.31, "(deferred)",
        ha="center", va="center", fontsize=9.5, color="black")

x2 = x1 + bw + gap
ax.add_patch(Rectangle(
    (x2, y0), bw, bh,
    facecolor="white",
    edgecolor="black",
    linewidth=1.0,
    zorder=3
))
ax.text(x2 + bw / 2, y0 + 0.58, r"$C_{active}$",
        ha="center", va="center", fontsize=11, color="black", fontweight="bold")
ax.text(x2 + bw / 2, y0 + 0.31, "(kept)",
        ha="center", va="center", fontsize=9.5, color="black")

# =========================
# OFRS
# =========================
rounded(ax, 7.36, 7.25, 4.72, 1.22, fc="#dcefee", r=0.14)

ax.text(
    9.72, 8.22,
    "OFRS: Online Failure Risk Scorer",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold"
)

ax.text(
    9.72, 7.72,
    "Reorders queue by smoothed\nper-dimension risk",
    ha="center",
    va="center",
    fontsize=12.8,
    style="italic",
    color="#555555",
    linespacing=1.2
)

# =========================
# Queue
# =========================
rounded(ax, 7.66, 4.65, 2.74, 1.16, fc="#fff7dd", r=0.14)

ax.text(
    9.03, 5.55,
    r"Queue $Q_t$",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold"
)

cx = 7.96
for i in range(4):
    ax.add_patch(Circle(
        (cx + i * 0.33, 5.02),
        0.12,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        zorder=4
    ))

ax.text(9.27, 5.02, r"$\cdots$", ha="center", va="center", fontsize=15)

for i in range(3):
    ax.add_patch(Circle(
        (9.60 + i * 0.32, 5.02),
        0.12,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        zorder=4
    ))

# =========================
# HLS synthesis
# =========================
rounded(ax, 11.98, 4.68, 2.12, 1.08, fc="#eeeeee", r=0.08)

ax.text(
    13.04, 5.30,
    "HLS Synthesis",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold"
)

ax.text(
    13.04, 4.94,
    "(Bambu / Dynamatic)",
    ha="center",
    va="center",
    fontsize=12
)

# =========================
# Outcome
# =========================
rounded(ax, 14.82, 4.65, 1.45, 1.18, fc="white", r=0.06)

ax.text(
    15.545, 5.32,
    "Outcome",
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold"
)

ax.text(
    15.545, 4.93,
    "(success /\nfail-type)",
    ha="center",
    va="center",
    fontsize=11.5,
    linespacing=1.05
)

# =========================
# RPE
# =========================
rounded(ax, 9.28, 2.02, 3.92, 0.98, fc="#ffe1ce", r=0.10)

ax.text(
    11.24, 2.67,
    "RPE: Recurrent Pattern Extractor",
    ha="center",
    va="center",
    fontsize=13,
    fontweight="bold"
)

ax.text(
    11.24, 2.32,
    "Extracts failure signatures",
    ha="center",
    va="center",
    fontsize=12,
    style="italic",
    color="#5a3d31"
)

# =========================
# Solid arrows
# =========================
arrow(ax, (2.10, 5.22), (3.18, 5.22))
ax.text(
    2.68, 5.42,
    "filter",
    ha="center",
    va="center",
    fontsize=11,
    fontfamily="DejaVu Sans Mono"
)

arrow(ax, (6.32, 5.22), (7.62, 5.22))
ax.text(
    6.95, 5.44,
    r"initialize $Q$",
    ha="center",
    va="center",
    fontsize=11,
    fontfamily="DejaVu Sans Mono"
)

arrow(ax, (9.02, 7.25), (9.02, 5.82))
ax.text(
    9.20, 6.62,
    r"sort by r(c)",
    ha="left",
    va="center",
    fontsize=11,
    fontfamily="DejaVu Sans Mono"
)

arrow(ax, (10.40, 5.22), (11.96, 5.22))
ax.text(
    11.20, 5.45,
    "pop lowest-risk",
    ha="center",
    va="center",
    fontsize=10,
    fontfamily="DejaVu Sans Mono"
)

arrow(ax, (14.10, 5.22), (14.80, 5.22))

# =========================
# Red dashed feedback arrows
# =========================

# update statistics
ax.plot(
    [15.45, 15.45],
    [4.65, 7.94],
    color=red,
    linewidth=1.25,
    linestyle=dash,
    zorder=1
)
arrow(
    ax,
    (15.45, 7.94),
    (12.10, 7.94),
    color=red,
    lw=1.25,
    ls=dash,
    ms=15
)
ax.text(
    13.58, 7.66,
    "update statistics",
    ha="center",
    va="center",
    fontsize=10.5,
    fontfamily="DejaVu Sans Mono"
)

# extract signatures
ax.plot(
    [15.45, 15.45],
    [4.65, 2.58],
    color=red,
    linewidth=1.25,
    linestyle=dash,
    zorder=1
)
arrow(
    ax,
    (15.45, 2.58),
    (13.22, 2.58),
    color=red,
    lw=1.25,
    ls=dash,
    ms=15
)
ax.text(
    15.25, 3.62,
    "extract signatures",
    ha="center",
    va="center",
    fontsize=10.5,
    fontfamily="DejaVu Sans Mono"
)

# skip matched configs
ax.plot(
    [9.28, 8.28],
    [2.58, 2.58],
    color=red,
    linewidth=1.25,
    linestyle=dash,
    zorder=1
)
ax.plot(
    [8.28, 8.28],
    [2.58, 4.22],
    color=red,
    linewidth=1.25,
    linestyle=dash,
    zorder=1
)
arrow(
    ax,
    (8.28, 4.22),
    (8.28, 4.63),
    color=red,
    lw=1.25,
    ls=dash,
    ms=15
)
ax.text(
    9.15, 3.72,
    "skip matched configs",
    ha="center",
    va="center",
    fontsize=10.5,
    fontfamily="DejaVu Sans Mono"
)

# =========================
# Evidence hierarchy
# =========================
rounded(ax, 3.02, 0.43, 10.70, 0.92, fc="white", r=0.06)

ax.text(
    8.37, 0.88,
    "Evidence hierarchy:  SCF (proven) → permanent block | "
    "RPE (typed empirical) → hard skip |\n"
    "OFRS (marginal empirical) → reorder only",
    ha="center",
    va="center",
    fontsize=13,
    linespacing=1.25
)

# =========================
# Save — to paper_figures/out/ so that paper picks it up
# =========================
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(out_dir, exist_ok=True)

plt.savefig(os.path.join(out_dir, "fig_architecture.png"),
            dpi=200, facecolor="white", bbox_inches="tight")
plt.savefig(os.path.join(out_dir, "fig_architecture.pdf"),
            facecolor="white", bbox_inches="tight")

print(f"Saved fig_architecture.pdf and .png to {out_dir}")
