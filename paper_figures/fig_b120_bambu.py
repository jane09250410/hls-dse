#!/usr/bin/env python3
"""
fig_b120_bambu.py
B=120 Bambu SR bar chart showing RPE contribution.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from plot_style import COLORS as BASE_COLORS

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

ROOT = Path(__file__).resolve().parent.parent / "results"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "b120/bambu_main/run_summary.csv")

RENAME = {'Random':'Random','Filtered_Random':'FilteredRandom',
          'SimulatedAnnealing':'SA','GeneticAlgorithm':'GA',
          'GP-BO':'GP-BO','RF_Classifier':'RF',
          'PA-DSE_SCF+DFRL':'PA-DSE','PA-DSE_SCF+OFRS':'SCF+OFRS'}
df['method'] = df['strategy'].map(RENAME).fillna(df['strategy'])

ORDER = ['Random','FilteredRandom','SA','GA','GP-BO','RF','SCF+OFRS','PA-DSE']

COLORS = {
    'Random':         BASE_COLORS['Random'],
    'FilteredRandom': BASE_COLORS['Filtered_Random'],
    'SA':             BASE_COLORS['SA'],
    'GA':             BASE_COLORS['GA'],
    'GP-BO':          BASE_COLORS['GP-BO'],
    'RF':             BASE_COLORS['RF'],
    'SCF+OFRS':       '#6B8E9B',
    'PA-DSE':         BASE_COLORS['PA-DSE'],
}

fig, ax = plt.subplots(figsize=(7, 4))

means = [df[df['method']==m]['sr_pct'].mean() for m in ORDER]
stds = [df[df['method']==m]['sr_pct'].std() for m in ORDER]
colors = [COLORS[m] for m in ORDER]

x = np.arange(len(ORDER))
bars = ax.bar(x, means, yerr=stds, color=colors, edgecolor='black',
              linewidth=0.6, capsize=3, error_kw={'elinewidth':0.8, 'ecolor':'#333'})

for i, (m, v, s) in enumerate(zip(ORDER, means, stds)):
    ax.text(i, v+s+2, f'{v:.1f}', ha='center', fontsize=8,
            fontweight='bold' if m=='PA-DSE' else 'normal',
            color=COLORS[m] if m in ('PA-DSE','SCF+OFRS') else '#333')

ax.annotate('', xy=(7, 83), xytext=(6, 47),
            arrowprops=dict(arrowstyle='<->', color=COLORS['PA-DSE'], lw=1.5))
ax.text(6.5, 64, '+36.3 pp\n(RPE)', ha='center', fontsize=9,
        color=COLORS['PA-DSE'], fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(ORDER, rotation=30, ha='right')
ax.set_ylabel('Success Rate (%)')
ax.set_ylim(0, 105)
ax.set_title('Bambu at B=120: RPE contributes +36.3 pp over OFRS-only')
plt.tight_layout()
plt.savefig(OUT / 'fig_b120_bambu.pdf', bbox_inches='tight')
plt.savefig(OUT / 'fig_b120_bambu.png', bbox_inches='tight')
plt.close()
print(f"Saved: {OUT / 'fig_b120_bambu.pdf'}")
