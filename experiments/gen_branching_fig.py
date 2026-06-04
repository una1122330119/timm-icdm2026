#!/usr/bin/env python3
"""Generate branching factor sensitivity figure."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')

# Real data from your experiment
b_values = [0.30, 0.50, 0.70, 0.90]
timm_spread = [510.5, 1161.6, 1992.8, 2800.0]
static_spread = [440.3, 1055.5, 1865.1, 2671.2]
random_spread = [24.4, 32.5, 45.7, 74.7]
ti_gain_pct = [16, 10, 7, 5]

plt.rcParams.update({'font.size': 10, 'font.family': 'serif', 'figure.dpi': 150})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left panel: absolute spread
ax1.plot(b_values, timm_spread, 'o-', color='#1f77b4', linewidth=2, markersize=8, label='TIMM')
ax1.plot(b_values, static_spread, 's-', color='#ff7f0e', linewidth=2, markersize=8, label='StaticIMM')
ax1.plot(b_values, random_spread, 'x--', color='#7f7f7f', linewidth=1.5, markersize=8, label='Random')
ax1.set_xlabel('Effective Branching Factor $b$', fontweight='bold')
ax1.set_ylabel('Influence Spread', fontweight='bold')
ax1.set_title('Spread vs Branching Factor (DBLP, $k=20$)', fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)
ax1.annotate('TIMM consistently\nabove StaticIMM', xy=(0.7, 2050), fontsize=8,
            color='#1f77b4', ha='center')

# Right panel: relative gain
colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#1f77b4']
bars = ax2.bar(range(4), ti_gain_pct, color=colors, edgecolor='white', width=0.6)
ax2.set_xticks(range(4))
ax2.set_xticklabels([f'$b={b:.2f}$' for b in b_values])
ax2.set_ylabel('TIMM Gain over StaticIMM (%)', fontweight='bold')
ax2.set_title('Temporal Advantage by Cascade Regime', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for bar, v in zip(bars, ti_gain_pct):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
            f'+{v}%', ha='center', fontsize=10, fontweight='bold', color='#1f77b4')
ax2.set_ylim(0, 22)
ax2.annotate('Sparse → Dense\ncascade regime', xy=(3, 12), fontsize=7,
            ha='center', style='italic')

fig.suptitle('Branching Factor Robustness', fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_branching_sensitivity.pdf'), bbox_inches='tight')
fig.savefig(os.path.join(OUT, 'fig_branching_sensitivity.png'), bbox_inches='tight', dpi=300)
plt.close(fig)
print(f'Saved to {OUT}')


