"""
图3(c)：依从性衰减敏感性分析
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'Arial Unicode MS', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('output/p3_sensitivity_analysis.csv')
summary = df.groupby('config').agg({
    'avg_cost_278': 'first',
    'avg_drop_278': 'first'
}).reindex(['保守方案', '基准方案', '乐观方案'])

fig, ax1 = plt.subplots(figsize=(9, 6.5), dpi=300)

x_pos = np.arange(len(summary))
width = 0.35

# 柱状图：平均成本
bars = ax1.bar(x_pos - width/2, summary['avg_cost_278'], width,
               color='steelblue', edgecolor='black', linewidth=0.5, alpha=0.85,
               label='平均成本（元）')
ax1.set_ylabel('平均成本（元）', color='steelblue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_ylim(1200, 1400)

# 折线图：平均降分
ax2 = ax1.twinx()
line = ax2.plot(x_pos + width/2, summary['avg_drop_278'], 'o-',
                color='darkorange', linewidth=2.5, markersize=10,
                markeredgecolor='black', markeredgewidth=0.8,
                label='平均降分（分）')
ax2.set_ylabel('平均降分（分）', color='darkorange', fontsize=12)
ax2.tick_params(axis='y', labelcolor='darkorange')
ax2.set_ylim(12, 20)

# 标注数值
for i, (cost, drop) in enumerate(zip(summary['avg_cost_278'], summary['avg_drop_278'])):
    ax1.text(i - width/2, cost + 5, f'{cost:.1f}', ha='center', va='bottom', fontsize=9, color='steelblue')
    ax2.text(i + width/2, drop + 0.25, f'{drop:.2f}', ha='center', va='bottom', fontsize=9, color='darkorange')

ax1.set_xticks(x_pos)
ax1.set_xticklabels(summary.index, fontsize=11)
ax1.set_title('依从性衰减敏感性分析（278人平均）', fontsize=14, fontweight='bold')

from matplotlib.lines import Line2D
legend_elements = [Patch(facecolor='steelblue', edgecolor='black', label='平均成本（元）'),
                   Line2D([0], [0], color='darkorange', marker='o', markersize=8, label='平均降分（分）')]
ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

ax1.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('figures/fig3c_compliance_sensitivity.png', dpi=300, bbox_inches='tight')
print("已保存 figures/fig3c_compliance_sensitivity.png")
