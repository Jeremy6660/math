"""
图3(b)：278例痰湿体质患者的成本-效果散点图
（按活动能力分组着色）
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'Arial Unicode MS', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('output/p3_phlegm_patients_results.csv')

fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)

# 按活动能力分组着色
groups = {
    '差 (M<40)': df[df['mobility_group'] == '差(<40)'],
    '中 (40≤M<60)': df[df['mobility_group'] == '中(40-60)'],
    '好 (M≥60)': df[df['mobility_group'] == '好(≥60)']
}
colors = {'差 (M<40)': '#e74c3c', '中 (40≤M<60)': '#f39c12', '好 (M≥60)': '#2ecc71'}

for label, subset in groups.items():
    ax.scatter(subset['opt_cost'], subset['opt_s_drop'],
               c=colors[label], alpha=0.6, s=40, edgecolors='black', linewidth=0.3,
               label=f'{label} (n={len(subset)})')

ax.set_xlabel('总成本（元）', fontsize=12)
ax.set_ylabel('痰湿积分下降（分）', fontsize=12)
ax.set_title('278例痰湿体质患者成本-效果散点图（按活动能力分组）', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, linestyle='--', alpha=0.4)

# 添加成本上限线
ax.axvline(2000, color='gray', linestyle='--', linewidth=1.5, label='成本上限 2000元')

plt.tight_layout()
plt.savefig('figures/fig3b_cost_effect_scatter.png', dpi=300, bbox_inches='tight')
print("已保存 figures/fig3b_cost_effect_scatter.png")
