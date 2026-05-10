"""
图2(b)：综合风险评分分布与三级风险阈值
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'Arial Unicode MS', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('output/p2_risk_scores.csv')

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 按风险等级分色绘制直方图
low = df[df['risk_level'] == '低']['risk_score']
mid = df[df['risk_level'] == '中']['risk_score']
high = df[df['risk_level'] == '高']['risk_score']

bins = np.linspace(df['risk_score'].min(), df['risk_score'].max(), 40)
ax.hist(low, bins=bins, alpha=0.75, label=f'低风险 (n={len(low)})', color='#2ecc71', edgecolor='white', linewidth=0.3)
ax.hist(mid, bins=bins, alpha=0.75, label=f'中风险 (n={len(mid)})', color='#f39c12', edgecolor='white', linewidth=0.3)
ax.hist(high, bins=bins, alpha=0.85, label=f'高风险 (n={len(high)})', color='#e74c3c', edgecolor='white', linewidth=0.3)

# 阈值线
q30 = df['risk_score'].quantile(0.30)
q90 = df['risk_score'].quantile(0.90)
anchor_high = 0.801

ax.axvline(q30, color='green', linestyle='--', linewidth=2, label=f'低中阈值 $\\theta_L$ = {q30:.3f}')
ax.axvline(anchor_high, color='red', linestyle='--', linewidth=2, label=f'高风险阈值 $\\theta_H$ = {anchor_high:.3f}')

ax.set_xlabel('综合风险评分 $R_i$', fontsize=12)
ax.set_ylabel('样本数', fontsize=12)
ax.set_title('综合风险评分分布与三级风险阈值划分', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('figures/fig2b_risk_score_distribution.png', dpi=300, bbox_inches='tight')
print("已保存 figures/fig2b_risk_score_distribution.png")
