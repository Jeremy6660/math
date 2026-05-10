"""
图1(a)：关键指标与痰湿质积分、高血脂标签的相关性对比条形图
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'Arial Unicode MS', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('output/p1_correlation.csv')

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

indicators = df['指标'].values
x = np.arange(len(indicators))
width = 0.35

bars1 = ax.bar(x - width/2, df['与痰湿质Pearson'], width, label='与痰湿质Pearson $r$', color='steelblue', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, df['与高血脂点二列'], width, label='与高血脂点二列 $r$', color='coral', edgecolor='black', linewidth=0.5)

ax.set_ylabel('相关系数', fontsize=12)
ax.set_title('关键指标与目标变量的相关性对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(indicators, fontsize=10)
ax.axhline(0, color='black', linewidth=0.8)
ax.legend(fontsize=10, loc='upper right')
ax.set_ylim(-0.6, 0.6)

for bar in bars2:
    height = bar.get_height()
    if abs(height) > 0.15:
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -12),
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8, color='darkred')

plt.tight_layout()
plt.savefig('figures/fig1a_indicator_correlation.png', dpi=300, bbox_inches='tight')
print("已保存 figures/fig1a_indicator_correlation.png")
