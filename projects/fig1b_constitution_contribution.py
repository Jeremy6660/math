"""
图1(b)：九种体质对高血脂发病风险的贡献度对比
（标准化Logistic回归系数 vs 随机森林Gini重要性）
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'Arial Unicode MS', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

coef_df = pd.read_csv('output/p1_constitution_logistic.csv')
rf_df = pd.read_csv('output/p1_rf_importance.csv')

# 统一体质顺序
constitution_names = {
    'pinghe': '平和质', 'qixu': '气虚质', 'yangxu': '阳虚质',
    'yinxu': '阴虚质', 'phlegm': '痰湿质', 'shire': '湿热质',
    'xueyu': '血瘀质', 'qiyu': '气郁质', 'tebing': '特禀质'
}

coef_df['体质中文'] = coef_df['体质'].map(constitution_names)
rf_df['体质中文'] = rf_df['体质'].map(constitution_names)

# 按Logistic系数绝对值排序
order = coef_df.sort_values('abs_coef', ascending=False)['体质中文'].values
coef_df = coef_df.set_index('体质中文').reindex(order).reset_index()
rf_df = rf_df.set_index('体质中文').reindex(order).reset_index()

fig, ax1 = plt.subplots(figsize=(11, 6), dpi=300)

x = np.arange(len(order))
width = 0.35

colors_coef = ['firebrick' if p >= 0.05 else 'steelblue' for p in coef_df['pvalue']]
bars1 = ax1.bar(x - width/2, coef_df['coef'], width, label='标准化Logistic系数', color=colors_coef, edgecolor='black', linewidth=0.5)
ax1.set_ylabel(r'标准化Logistic系数 $\beta^{*}$', color='steelblue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.axhline(0, color='black', linewidth=0.8)
ax1.set_ylim(-0.15, 0.18)

ax2 = ax1.twinx()
bars2 = ax2.bar(x + width/2, rf_df['Gini重要性'], width, label='随机森林Gini重要性', color='darkorange', edgecolor='black', linewidth=0.5, alpha=0.8)
ax2.set_ylabel('Gini重要性', color='darkorange', fontsize=12)
ax2.tick_params(axis='y', labelcolor='darkorange')
ax2.set_ylim(0, 0.18)

ax1.set_xticks(x)
ax1.set_xticklabels(order, fontsize=10)
ax1.set_title('九种体质对高血脂发病风险的贡献度对比', fontsize=14, fontweight='bold')

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='steelblue', edgecolor='black', label='Logistic系数 (p<0.05)'),
    Patch(facecolor='firebrick', edgecolor='black', label='Logistic系数 (p≥0.05)'),
    Patch(facecolor='darkorange', edgecolor='black', label='随机森林Gini重要性')
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('figures/fig1b_constitution_contribution.png', dpi=300, bbox_inches='tight')
print("已保存 figures/fig1b_constitution_contribution.png")
