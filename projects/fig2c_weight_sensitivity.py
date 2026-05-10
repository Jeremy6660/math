"""
图2(c)：风险评分权重敏感性分析
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'Arial Unicode MS', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 根据修正后代码实际运行结果（problem2.py 敏感性分析输出）
scenarios = ['基准\n(组合赋权)', r'$\alpha-20\%$', r'$\alpha+20\%$', r'$\beta-20\%$', r'$\beta+20\%$']
high_risk_pct = [9.4, 5.1, 17.8, 9.4, 9.6]
changes = [0, -4.3, 8.4, 0, 0.2]

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

colors = ['steelblue'] + ['coral' if c < 0 else 'seagreen' for c in changes[1:]]
bars = ax.bar(scenarios, high_risk_pct, color=colors, edgecolor='black', linewidth=0.5, alpha=0.85)

ax.set_ylabel('高风险人群比例 (%)', fontsize=12)
ax.set_title('风险评分权重敏感性分析（高风险比例变化）', fontsize=14, fontweight='bold')
ax.set_ylim(0, 22)
ax.axhline(9.4, color='gray', linestyle='--', linewidth=1.2, label='基准值 9.4%')

for bar, val, chg in zip(bars, high_risk_pct, changes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}%\n({chg:+.1f}pp)', ha='center', va='bottom', fontsize=9)

ax.legend(fontsize=10)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('figures/fig2c_weight_sensitivity.png', dpi=300, bbox_inches='tight')
print("已保存 figures/fig2c_weight_sensitivity.png")
