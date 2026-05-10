"""
图2(a)：两层模型ROC曲线对比
（现症识别模型 vs 治未病预警模型）
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'Arial Unicode MS', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('output/p2_risk_scores.csv')

fig, ax = plt.subplots(figsize=(8, 7), dpi=300)

# 治未病预警模型ROC（使用5折交叉验证out-of-fold预测概率）
fpr_prev, tpr_prev, _ = roc_curve(df['disease'], df['oof_prob_prev'])
auc_prev = auc(fpr_prev, tpr_prev)
ax.plot(fpr_prev, tpr_prev, color='steelblue', linewidth=2.5,
        label=f'治未病预警模型 (AUC = {auc_prev:.3f})')

# 现症识别模型ROC（使用5折交叉验证out-of-fold预测概率）
fpr_diag, tpr_diag, _ = roc_curve(df['disease'], df['oof_prob_diag'])
auc_diag = auc(fpr_diag, tpr_diag)
ax.plot(fpr_diag, tpr_diag, color='darkorange', linewidth=2.5,
        label=f'现症识别模型 (AUC = {auc_diag:.3f})')

# 对角线
ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='随机猜测 (AUC = 0.500)')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('假阳性率 (False Positive Rate)', fontsize=12)
ax.set_ylabel('真阳性率 (True Positive Rate)', fontsize=12)
ax.set_title('两层风险模型ROC曲线对比（5折交叉验证 out-of-fold）', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('figures/fig2a_roc_comparison.png', dpi=300, bbox_inches='tight')
print("已保存 figures/fig2a_roc_comparison.png")
