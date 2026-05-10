"""
图3(a)：样本ID 1、2、3的6个月干预轨迹
（痰湿积分动态变化 + 调理分级标注）
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'Arial Unicode MS', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

traj1 = pd.read_csv('output/p3_id1_trajectory.csv')
traj2 = pd.read_csv('output/p3_id2_trajectory.csv')
traj3 = pd.read_csv('output/p3_id3_trajectory.csv')

fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)

month_labels = ['初始'] + [f'{m}月' for m in range(1, 7)]
colors = ['#e74c3c', '#3498db', '#2ecc71']
labels = ['ID 1 (50~59岁, M=38)', 'ID 2 (40~49岁, M=40)', 'ID 3 (40~49岁, M=63)']
trajs = [traj1, traj2, traj3]

for traj, label, color in zip(trajs, labels, colors):
    scores = [traj['月初积分'].iloc[0]] + traj['月初积分'].iloc[1:6].tolist() + [traj['月初积分'].iloc[-1]]
    months = list(range(7))
    ax.plot(months, scores, marker='o', markersize=8, linewidth=2.5, label=label, color=color)

    # 标注调理分级变化点
    for i in range(len(traj)-2):
        curr_x = traj['调理分级'].iloc[i]
        next_x = traj['调理分级'].iloc[i+1]
        if pd.isna(curr_x) or pd.isna(next_x) or curr_x == '--' or next_x == '--':
            continue
        if int(curr_x) != int(next_x):
            ax.axvline(i+1 + 0.5, color=color, linestyle='--', alpha=0.4, linewidth=1.5)
            ax.annotate(f'降级→{int(next_x)}级', xy=(i+1 + 0.5, scores[i+1]),
                       xytext=(5, 10), textcoords='offset points',
                       fontsize=8, color=color, fontweight='bold')

# 阈值线
ax.axhline(62, color='purple', linestyle=':', linewidth=1.5, alpha=0.7, label='强化调理阈值 (62分)')
ax.axhline(58.5, color='brown', linestyle=':', linewidth=1.5, alpha=0.7, label='中度调理阈值 (58.5分)')

ax.set_xlabel('干预月份', fontsize=12)
ax.set_ylabel('痰湿积分', fontsize=12)
ax.set_title('样本ID 1/2/3 的6个月干预轨迹（动态逐月规划）', fontsize=14, fontweight='bold')
ax.set_xticks(range(7))
ax.set_xticklabels(month_labels)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, linestyle='--', alpha=0.4)
ax.set_ylim(25, 70)

plt.tight_layout()
plt.savefig('figures/fig3a_intervention_trajectory.png', dpi=300, bbox_inches='tight')
print("已保存 figures/fig3a_intervention_trajectory.png")
