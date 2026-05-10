"""
制图专用程序：生成论文所需的最终高质量图表
读取 problem1/2/3 的输出数据，绘制综合可视化图
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font='Arial Unicode MS')
sns.set_style("whitegrid")

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
import os

os.makedirs('output', exist_ok=True)
plt.rcParams['figure.dpi'] = 300

print("=" * 60)
print("制图专用程序：生成论文最终图表")
print("=" * 60)

# ========================== 图1: 问题1综合图 ==========================
print("\n【图1】问题1：特征筛选与体质贡献度综合图")

fig = plt.figure(figsize=(18, 10))
# 使用GridSpec进行复杂布局
from matplotlib.gridspec import GridSpec
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# 1a: 指标相关性对比
corr_data = pd.read_csv('output/p1_correlation.csv')
ax1 = fig.add_subplot(gs[0, 0])
corr_melt = corr_data[['指标', '与痰湿质Pearson', '与高血脂点二列']].melt(
    id_vars='指标', var_name='目标', value_name='r')
sns.barplot(data=corr_melt, x='指标', y='r', hue='目标', ax=ax1, palette='Set2')
ax1.set_title('(a) 关键指标与目标变量的相关性')
ax1.tick_params(axis='x', rotation=45)
ax1.axhline(0, color='black', linewidth=0.5)

# 1b: 九种体质Logistic系数
coef_df = pd.read_csv('output/p1_constitution_logistic.csv')
ax2 = fig.add_subplot(gs[0, 1])
coef_plot = coef_df.sort_values('abs_coef', ascending=False)
colors = ['red' if p >= 0.05 else 'steelblue' for p in coef_plot['pvalue']]
sns.barplot(data=coef_plot, x='体质', y='coef', palette=colors, ax=ax2)
ax2.set_title('(b) 九种体质标准化Logistic系数')
ax2.tick_params(axis='x', rotation=45)
ax2.axhline(0, color='black', linewidth=0.5)
# 添加显著性说明
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', label='p≥0.05（不显著）'),
                   Patch(facecolor='steelblue', label='p<0.05')]
ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)

# 1c: 随机森林特征重要性
rf_imp = pd.read_csv('output/p1_rf_importance.csv')
ax3 = fig.add_subplot(gs[0, 2])
sns.barplot(data=rf_imp, x='体质', y='Gini重要性', ax=ax3, palette='viridis')
ax3.set_title('(c) 随机森林Gini特征重要性')
ax3.tick_params(axis='x', rotation=45)

# 1d: VIF诊断
vif_data = pd.read_csv('output/p1_vif.csv')
ax4 = fig.add_subplot(gs[1, 0])
sns.barplot(data=vif_data, x='体质', y='VIF', ax=ax4, palette='coolwarm')
ax4.set_title('(d) 九种体质VIF值')
ax4.tick_params(axis='x', rotation=45)
ax4.axhline(5, color='red', linestyle='--', linewidth=1.5, label='VIF=5阈值')
ax4.legend()

# 1e: 组间差异检验效应量（从problem1输出中读取需要的数据，这里简化展示）
ax5 = fig.add_subplot(gs[1, 1:])
ax5.text(0.5, 0.5, '问题1结论：\n\n'
         '1. 血脂指标(TG/TC/LDL-C/HDL-C)与尿酸对高血脂标签有显著预警能力\n'
         '   但与痰湿质积分的线性关联极弱。\n\n'
         '2. 关键指标分为两类：\n'
         '   · 高血脂风险识别指标：TG、TC、LDL-C、HDL-C、尿酸\n'
         '   · 痰湿严重程度直接表征指标：痰湿质积分、主体质标签\n\n'
         '3. 九种体质标准化系数均未达显著水平(p≥0.12)，\n'
         '   随机森林显示痰湿质Gini重要性相对较高，提示非线性交互效应存在。\n\n'
         '4. VIF均低于5，多重共线性不是体质系数不显著的主因。',
         transform=ax5.transAxes, fontsize=11, verticalalignment='center',
         horizontalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.axis('off')
ax5.set_title('(e) 问题1核心结论')

plt.savefig('output/fig1_final_problem1.png', dpi=300, bbox_inches='tight')
plt.close()
print("  已保存 output/fig1_final_problem1.png")

# ========================== 图2: 问题2综合图 ==========================
print("\n【图2】问题2：风险预警模型综合图")

fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# 2a: 风险评分分布
result_df = pd.read_csv('output/p2_risk_scores.csv')
ax1 = fig.add_subplot(gs[0, 0])
sns.histplot(data=result_df, x='risk_score', hue='risk_level', bins=30, kde=True, ax=ax1, palette='RdYlGn_r')
q30 = result_df['risk_score'].quantile(0.30)
q90 = result_df['risk_score'].quantile(0.90)
ax1.axvline(q30, color='green', linestyle='--', linewidth=1.5, label=f'低中阈值={q30:.3f}')
ax1.axvline(q90, color='red', linestyle='--', linewidth=1.5, label=f'中高阈值={q90:.3f}')
ax1.set_title('(a) 综合风险评分分布')
ax1.legend(fontsize=8)

# 2b: 两层模型ROC对比（使用out-of-fold预测概率）
ax2 = fig.add_subplot(gs[0, 1])
from sklearn.metrics import roc_curve
fpr_prev, tpr_prev, _ = roc_curve(result_df['disease'], result_df['oof_prob_prev'])
auc_prev = np.trapezoid(tpr_prev, fpr_prev)
ax2.plot(fpr_prev, tpr_prev, label=f'未病预警 AUC={auc_prev:.3f}', linewidth=2, color='steelblue')
ax2.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
ax2.set_xlabel('False Positive Rate')
ax2.set_ylabel('True Positive Rate')
ax2.set_title('(b) 未病预警模型 out-of-fold ROC 曲线')
ax2.legend()

# 2c: 高风险人群年龄分布
ax3 = fig.add_subplot(gs[0, 2])
high_risk = result_df[result_df['risk_level'] == '高']
age_dist = high_risk['constitution'].value_counts()  # 这里需要重新读取年龄
# 由于result_df没有年龄，我们用原始数据
df_raw = pd.read_csv('题目（处理后）/附件1：样例数据.csv')
df_raw['risk_level'] = result_df['risk_level'].values
high_risk_full = df_raw[df_raw['risk_level'] == '高']
age_dist = high_risk_full['年龄组'].value_counts().sort_index()
age_labels = {1: '40-49', 2: '50-59', 3: '60-69', 4: '70-79', 5: '80-89'}
age_dist_df = pd.DataFrame({'年龄组': [age_labels[i] for i in age_dist.index], '人数': age_dist.values})
sns.barplot(data=age_dist_df, x='年龄组', y='人数', ax=ax3, palette='Blues_d')
ax3.set_title(f'(c) 高风险人群年龄分布 (n={len(high_risk)})')

# 2d: 熵权法权重
ax4 = fig.add_subplot(gs[1, 0])
entropy_df = pd.read_csv('output/p2_entropy_weights.csv')
entropy_melt = entropy_df.melt(id_vars='指标', var_name='方法', value_name='权重')
sns.barplot(data=entropy_melt, x='指标', y='权重', hue='方法', ax=ax4, palette='Set1')
ax4.set_title('(d) 熵权法 vs 组合赋权')
ax4.tick_params(axis='x', rotation=15)

# 2e: 核心特征组合条件概率
ax5 = fig.add_subplot(gs[1, 1:])
combo_data = pd.read_csv('output/p2_feature_combos.csv')
combo_data['特征组合'] = ['痰湿≥60\n&活动<40', '痰湿≥60\n&BMI≥24\n&TG>1.7', '痰湿为最高体质\n&血脂异常≥2项']
bars = sns.barplot(data=combo_data, x='特征组合', y='高风险命中率', ax=ax5, palette='Reds')
ax5.set_title('(e) 核心特征组合模型高风险分层命中率')
ax5.set_ylim(0, 1)
for i, (p, n) in enumerate(zip(combo_data['高风险命中率'], combo_data['覆盖人数'])):
    ax5.text(i, p + 0.02, f'{p:.1%}\n(n={n})', ha='center', fontsize=9)

plt.savefig('output/fig2_final_problem2.png', dpi=300, bbox_inches='tight')
plt.close()
print("  已保存 output/fig2_final_problem2.png")

# ========================== 图3: 问题3综合图 ==========================
print("\n【图3】问题3：干预优化综合图")

fig = plt.figure(figsize=(18, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

# 读取样本轨迹
traj1 = pd.read_csv('output/p3_id1_trajectory.csv')
traj2 = pd.read_csv('output/p3_id2_trajectory.csv')
traj3 = pd.read_csv('output/p3_id3_trajectory.csv')

# 3a: 三个样本的积分变化轨迹
ax1 = fig.add_subplot(gs[0, :2])
for traj, label, color in zip([traj1, traj2, traj3],
                               ['ID 1', 'ID 2', 'ID 3'],
                               ['#e74c3c', '#3498db', '#2ecc71']):
    months = list(range(len(traj)))
    ax1.plot(months, traj['月初积分'], marker='o', label=label, color=color, linewidth=2, markersize=8)
    # 标注调理分级变化点
    for i in range(len(traj)-1):
        if traj['调理分级'].iloc[i] != traj['调理分级'].iloc[i+1] and traj['调理分级'].iloc[i+1] != '--':
            ax1.axvline(i+0.5, color=color, linestyle='--', alpha=0.3)
ax1.set_xlabel('月份')
ax1.set_ylabel('痰湿积分')
ax1.set_title('(a) 样本ID 1/2/3 干预历程（痰湿积分动态变化）')
ax1.set_xticks(range(7))
ax1.set_xticklabels(['初始'] + [f'{m}月' for m in range(1, 7)])
ax1.legend()
ax1.grid(True, alpha=0.3)

# 3b: 月度成本堆叠图
ax2 = fig.add_subplot(gs[0, 2])
cost_matrix = np.array([traj1['月成本(元)'].iloc[:6].values,
                        traj2['月成本(元)'].iloc[:6].values,
                        traj3['月成本(元)'].iloc[:6].values])
x_months = np.arange(6) + 1
width = 0.25
for i, (costs, label, color) in enumerate(zip(cost_matrix, ['ID1', 'ID2', 'ID3'],
                                               ['#e74c3c', '#3498db', '#2ecc71'])):
    ax2.bar(x_months + (i-1)*width, costs, width, label=label, color=color, alpha=0.8)
ax2.set_xlabel('月份')
ax2.set_ylabel('月度成本（元）')
ax2.set_title('(b) 月度成本对比')
ax2.set_xticks(x_months)
ax2.legend()

# 3c: 278人最优频率分布
ax3 = fig.add_subplot(gs[1, 0])
p3_results = pd.read_csv('output/p3_phlegm_patients_results.csv')
sns.boxplot(data=p3_results, x='age_group', y='opt_f', ax=ax3, palette='Set3')
ax3.set_title('(c) 最优频率按年龄组分布（n=278）')
ax3.set_xlabel('年龄组')
ax3.set_ylabel('最优频率（次/周）')

# 3d: 成本-效果散点图
ax4 = fig.add_subplot(gs[1, 1])
scatter = ax4.scatter(p3_results['opt_cost'], p3_results['opt_s_drop'],
                      c=p3_results['phlegm'], cmap='RdYlGn_r', alpha=0.6, s=30)
ax4.set_xlabel('总成本（元）')
ax4.set_ylabel('痰湿积分下降（分）')
ax4.set_title('(d) 成本-效果散点图（颜色=初始积分）')
plt.colorbar(scatter, ax=ax4, label='初始痰湿积分')

# 3e: 敏感性分析
ax5 = fig.add_subplot(gs[1, 2])
sens_df = pd.read_csv('output/p3_sensitivity_analysis.csv')
sens_summary = sens_df.groupby('config').agg({
    'avg_cost_278': 'first',
    'avg_drop_278': 'first'
}).reindex(['保守方案', '基准方案', '乐观方案'])
x_pos = np.arange(len(sens_summary))
ax5_twin = ax5.twinx()
bars = ax5.bar(x_pos - 0.2, sens_summary['avg_cost_278'], 0.4, label='平均成本', color='steelblue', alpha=0.8)
line = ax5_twin.plot(x_pos + 0.2, sens_summary['avg_drop_278'], 'ro-', label='平均降分', linewidth=2, markersize=8)
ax5.set_xlabel('依从性衰减假设')
ax5.set_ylabel('平均成本（元）', color='steelblue')
ax5_twin.set_ylabel('平均降分（分）', color='red')
ax5.set_title('(e) 依从性衰减敏感性分析')
ax5.set_xticks(x_pos)
ax5.set_xticklabels(sens_summary.index, rotation=15)
ax5.legend(loc='upper left')
ax5_twin.legend(loc='upper right')

plt.savefig('output/fig3_final_problem3.png', dpi=300, bbox_inches='tight')
plt.close()
print("  已保存 output/fig3_final_problem3.png")

print("\n" + "=" * 60)
print("全部图表生成完毕")
print("=" * 60)
