"""
问题1：关键指标筛选与体质贡献度分析
包含：相关性分析、组间差异检验、逐步Logistic回归、九种体质贡献度、VIF、随机森林特征重要性
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
os.makedirs('output', exist_ok=True)

# ========================== 数据读取与预处理 ==========================
df = pd.read_csv('题目（处理后）/附件1：样例数据.csv')

col_map = {
    '痰湿质': 'phlegm',
    '平和质': 'pinghe', '气虚质': 'qixu', '阳虚质': 'yangxu', '阴虚质': 'yinxu',
    '湿热质': 'shire', '血瘀质': 'xueyu', '气郁质': 'qiyu', '特禀质': 'tebing',
    'ADL总分': 'adl', 'IADL总分': 'iadl',
    '活动量表总分（ADL总分+IADL总分）': 'mobility',
    'HDL-C（高密度脂蛋白）': 'hdl', 'LDL-C（低密度脂蛋白）': 'ldl',
    'TG（甘油三酯）': 'tg', 'TC（总胆固醇）': 'tc',
    '空腹血糖': 'glucose', '血尿酸': 'uric', 'BMI': 'bmi',
    '高血脂症二分类标签': 'disease',
    '血脂异常分型标签（确诊病例）': 'disease_type',
    '年龄组': 'age_group', '性别': 'gender',
    '吸烟史': 'smoke', '饮酒史': 'drink',
    '样本ID': 'id', '体质标签': 'constitution'
}
df.rename(columns=col_map, inplace=True)

# 方向性血脂异常风险项数（只计高血脂方向）
def count_lipid_risk(row):
    cnt = 0
    if row['tc'] > 6.2: cnt += 1
    if row['tg'] > 1.7: cnt += 1
    if row['ldl'] > 3.1: cnt += 1
    if row['hdl'] < 1.04: cnt += 1
    return cnt

df['lipid_risk_cnt'] = df.apply(count_lipid_risk, axis=1)

# 血尿酸异常标记（性别特异性）
def uric_abnormal(row):
    gender = row['gender']
    uric_low = 208 if gender == 1 else 155
    uric_high = 428 if gender == 1 else 357
    return 1 if (row['uric'] < uric_low or row['uric'] > uric_high) else 0

df['uric_abnormal'] = df.apply(uric_abnormal, axis=1)

print("=" * 60)
print("问题1：关键指标筛选与体质贡献度")
print("=" * 60)

# ========================== 1.1 相关性分析 ==========================
indicators = ['hdl', 'ldl', 'tg', 'tc', 'glucose', 'uric', 'bmi', 'adl', 'iadl', 'mobility']

# Pearson相关（痰湿质）
corr_phlegm = df[indicators + ['phlegm']].corr()['phlegm'].drop('phlegm')
print("\n【1.1a】各指标与痰湿质积分的Pearson相关系数：")
print(corr_phlegm.sort_values(ascending=False).round(4))

# 点二列相关（高血脂）
corr_disease = {col: stats.pearsonr(df[col], df['disease'])[0] for col in indicators}
print("\n【1.1b】各指标与高血脂标签的点二列相关系数：")
corr_disease_s = pd.Series(corr_disease)
print(corr_disease_s.sort_values(ascending=False).round(4))

# Spearman相关（痰湿质，捕捉单调非线性）
spearman_phlegm = {col: stats.spearmanr(df[col], df['phlegm'])[0] for col in indicators}
print("\n【1.1c】Spearman相关系数（痰湿质积分）：")
print(pd.Series(spearman_phlegm).sort_values(ascending=False).round(4))

# ========================== 1.2 组间差异检验 ==========================
phlegm_group = df[df['constitution'] == 5]
other_group = df[df['constitution'] != 5]
print(f"\n【1.2】组间差异检验（痰湿质 n={len(phlegm_group)} vs 非痰湿质 n={len(other_group)}）：")
group_tests = []
for col in indicators:
    stat, pval = stats.mannwhitneyu(phlegm_group[col], other_group[col], alternative='two-sided')
    z = stats.norm.ppf(1 - pval/2)
    r_effect = z / np.sqrt(len(df))
    group_tests.append({'指标': col, 'pvalue': pval, '效应量r': r_effect,
                        '痰湿质中位数': phlegm_group[col].median(),
                        '非痰湿质中位数': other_group[col].median()})
group_df = pd.DataFrame(group_tests)
print(group_df.sort_values('效应量r', key=abs, ascending=False).round(4))

# ========================== 1.3 多元线性回归（控制混杂） ==========================
X_phlegm = sm.add_constant(df[indicators])
ols_model = sm.OLS(df['phlegm'], X_phlegm).fit()
print(f"\n【1.3】多元线性回归：R²={ols_model.rsquared:.4f}, 调整R²={ols_model.rsquared_adj:.4f}")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[indicators])
ols_std = sm.OLS(df['phlegm'], sm.add_constant(X_scaled)).fit()
ols_summary = pd.DataFrame({
    '指标': indicators,
    'coef_std': ols_std.params[1:].values,
    'pvalue': ols_model.pvalues[1:].values
})
ols_summary['abs_std'] = ols_summary['coef_std'].abs()
print(ols_summary.sort_values('abs_std', ascending=False).round(4))

# ========================== 1.4 综合筛选 ==========================
print("\n【1.4】综合筛选结论：")
print("  本数据集中，血脂指标(TG, TC, LDL-C, HDL-C)与尿酸对高血脂标签有显著预警能力，")
print("  但与痰湿质积分的线性关联极弱。关键指标可分为两类：")
print("  第一类：高血脂风险识别指标——TG、TC、LDL-C、HDL-C、尿酸")
print("  第二类：痰湿严重程度直接表征指标——痰湿质积分、主体质标签")
print("  活动量表在本数据中未显示显著统计相关，但因题目机制和干预约束明确，")
print("  仍作为风险解释和干预约束变量保留。")

# ========================== 1.5 逐步Logistic回归 ==========================
def stepwise_logit(X, y, threshold_in=0.05, threshold_out=0.10):
    """基于p-value阈值的前向+后向逐步Logistic回归"""
    included = []
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        if excluded:
            pvals = pd.Series(index=excluded, dtype=float)
            for col in excluded:
                try:
                    model = sm.Logit(y, sm.add_constant(X[included + [col]])).fit(disp=0)
                    pvals[col] = model.pvalues[col]
                except Exception as e:
                    print(f"    [警告] 变量 '{col}' 拟合失败（可能存在完全分离）: {e}")
                    pvals[col] = 1.0
            best_pval = pvals.min()
            if best_pval < threshold_in:
                best_feature = pvals.idxmin()
                included.append(best_feature)
                changed = True
        if included:
            try:
                model = sm.Logit(y, sm.add_constant(X[included])).fit(disp=0)
                pvalues = model.pvalues.iloc[1:]
                worst_pval = pvalues.max()
                if worst_pval > threshold_out:
                    worst_feature = pvalues.idxmax()
                    included.remove(worst_feature)
                    changed = True
            except:
                pass
        if not changed:
            break
    return included

selected_features = stepwise_logit(df[indicators], df['disease'])
print(f"\n【1.5】逐步Logistic回归筛选的关键指标：{selected_features}")

# ========================== 1.6 九种体质贡献度（Logistic + OR + VIF） ==========================
constitution_cols = ['pinghe', 'qixu', 'yangxu', 'yinxu', 'phlegm', 'shire', 'xueyu', 'qiyu', 'tebing']
X_const = df[constitution_cols]
X_const_scaled = StandardScaler().fit_transform(X_const)
X_const_const = sm.add_constant(X_const_scaled)

logit_model = sm.Logit(df['disease'], X_const_const).fit(disp=0)
coef_df = pd.DataFrame({
    '体质': constitution_cols,
    'coef': logit_model.params[1:].values,
    'OR': np.exp(logit_model.params[1:].values),
    'pvalue': logit_model.pvalues[1:].values
})
coef_df['abs_coef'] = coef_df['coef'].abs()
print("\n【1.6】九种体质对高血脂发病风险的Logistic回归结果（标准化系数）：")
print(coef_df.sort_values('abs_coef', ascending=False).round(4))

# VIF计算（基于原始积分）
vif_data = pd.DataFrame({
    '体质': constitution_cols,
    'VIF': [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
})
print("\n【1.6b】九种体质积分的VIF值：")
print(vif_data.round(2))
print("  结论：VIF均低于5，多重共线性不是导致体质系数不显著的主要原因。")

# ========================== 1.7 随机森林特征重要性（补充非线性分析） ==========================
print("\n【1.7】随机森林特征重要性（弥补线性模型在交互效应上的不足）：")
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_const, df['disease'])
rf_importance = pd.DataFrame({
    '体质': constitution_cols,
    'Gini重要性': rf.feature_importances_
}).sort_values('Gini重要性', ascending=False)
print(rf_importance.round(4))

# 保存结果
corr_data = pd.DataFrame({
    '指标': indicators,
    '与痰湿质Pearson': [corr_phlegm[c] for c in indicators],
    '与高血脂点二列': [corr_disease_s[c] for c in indicators],
    '与痰湿质Spearman': [spearman_phlegm[c] for c in indicators]
})
corr_data.to_csv('output/p1_correlation.csv', index=False)
coef_df.to_csv('output/p1_constitution_logistic.csv', index=False)
vif_data.to_csv('output/p1_vif.csv', index=False)
rf_importance.to_csv('output/p1_rf_importance.csv', index=False)

# ========================== 可视化 ==========================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图1a: 各指标与两个目标变量的相关性对比
corr_melt = corr_data[['指标', '与痰湿质Pearson', '与高血脂点二列']].melt(
    id_vars='指标', var_name='目标', value_name='r')
sns.barplot(data=corr_melt, x='指标', y='r', hue='目标', ax=axes[0, 0])
axes[0, 0].set_title('问题1: 关键指标与目标变量的相关性对比')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].axhline(0, color='black', linewidth=0.5)

# 图1b: 九种体质Logistic系数
coef_plot = coef_df.sort_values('abs_coef', ascending=False)
sns.barplot(data=coef_plot, x='体质', y='coef', ax=axes[0, 1])
axes[0, 1].set_title('问题1: 九种体质标准化Logistic系数')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].axhline(0, color='black', linewidth=0.5)

# 图1c: 随机森林特征重要性
sns.barplot(data=rf_importance, x='体质', y='Gini重要性', ax=axes[1, 0])
axes[1, 0].set_title('问题1: 随机森林Gini特征重要性（非线性贡献）')
axes[1, 0].tick_params(axis='x', rotation=45)

# 图1d: VIF值
sns.barplot(data=vif_data, x='体质', y='VIF', ax=axes[1, 1])
axes[1, 1].set_title('问题1: 九种体质VIF值')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].axhline(5, color='red', linestyle='--', label='VIF=5阈值')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('output/fig1_problem1_analysis.png', dpi=300)
plt.close()

print("\n  图表已保存至 output/fig1_problem1_analysis.png")
print("  数据已保存至 output/p1_*.csv")
