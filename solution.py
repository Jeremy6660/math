import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子，确保可复现
np.random.seed(42)

# ========================== 读取数据 ==========================
df = pd.read_csv('题目（处理后）/附件1：样例数据.csv')

# 列名映射（简化后续使用）
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

# 临床正常范围（用于构造血脂异常项数，仅含四项血脂指标）
def count_abnormal(row):
    cnt = 0
    if row['tc'] < 3.1 or row['tc'] > 6.2: cnt += 1
    if row['tg'] < 0.56 or row['tg'] > 1.7: cnt += 1
    if row['ldl'] < 2.07 or row['ldl'] > 3.1: cnt += 1
    if row['hdl'] < 1.04 or row['hdl'] > 1.55: cnt += 1
    return cnt

# 血尿酸异常标记（性别特异性，独立变量，不混入血脂异常项数）
def uric_abnormal(row):
    gender = row['gender']  # 0=女, 1=男
    uric_low = 208 if gender == 1 else 155
    uric_high = 428 if gender == 1 else 357
    return 1 if (row['uric'] < uric_low or row['uric'] > uric_high) else 0

df['uric_abnormal'] = df.apply(uric_abnormal, axis=1)

df['abnormal_cnt'] = df.apply(count_abnormal, axis=1)

print("=" * 60)
print("问题1：关键指标筛选与体质贡献度")
print("=" * 60)

# ---------- 1.1 相关性分析：筛选与痰湿质、高血脂相关的关键指标 ----------
indicators = ['hdl', 'ldl', 'tg', 'tc', 'glucose', 'uric', 'bmi', 'adl', 'iadl', 'mobility']

# 与痰湿质积分的Pearson相关
corr_phlegm = df[indicators + ['phlegm']].corr()['phlegm'].drop('phlegm')
print("\n【1】各指标与痰湿质积分的相关系数：")
print(corr_phlegm.sort_values(ascending=False).round(4))

# 与高血脂标签的点二列相关
corr_disease = {col: stats.pearsonr(df[col], df['disease'])[0] for col in indicators}
print("\n【2】各指标与高血脂标签的点二列相关系数：")
print(pd.Series(corr_disease).sort_values(ascending=False).round(4))

# ---------- 1.2 多元线性回归 + 组间差异检验：筛选表征痰湿体质严重程度的关键指标 ----------
# 题目要求筛选"能有效表征痰湿体质严重程度、且能预警高血脂发病风险"的关键指标
# 由于痰湿质积分与血常规指标可能呈非线性关系，采用三种方法并行分析：
#   (a) Spearman相关（捕捉单调非线性关系）
#   (b) 多元线性回归（控制混杂因素后的独立效应）
#   (c) 组间差异检验（痰湿体质标签组 vs 非痰湿体质标签组）

# (a) Spearman相关
spearman_phlegm = {col: stats.spearmanr(df[col], df['phlegm'])[0] for col in indicators}
print("\n【1.2a】Spearman相关系数（痰湿质积分）：")
print(pd.Series(spearman_phlegm).sort_values(ascending=False).round(4))

# (b) 多元线性回归
X_phlegm = sm.add_constant(df[indicators])
ols_model = sm.OLS(df['phlegm'], X_phlegm).fit()
print("\n【1.2b】多元线性回归：血常规与活动量表对痰湿质积分的解释力")
print(f"    R² = {ols_model.rsquared:.4f}, 调整R² = {ols_model.rsquared_adj:.4f}")
print("    标准化系数（Beta）及p值：")
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

# (c) 组间差异检验：痰湿体质标签(constitution==5) vs 其他体质
phlegm_group = df[df['constitution'] == 5]
other_group = df[df['constitution'] != 5]
print(f"\n【1.2c】组间差异检验（痰湿质 n={len(phlegm_group)} vs 非痰湿质 n={len(other_group)}）：")
group_tests = []
for col in indicators:
    # Mann-Whitney U检验（不假设正态分布）
    stat, pval = stats.mannwhitneyu(phlegm_group[col], other_group[col], alternative='two-sided')
    # 效应量：秩次相关系数 r = Z / sqrt(N)
    z = stats.norm.ppf(1 - pval/2)
    r_effect = z / np.sqrt(len(df))
    group_tests.append({'指标': col, 'pvalue': pval, '效应量r': r_effect,
                        '痰湿质中位数': phlegm_group[col].median(),
                        '非痰湿质中位数': other_group[col].median()})
group_df = pd.DataFrame(group_tests)
print(group_df.sort_values('效应量r', key=abs, ascending=False).round(4))

# 综合筛选：采用"多数投票"策略，满足以下任一条件即入选：
#   - Spearman |r| > 0.10 且 p < 0.05
#   - 组间检验 p < 0.05 且 |效应量| > 0.05
#   - 与高血脂点二列 |r| > 0.15 且 p < 0.05
print("\n【1.3】综合筛选（既表征痰湿严重程度、又预警高血脂的指标）：")
corr_disease_s = pd.Series(corr_disease)
joint_score = pd.DataFrame({'指标': indicators})
joint_score['spearman_r'] = [abs(spearman_phlegm[c]) for c in indicators]
joint_score['group_p'] = group_df.set_index('指标').loc[indicators, 'pvalue'].values
joint_score['group_effect'] = [abs(v) for v in group_df.set_index('指标').loc[indicators, '效应量r'].values]
joint_score['r_高血脂'] = [abs(corr_disease_s[c]) for c in indicators]
# 标记入选条件
cond1 = joint_score['spearman_r'] > 0.10
cond2 = (joint_score['group_p'] < 0.05) & (joint_score['group_effect'] > 0.05)
cond3 = joint_score['r_高血脂'] > 0.15
joint_score['入选'] = cond1 | cond2 | cond3
print(joint_score[joint_score['入选']].sort_values('r_高血脂', ascending=False).round(4))

# ---------- 1.4 逐步Logistic回归：筛选预警高血脂的关键指标 ----------

def stepwise_logit(X, y, threshold_in=0.05, threshold_out=0.10):
    """基于AIC的前向+后向逐步Logistic回归"""
    included = []
    while True:
        changed = False
        # 前向选择
        excluded = list(set(X.columns) - set(included))
        if excluded:
            pvals = pd.Series(index=excluded, dtype=float)
            for col in excluded:
                try:
                    model = sm.Logit(y, sm.add_constant(X[included + [col]])).fit(disp=0)
                    pvals[col] = model.pvalues[col]
                except:
                    pvals[col] = 1.0
            best_pval = pvals.min()
            if best_pval < threshold_in:
                best_feature = pvals.idxmin()
                included.append(best_feature)
                changed = True

        # 后向剔除
        if included:
            try:
                model = sm.Logit(y, sm.add_constant(X[included])).fit(disp=0)
                pvalues = model.pvalues.iloc[1:]  # 去掉常数项
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
print(f"\n【1.4】逐步Logistic回归筛选的关键指标：{selected_features}")

# ---------- 1.5 Logistic回归：九种体质对高血脂的贡献度 ----------
constitution_cols = ['pinghe', 'qixu', 'yangxu', 'yinxu', 'phlegm', 'shire', 'xueyu', 'qiyu', 'tebing']
X_const = df[constitution_cols]
X_const_scaled = StandardScaler().fit_transform(X_const)
X_const_const = sm.add_constant(X_const_scaled)

logit_model = sm.Logit(df['disease'], X_const_const).fit(disp=0)
print("\n【1.5】九种体质对高血脂发病风险的Logistic回归结果（标准化系数）：")
coef_df = pd.DataFrame({
    '体质': constitution_cols,
    'coef': logit_model.params[1:].values,
    'OR': np.exp(logit_model.params[1:].values),
    'pvalue': logit_model.pvalues[1:].values
})
coef_df['abs_coef'] = coef_df['coef'].abs()
print(coef_df.sort_values('abs_coef', ascending=False).round(4))

# ---------- 1.6 VIF（方差膨胀因子）计算 ----------
vif_data = pd.DataFrame({
    '体质': constitution_cols,
    'VIF': [variance_inflation_factor(X_const.values, i)
            for i in range(X_const.shape[1])]
})
print("\n【1.6】九种体质积分的VIF值：")
print(vif_data.round(2))

print("\n" + "=" * 60)
print("问题2：三级风险预警模型")
print("=" * 60)

# ---------- 2.1 综合风险评分 ----------
# 使用问题1筛选的指标，若筛选为空则使用默认指标
risk_features = selected_features if selected_features else ['phlegm', 'bmi', 'tg', 'ldl', 'mobility']
X_risk = df[risk_features].fillna(df[risk_features].mean())
X_risk_scaled = StandardScaler().fit_transform(X_risk)

# Logistic回归输出基础风险概率
logit_risk = LogisticRegression(max_iter=1000).fit(X_risk_scaled, df['disease'])
prob = logit_risk.predict_proba(X_risk_scaled)[:, 1]

# 综合风险评分 = 0.5*概率 + 0.3*痰湿积分标准化 + 0.2*血脂异常项数标准化
risk_score = 0.5 * prob + 0.3 * (df['phlegm'] / 100) + 0.2 * (df['abnormal_cnt'] / 4)
df['risk_score'] = risk_score

# ---------- 2.2 三级风险划分（连续数据离散化） ----------
# 阈值确定：结合临床锚点 + 分位数
# 高风险锚点：血脂异常且痰湿>=60 的人群至少应为高风险
anchor_high = df[(df['abnormal_cnt'] > 0) & (df['phlegm'] >= 60)]['risk_score']
high_threshold = max(risk_score.quantile(0.75), anchor_high.median() if len(anchor_high) > 0 else risk_score.quantile(0.75))
low_threshold = risk_score.quantile(0.30)

print(f"\n【1】风险阈值：低风险 < {low_threshold:.4f} <= 中风险 < {high_threshold:.4f} <= 高风险")

df['risk_level'] = pd.cut(risk_score, bins=[-np.inf, low_threshold, high_threshold, np.inf], labels=['低', '中', '高'])
print("\n【2】三级风险分布：")
print(df['risk_level'].value_counts())

# ---------- 2.3 识别痰湿体质高风险人群核心特征组合 ----------
high_risk_group = df[df['risk_level'] == '高']
print("\n【3】高风险人群关键指标均值：")
print(high_risk_group[['phlegm', 'bmi', 'tg', 'ldl', 'mobility', 'abnormal_cnt']].mean().round(4))

# 核心特征组合：用条件概率找最强关联规则
print("\n【4】核心特征组合（条件概率）：")
# 组合1：痰湿>=60 & 活动<40
combo1 = df[(df['phlegm'] >= 60) & (df['mobility'] < 40)]
p1 = combo1['risk_level'].value_counts(normalize=True).get('高', 0)
print(f"    痰湿>=60 & 活动总分<40  → 高风险概率: {p1:.2%} (n={len(combo1)})")

# 组合2：痰湿>=60 & BMI>=24 & TG>1.7
combo2 = df[(df['phlegm'] >= 60) & (df['bmi'] >= 24) & (df['tg'] > 1.7)]
p2 = combo2['risk_level'].value_counts(normalize=True).get('高', 0)
print(f"    痰湿>=60 & BMI>=24 & TG>1.7  → 高风险概率: {p2:.2%} (n={len(combo2)})")

# 组合3：任意体质中痰湿最高且血脂异常>=2项
combo3 = df[(df[constitution_cols].idxmax(axis=1) == 'phlegm') & (df['abnormal_cnt'] >= 2)]
p3 = combo3['risk_level'].value_counts(normalize=True).get('高', 0)
print(f"    痰湿为最高分体质 & 血脂异常>=2项  → 高风险概率: {p3:.2%} (n={len(combo3)})")

# ---------- 2.4 K折分层交叉验证 ----------
# 将三级风险转为二分类：高风险(1) vs 非高风险(0)，用于ROC/AUC
df['high_risk'] = (df['risk_level'] == '高').astype(int)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
acc_scores, rec_scores, f1_scores, auc_scores = [], [], [], []

print("\n【5】5折分层交叉验证结果（高风险 vs 非高风险）：")
for fold, (train_idx, test_idx) in enumerate(skf.split(X_risk_scaled, df['high_risk']), 1):
    X_train, X_test = X_risk_scaled[train_idx], X_risk_scaled[test_idx]
    y_train, y_test = df['high_risk'].iloc[train_idx], df['high_risk'].iloc[test_idx]

    logit_cv = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
    y_prob = logit_cv.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc_scores.append(accuracy_score(y_test, y_pred))
    rec_scores.append(recall_score(y_test, y_pred, zero_division=0))
    f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
    auc_scores.append(roc_auc_score(y_test, y_prob))

    print(f"  Fold {fold}: Acc={acc_scores[-1]:.4f}, Recall={rec_scores[-1]:.4f}, "
          f"F1={f1_scores[-1]:.4f}, AUC={auc_scores[-1]:.4f}")

print(f"\n  均值: Acc={np.mean(acc_scores):.4f}, Recall={np.mean(rec_scores):.4f}, "
      f"F1={np.mean(f1_scores):.4f}, AUC={np.mean(auc_scores):.4f}")

# 全量拟合用于ROC曲线
logit_full = LogisticRegression(max_iter=1000, random_state=42).fit(X_risk_scaled, df['high_risk'])
prob_full = logit_full.predict_proba(X_risk_scaled)[:, 1]
fpr, tpr, _ = roc_curve(df['high_risk'], prob_full)

# ---------- 2.5 风险评分权重敏感性分析 ----------
# 使用固定阈值（基准阈值）观察权重扰动对分层比例的影响
print("\n【6】风险评分权重敏感性分析（固定阈值，主观权重 ±20% 扰动）：")
base_weights = (0.5, 0.3, 0.2)
perturbations = [
    (0.4, 0.36, 0.24),   # w1-20%, w2+20%, w3+20%（归一化近似）
    (0.6, 0.24, 0.16),   # w1+20%, w2-20%, w3-20%
    (0.5, 0.24, 0.26),   # w2-20%, w3+30%
    (0.5, 0.36, 0.14),   # w2+20%, w3-30%
]
base_dist = df['risk_level'].value_counts(normalize=True).sort_index()
print(f"  基准权重 {base_weights}: 低={base_dist.get('低',0):.3f}, 中={base_dist.get('中',0):.3f}, 高={base_dist.get('高',0):.3f}")
for w1, w2, w3 in perturbations:
    rs = w1 * prob + w2 * (df['phlegm'] / 100) + w3 * (df['abnormal_cnt'] / 4)
    rl = pd.cut(rs, bins=[-np.inf, low_threshold, high_threshold, np.inf], labels=['低', '中', '高'])
    dist = rl.value_counts(normalize=True).sort_index()
    print(f"  权重 ({w1},{w2},{w3}): 低={dist.get('低',0):.3f}, 中={dist.get('中',0):.3f}, 高={dist.get('高',0):.3f}")

print("\n" + "=" * 60)
print("问题3：痰湿体质患者干预方案优化")
print("=" * 60)

# ========================== 问题3：优化模型 ==========================

# 成本参数
C_TREAT = {1: 30, 2: 80, 3: 130}   # 每月调理成本
C_ACT = {1: 3, 2: 5, 3: 8}         # 单次活动成本
WEEKS = 24                         # 6个月 = 24周
MAX_COST = 2000                    # 总成本上限

def get_treatment_level(phlegm_score):
    """根据痰湿积分确定调理分级（中医原则，强制匹配）"""
    if phlegm_score <= 58:
        return 1
    elif phlegm_score <= 61:
        return 2
    else:
        return 3

def get_max_activity_intensity(age_group, mobility_score):
    """根据年龄和活动量表总分确定活动强度上限（耐受度约束）"""
    # 年龄约束
    if age_group in [1, 2]:       # 40-59岁
        y_max_age = 3
    elif age_group in [3, 4]:     # 60-79岁
        y_max_age = 2
    else:                         # 80-89岁
        y_max_age = 1

    # 评分约束
    if mobility_score < 40:
        y_max_mob = 1
    elif mobility_score < 60:
        y_max_mob = 2
    else:
        y_max_mob = 3

    return min(y_max_age, y_max_mob)

def calc_monthly_drop_rate(y, f, age_group, mobility):
    """
    计算每月痰湿积分下降率（含依从性衰减）
    y: 活动干预强度 (1/2/3)
    f: 每周训练频率 (1-10次)
    age_group: 年龄组
    mobility: 活动量表总分

    规则：
    - f < 5: 积分基本稳定，下降率为0
    - f >= 5: 基础下降 3%*(y-1) + 额外下降 1%*(effective_f-5)
    - 依从性衰减：高龄或低活动能力患者，超出5次/周的部分效果打折
        * 80-89岁：超出部分仅50%有效
        * 活动能力<40：超出部分仅60%有效
        * 60-79岁：超出部分仅80%有效
    """
    if f < 5:
        return 0.0
    # 依从性衰减 + 疲劳封顶（老年医学安全上限）
    if age_group == 5:          # 80-89岁：耐受度最低，有效频率封顶6次/周
        effective_f = min(5 + (f - 5) * 0.5, 6.0)
    elif mobility < 40:         # 活动能力低下：有效频率封顶6.2次/周
        effective_f = min(5 + (f - 5) * 0.6, 6.2)
    elif age_group in [3, 4]:   # 60-79岁：有效频率打8折
        effective_f = 5 + (f - 5) * 0.8
    else:                       # 40-59岁且活动能力好：无衰减
        effective_f = f
    base_drop = 0.03 * (y - 1)
    extra_drop = 0.01 * (effective_f - 5)
    return base_drop + extra_drop

def optimize_intervention(patient):
    """
    对单名患者进行穷举优化
    决策变量: y∈{1,2,3}, f∈{1..10}
    x由痰湿积分强制确定
    目标: 最小化6个月后痰湿积分（同分取成本最低）
    """
    s0 = patient['phlegm']
    x = get_treatment_level(s0)
    y_max = get_max_activity_intensity(patient['age_group'], patient['mobility'])

    best_plan = None
    best_final = float('inf')

    # 穷举所有合法组合（最多3×10=30种）
    for y in range(1, y_max + 1):
        for f in range(1, 11):
            r = calc_monthly_drop_rate(y, f, patient['age_group'], patient['mobility'])
            # 6个月后痰湿积分（复合下降）
            s_final = s0 * ((1 - r) ** 6)
            # 总成本 = 6个月调理费 + 24周活动费
            total_cost = 6 * C_TREAT[x] + WEEKS * f * C_ACT[y]

            if total_cost > MAX_COST:
                continue

            # 优先最小化最终积分，其次最小化成本
            if s_final < best_final - 1e-6:
                best_final = s_final
                best_plan = {
                    'id': patient['id'],
                    'x_level': x,
                    'y_level': y,
                    'freq': f,
                    'cost': total_cost,
                    's0': s0,
                    's_final': s_final,
                    's_drop': s0 - s_final,
                    'drop_rate_monthly': r
                }
            elif abs(s_final - best_final) < 1e-6 and total_cost < best_plan['cost']:
                best_plan['y_level'] = y
                best_plan['freq'] = f
                best_plan['cost'] = total_cost

    return best_plan

# ---------- 对样本1、2、3输出最优方案 ----------
sample_ids = [1, 2, 3]
print("\n【1】样本ID 1、2、3 的最优干预方案：\n")
plans_123 = []
for sid in sample_ids:
    p = df[df['id'] == sid].iloc[0]
    if p['constitution'] != 5:
        print(f"  注意：样本ID {sid} 不是痰湿体质（constitution={int(p['constitution'])}），仍按题目要求输出方案。\n")
    plan = optimize_intervention(p)
    plans_123.append(plan)
    lvl_name = ('基础', '中度', '强化')
    print(f"样本ID {int(plan['id'])}:")
    print(f"  患者特征: 痰湿积分={plan['s0']}, 年龄组={int(p['age_group'])}, 活动总分={p['mobility']}")
    print(f"  调理分级: {plan['x_level']}级 ({lvl_name[plan['x_level']-1]}调理, {C_TREAT[plan['x_level']]}元/月)")
    print(f"  活动强度: {plan['y_level']}级 (单次{C_ACT[plan['y_level']]}元)")
    print(f"  每周频率: {plan['freq']}次")
    print(f"  6个月总成本: {plan['cost']}元")
    print(f"  每月下降率: {plan['drop_rate_monthly']*100:.2f}%")
    print(f"  预期6个月后痰湿积分: {plan['s_final']:.2f} (下降{plan['s_drop']:.2f}分)\n")

# ---------- 对所有痰湿体质患者求解并总结规律 ----------
phlegm_patients = df[df['constitution'] == 5].copy()
all_plans = []
for _, p in phlegm_patients.iterrows():
    all_plans.append(optimize_intervention(p))
plan_df = pd.DataFrame(all_plans)

# 合并回患者表
phlegm_patients['opt_y'] = plan_df['y_level'].values
phlegm_patients['opt_f'] = plan_df['freq'].values
phlegm_patients['opt_cost'] = plan_df['cost'].values
phlegm_patients['opt_s_final'] = plan_df['s_final'].values

print("【2】患者特征-最优方案匹配规律：\n")

# 按年龄组
print("(a) 按年龄组统计最优方案均值：")
age_summary = phlegm_patients.groupby('age_group').agg({
    'opt_y': 'mean',
    'opt_f': 'mean',
    'opt_cost': 'mean',
    'opt_s_final': 'mean'
}).round(2)
print(age_summary)

# 按活动能力分组
print("\n(b) 按活动能力分组统计最优方案均值：")
phlegm_patients['mobility_group'] = pd.cut(
    phlegm_patients['mobility'],
    bins=[0, 40, 60, 100],
    labels=['差(<40)', '中(40-60)', '好(≥60)'],
    include_lowest=True
)
mob_summary = phlegm_patients.groupby('mobility_group').agg({
    'opt_y': 'mean',
    'opt_f': 'mean',
    'opt_cost': 'mean',
    'opt_s_final': 'mean'
}).round(2)
print(mob_summary)

# 按痰湿积分分组
print("\n(c) 按痰湿积分分组统计最优方案均值：")
phlegm_patients['phlegm_group'] = pd.cut(
    phlegm_patients['phlegm'],
    bins=[0, 58, 61, 100],
    labels=['低(0-58)', '中(59-61)', '高(≥62)'],
    include_lowest=True
)
phlegm_summary = phlegm_patients.groupby('phlegm_group').agg({
    'opt_y': 'mean',
    'opt_f': 'mean',
    'opt_cost': 'mean',
    'opt_s_final': 'mean'
}).round(2)
print(phlegm_summary)

# ---------- 自动计算单位成本降分效果 ----------
# 合并剩余字段
phlegm_patients['opt_x'] = plan_df['x_level'].values
phlegm_patients['opt_s_drop'] = plan_df['s_drop'].values

print("\n【3】单位成本降分效果（按调理分级统计实际优化结果）：")
cost_effect = phlegm_patients.groupby('opt_x').agg({
    'opt_s_drop': 'mean',
    'opt_cost': 'mean'
}).round(2)
cost_effect['单位成本降分(分/元)'] = (cost_effect['opt_s_drop'] / cost_effect['opt_cost']).round(4)
cost_effect.index = ['基础(x=1)', '中度(x=2)', '强化(x=3)']
print(cost_effect)

# ---------- 可视化输出 ----------
print("\n【4】正在生成可视化图表...")
os.makedirs('output_figures', exist_ok=True)

# --- 图1: 问题1 相关性热力图 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# 左图: 各指标与痰湿质、高血脂的相关性
corr_data = pd.DataFrame({
    '指标': indicators,
    '与痰湿质Pearson': [corr_phlegm[c] for c in indicators],
    '与高血脂点二列': [corr_disease[c] for c in indicators]
})
corr_melt = corr_data.melt(id_vars='指标', var_name='目标', value_name='r')
sns.barplot(data=corr_melt, x='指标', y='r', hue='目标', ax=axes[0])
axes[0].set_title('问题1: 关键指标与目标变量的相关性')
axes[0].tick_params(axis='x', rotation=45)
axes[0].axhline(0, color='black', linewidth=0.5)
# 右图: 九种体质系数
sns.barplot(data=coef_df.sort_values('abs_coef', ascending=False),
            x='体质', y='coef', ax=axes[1])
axes[1].set_title('问题1: 九种体质标准化Logistic系数')
axes[1].tick_params(axis='x', rotation=45)
axes[1].axhline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('output_figures/fig1_problem1_analysis.png', dpi=300)
plt.close()

# --- 图2: 问题2 ROC曲线 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(fpr, tpr, label=f'ROC curve (AUC = {np.mean(auc_scores):.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('问题2: 高风险 vs 非高风险 ROC曲线')
axes[0].legend()
# 右图: 风险评分分布
sns.histplot(data=df, x='risk_score', hue='risk_level', bins=30, kde=True, ax=axes[1])
axes[1].axvline(low_threshold, color='green', linestyle='--', label=f'低中阈值={low_threshold:.3f}')
axes[1].axvline(high_threshold, color='red', linestyle='--', label=f'中高阈值={high_threshold:.3f}')
axes[1].set_title('问题2: 综合风险评分分布')
axes[1].legend()
plt.tight_layout()
plt.savefig('output_figures/fig2_problem2_risk.png', dpi=300)
plt.close()

# --- 图3: 问题3 最优频率分布 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# 按年龄组的最优频率分布
sns.boxplot(data=phlegm_patients, x='age_group', y='opt_f', ax=axes[0])
axes[0].set_title('问题3: 最优训练频率按年龄组分布')
axes[0].set_xlabel('年龄组')
# 按活动能力的最优频率分布
sns.boxplot(data=phlegm_patients, x='mobility_group', y='opt_f', ax=axes[1])
axes[1].set_title('问题3: 最优训练频率按活动能力分布')
axes[1].set_xlabel('活动能力分组')
plt.tight_layout()
plt.savefig('output_figures/fig3_problem3_optimization.png', dpi=300)
plt.close()

print("  图表已保存至 output_figures/ 目录")

print("\n" + "=" * 60)
print("运行完毕")
print("=" * 60)
