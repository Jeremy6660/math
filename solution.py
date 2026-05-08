import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ========================== 读取数据 ==========================
df = pd.read_csv('/Users/xuyunze/Desktop/数学建模/附件1：样例数据.csv')

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

# 临床正常范围（用于构造血脂异常项数）
def count_abnormal(row):
    cnt = 0
    if row['tc'] < 3.1 or row['tc'] > 6.2: cnt += 1
    if row['tg'] < 0.56 or row['tg'] > 1.7: cnt += 1
    if row['ldl'] < 2.07 or row['ldl'] > 3.1: cnt += 1
    if row['hdl'] < 1.04 or row['hdl'] > 1.55: cnt += 1
    return cnt

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

# ---------- 1.2 逐步Logistic回归：筛选预警高血脂的关键指标 ----------

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
print(f"\n【3】逐步Logistic回归筛选的关键指标：{selected_features}")

# ---------- 1.3 Logistic回归：九种体质对高血脂的贡献度 ----------
constitution_cols = ['pinghe', 'qixu', 'yangxu', 'yinxu', 'phlegm', 'shire', 'xueyu', 'qiyu', 'tebing']
X_const = df[constitution_cols]
X_const_scaled = StandardScaler().fit_transform(X_const)
X_const_const = sm.add_constant(X_const_scaled)

logit_model = sm.Logit(df['disease'], X_const_const).fit(disp=0)
print("\n【4】九种体质对高血脂发病风险的Logistic回归结果（标准化系数）：")
coef_df = pd.DataFrame({
    '体质': constitution_cols,
    'coef': logit_model.params[1:].values,
    'OR': np.exp(logit_model.params[1:].values),
    'pvalue': logit_model.pvalues[1:].values
})
coef_df['abs_coef'] = coef_df['coef'].abs()
print(coef_df.sort_values('abs_coef', ascending=False).round(4))

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

# ---------- 2.4 Fisher线性判别验证 ----------
# 只保留有效分类的样本
valid_idx = df['risk_level'].notna()
lda = LinearDiscriminantAnalysis().fit(X_risk_scaled[valid_idx], df.loc[valid_idx, 'risk_level'].cat.codes)
print("\n【5】Fisher线性判别系数（验证风险分层）：")
lda_classes = lda.classes_
lda_df = pd.DataFrame(lda.coef_, columns=risk_features, index=[f'基准→{c}' for c in lda_classes])
print(lda_df.round(4))

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

def calc_monthly_drop_rate(y, f):
    """
    计算每月痰湿积分下降率
    y: 活动干预强度 (1/2/3)
    f: 每周训练频率 (1-10次)

    规则：
    - f < 5: 积分基本稳定，下降率为0
    - f >= 5: 基础下降 3%*(y-1) + 额外下降 1%*(f-5)
    """
    if f < 5:
        return 0.0
    base_drop = 0.03 * (y - 1)
    extra_drop = 0.01 * (f - 5)
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
            r = calc_monthly_drop_rate(y, f)
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
    plan = optimize_intervention(p)
    plans_123.append(plan)
    print(f"样本ID {int(plan['id'])}:")
    print(f"  患者特征: 痰湿积分={plan['s0']}, 年龄组={int(p['age_group'])}, 活动总分={p['mobility']}")
    print(f"  调理分级: {plan['x_level']}级 ({'基础/中度/强化'[plan['x_level']-1]}调理, {C_TREAT[plan['x_level']]}元/月)")
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

print("\n" + "=" * 60)
print("运行完毕")
print("=" * 60)
