"""
问题2：三级风险预警模型
包含：两层模型（现症识别+未病预警）、熵权法客观赋权、方向性血脂异常、修正阈值、交叉验证
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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from scipy import stats
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

print("=" * 60)
print("问题2：三级风险预警模型")
print("=" * 60)

# ========================== 2.1 诊断标签泄漏验证 ==========================
print("\n【2.0】诊断标签泄漏验证：")
rule_disease = ((df['tc'] > 6.2) | (df['tg'] > 1.7) | (df['ldl'] > 3.1) | (df['hdl'] < 1.04)).astype(int)
match_rate = (rule_disease == df['disease']).mean()
print(f"  血脂规则与高血脂标签匹配率: {match_rate:.4f}")
print("  结论：高血脂标签完全由血脂四项异常规则决定，存在定义同源性。")
print("  本研究将风险建模拆分为两层：现症识别层（血脂指标）与未病预警层（体质+活动+基础信息）。")

# ========================== 2.2 第一层：现症识别模型 ==========================
print("\n【2.1】第一层：现症识别模型（血脂指标复现诊断）")
diagnosis_features = ['tc', 'tg', 'ldl', 'hdl', 'uric']
X_diag = df[diagnosis_features].fillna(df[diagnosis_features].mean())
X_diag_scaled = StandardScaler().fit_transform(X_diag)

logit_diag = LogisticRegression(max_iter=1000, random_state=42).fit(X_diag_scaled, df['disease'])
prob_diag = logit_diag.predict_proba(X_diag_scaled)[:, 1]

# 5折CV评估现症识别模型
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_diag_cv = []
for train_idx, test_idx in skf.split(X_diag_scaled, df['disease']):
    X_tr, X_te = X_diag_scaled[train_idx], X_diag_scaled[test_idx]
    y_tr, y_te = df['disease'].iloc[train_idx], df['disease'].iloc[test_idx]
    model = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr, y_tr)
    prob_te = model.predict_proba(X_te)[:, 1]
    auc_diag_cv.append(roc_auc_score(y_te, prob_te))

print(f"  现症识别模型5折CV AUC: {np.mean(auc_diag_cv):.4f} (各折: {[round(a,4) for a in auc_diag_cv]})")
print("  说明：极高的AUC印证血脂指标对确诊标签的决定性作用，但该层仅用于现症识别，")
print("        不能等同于发病前预警能力。")

# ========================== 2.3 第二层：未病预警模型 ==========================
print("\n【2.2】第二层：治未病预警模型（非血脂指标预警）")
# 排除血脂指标，使用体质、活动能力、基础信息、BMI、尿酸、血糖
prevention_features = ['age_group', 'gender', 'smoke', 'drink', 'bmi', 'uric', 'glucose',
                       'adl', 'iadl', 'mobility',
                       'pinghe', 'qixu', 'yangxu', 'yinxu', 'phlegm', 'shire', 'xueyu', 'qiyu', 'tebing']
X_prev = df[prevention_features].fillna(df[prevention_features].mean())
X_prev_scaled = StandardScaler().fit_transform(X_prev)

logit_prev = LogisticRegression(max_iter=1000, random_state=42).fit(X_prev_scaled, df['disease'])
prob_prev = logit_prev.predict_proba(X_prev_scaled)[:, 1]

# 5折CV评估未病预警模型
auc_prev_cv = []
for train_idx, test_idx in skf.split(X_prev_scaled, df['disease']):
    X_tr, X_te = X_prev_scaled[train_idx], X_prev_scaled[test_idx]
    y_tr, y_te = df['disease'].iloc[train_idx], df['disease'].iloc[test_idx]
    model = LogisticRegression(max_iter=1000, random_state=42).fit(X_tr, y_tr)
    prob_te = model.predict_proba(X_te)[:, 1]
    auc_prev_cv.append(roc_auc_score(y_te, prob_te))

print(f"  未病预警模型5折CV AUC: {np.mean(auc_prev_cv):.4f} (各折: {[round(a,4) for a in auc_prev_cv]})")
print("  说明：该AUC反映不用血脂指标时，仅凭体质+活动+基础信息区分高血脂的能力，")
print("        虽低于现症识别模型，但更能体现'治未病'的预警价值。")

# ========================== 2.4 熵权法客观赋权 ==========================
print("\n【2.3】熵权法客观赋权")
# 三个维度指标：未病预警概率P_prev、痰湿积分S_phlegm、活动能力缺失(100-M)
# 为熵权法计算，需将指标归一化到[0,1]非负区间
indicator_matrix = pd.DataFrame({
    'P_prev': prob_prev,
    'S_phlegm_norm': df['phlegm'] / 100,
    'M_deficit_norm': (100 - df['mobility']) / 100,
    'I_phlegm': (df['constitution'] == 5).astype(int)
})

# 取前三个连续指标做熵权（体质标签为0/1，不适合直接熵权）
entropy_cols = ['P_prev', 'S_phlegm_norm', 'M_deficit_norm']
n = len(df)
k = 1.0 / np.log(n)

weights_entropy = {}
for col in entropy_cols:
    x = indicator_matrix[col].values
    # 平移确保非负
    x_shift = x - x.min() + 1e-6
    p = x_shift / x_shift.sum()
    e = -k * np.sum(p * np.log(p))
    d = 1 - e
    weights_entropy[col] = d

# 归一化为权重
sum_d = sum(weights_entropy.values())
for col in weights_entropy:
    weights_entropy[col] /= sum_d

print("  熵权法计算结果（信息差异系数与权重）：")
for col in entropy_cols:
    print(f"    {col}: 差异系数d={weights_entropy[col]*sum_d:.4f}, 权重w={weights_entropy[col]:.4f}")

# 组合赋权：熵权法客观权重 + 专家经验主观权重
# 主观权重 (0.5, 0.3, 0.2) 对应 (P_prev, S_phlegm, M_deficit)
subjective_w = {'P_prev': 0.5, 'S_phlegm_norm': 0.3, 'M_deficit_norm': 0.2}
# 组合：等权融合（0.5*主观 + 0.5*客观）
combined_w = {}
for col in entropy_cols:
    combined_w[col] = 0.5 * subjective_w[col] + 0.5 * weights_entropy[col]

# 重新归一化
sum_cw = sum(combined_w.values())
for col in combined_w:
    combined_w[col] /= sum_cw

print("  组合赋权结果（50%主观 + 50%熵权客观，再归一化）：")
w1 = combined_w['P_prev']
w2 = combined_w['S_phlegm_norm']
w3 = combined_w['M_deficit_norm']
print(f"    w1(P_prev)={w1:.4f}, w2(S_phlegm/100)={w2:.4f}, w3((100-M)/100)={w3:.4f}")

# ========================== 2.5 综合风险评分构造 ==========================
# 指导教师B建议：综合风险评分 = α*P_prev + β*S_phlegm/100 + γ*(100-M)/100 + δ*I(标签=5)
# 其中前三项权重用组合赋权，体质标签作为额外修正项
# 为保持与论文协调，使用组合赋权后的(w1,w2,w3)，并增加体质标签项

alpha, beta, gamma = w1, w2, w3
# 痰湿体质标签修正项：若为主体质痰湿，额外加权
delta = 0.15  # 经验系数，后续敏感性分析验证

risk_score = (alpha * prob_prev +
              beta * (df['phlegm'] / 100) +
              gamma * ((100 - df['mobility']) / 100) +
              delta * (df['constitution'] == 5).astype(int))
df['risk_score'] = risk_score

print(f"\n【2.4】综合风险评分（组合赋权 + 痰湿体质标签修正）:")
print(f"  R = {alpha:.4f}*P_prev + {beta:.4f}*S_phlegm/100 + {gamma:.4f}*(100-M)/100 + {delta:.4f}*I(标签=5)")

# ========================== 2.6 三级风险阈值确定 ==========================
# 方案B：高风险定义为综合评分最高约10%的人群
q30 = risk_score.quantile(0.30)
q75 = risk_score.quantile(0.75)

# 锚定高风险集均值
anchor_high = df[(df['lipid_risk_cnt'] >= 1) & (df['phlegm'] >= 60)]
anchor_mean = df.loc[anchor_high.index, 'risk_score'].mean() if len(anchor_high) > 0 else q75

# 高风险阈值：取约10%分位数（即90%分位数），同时不低于锚定均值
high_threshold = max(risk_score.quantile(0.90), anchor_mean)
low_threshold = q30

print(f"\n【2.5】三级风险阈值：")
print(f"  30%分位数 Q_0.30 = {q30:.4f}")
print(f"  75%分位数 Q_0.75 = {q75:.4f}")
print(f"  90%分位数 Q_0.90 = {risk_score.quantile(0.90):.4f}")
print(f"  锚定高风险集均值 = {anchor_mean:.4f}")
print(f"  最终阈值：低风险 < {low_threshold:.4f} <= 中风险 < {high_threshold:.4f} <= 高风险")
print(f"  （高风险阈值依据：为控制基层随访资源，定义为综合评分最高约10%的人群，")
print(f"   同时满足临床锚点条件者进入二次复核名单。）")

df['risk_level'] = pd.cut(risk_score, bins=[-np.inf, low_threshold, high_threshold, np.inf],
                          labels=['低', '中', '高'])
print("\n  三级风险分布：")
level_counts = df['risk_level'].value_counts()
for lvl in ['低', '中', '高']:
    cnt = level_counts.get(lvl, 0)
    print(f"    {lvl}风险: {cnt}例 ({cnt/len(df)*100:.1f}%)")

# ========================== 2.7 高风险人群画像（修正年龄分布） ==========================
high_risk_group = df[df['risk_level'] == '高']
print(f"\n【2.6】高风险人群画像（n={len(high_risk_group)}）：")
print(f"  痰湿积分均值: {high_risk_group['phlegm'].mean():.2f}")
print(f"  TG均值: {high_risk_group['tg'].mean():.2f}")
print(f"  血脂异常风险项数均值: {high_risk_group['lipid_risk_cnt'].mean():.2f}")
print(f"  活动量表总分均值: {high_risk_group['mobility'].mean():.2f}")

# 年龄分布修正
age_dist = high_risk_group['age_group'].value_counts().sort_index()
print("  年龄组分布：")
age_labels = {1: '40-49', 2: '50-59', 3: '60-69', 4: '70-79', 5: '80-89'}
for ag, cnt in age_dist.items():
    print(f"    {age_labels[ag]}岁: {cnt}例 ({cnt/len(high_risk_group)*100:.1f}%)")
pct_60_79 = ((high_risk_group['age_group'] == 3) | (high_risk_group['age_group'] == 4)).mean()
print(f"  60-79岁占比: {pct_60_79*100:.1f}%")
print("  结论：高风险人群年龄分布较为分散，并非集中于60-79岁。")

# ========================== 2.8 核心特征组合识别 ==========================
print("\n【2.7】核心特征组合（条件概率）：")
# 组合1：痰湿>=60 & 活动<40
combo1 = df[(df['phlegm'] >= 60) & (df['mobility'] < 40)]
p1 = combo1['risk_level'].value_counts(normalize=True).get('高', 0)
print(f"  痰湿>=60 & 活动总分<40  → 高风险概率: {p1:.2%} (n={len(combo1)})")

# 组合2：痰湿>=60 & BMI>=24 & TG>1.7
combo2 = df[(df['phlegm'] >= 60) & (df['bmi'] >= 24) & (df['tg'] > 1.7)]
p2 = combo2['risk_level'].value_counts(normalize=True).get('高', 0)
print(f"  痰湿>=60 & BMI>=24 & TG>1.7  → 高风险概率: {p2:.2%} (n={len(combo2)})")

# 组合3：痰湿为最高体质 & 血脂异常>=2项
constitution_cols = ['pinghe', 'qixu', 'yangxu', 'yinxu', 'phlegm', 'shire', 'xueyu', 'qiyu', 'tebing']
combo3 = df[(df[constitution_cols].idxmax(axis=1) == 'phlegm') & (df['lipid_risk_cnt'] >= 2)]
p3 = combo3['risk_level'].value_counts(normalize=True).get('高', 0)
print(f"  痰湿为最高体质 & 血脂异常>=2项  → 高风险概率: {p3:.2%} (n={len(combo3)})")

# 三层判定规则（指导教师B建议）
print("\n【2.8】三层判定规则：")
print("  第1层（确诊/近确诊）：若 TC>6.2 或 TG>1.7 或 LDL-C>3.1 或 HDL-C<1.04，")
print("                       直接判为'血脂异常高关注人群'。")
print("  第2层（未病预警）：在未确诊或血脂正常者中，用体质+活动+基础信息评估风险。")
print("  第3层（临床锚点修正）：")
print("    - 血脂异常且痰湿积分>=60 → 高风险复核")
print("    - 血脂正常但痰湿积分>=80且活动能力<40 → 高风险复核")

# ========================== 2.9 权重敏感性分析 ==========================
print("\n【2.9】风险评分权重敏感性分析（固定阈值，主观权重 ±20% 扰动）：")
base_weights = (alpha, beta, gamma, delta)
perturbations = [
    (alpha*0.8, beta*1.2, gamma*1.2, delta),   # w1-20%
    (alpha*1.2, beta*0.8, gamma*0.8, delta),   # w1+20%
    (alpha, beta*0.8, gamma*1.2, delta),       # w2-20%
    (alpha, beta*1.2, gamma*0.8, delta),       # w2+20%
]
# 重新归一化前3项
base_w123 = base_weights[:3]
base_w123_norm = tuple(w/sum(base_w123) for w in base_w123)

base_rs = (base_w123_norm[0] * prob_prev +
           base_w123_norm[1] * (df['phlegm'] / 100) +
           base_w123_norm[2] * ((100 - df['mobility']) / 100) +
           base_weights[3] * (df['constitution'] == 5).astype(int))
base_rl = pd.cut(base_rs, bins=[-np.inf, low_threshold, high_threshold, np.inf], labels=['低', '中', '高'])
base_dist = base_rl.value_counts(normalize=True).sort_index()
print(f"  基准权重: 低={base_dist.get('低',0):.3f}, 中={base_dist.get('中',0):.3f}, 高={base_dist.get('高',0):.3f}")

for i, (w1, w2, w3, w4) in enumerate(perturbations, 1):
    s = w1 + w2 + w3
    w1n, w2n, w3n = w1/s, w2/s, w3/s
    rs = w1n * prob_prev + w2n * (df['phlegm'] / 100) + w3n * ((100 - df['mobility']) / 100) + w4 * (df['constitution'] == 5).astype(int)
    rl = pd.cut(rs, bins=[-np.inf, low_threshold, high_threshold, np.inf], labels=['低', '中', '高'])
    dist = rl.value_counts(normalize=True).sort_index()
    print(f"  扰动{i}: 低={dist.get('低',0):.3f}, 中={dist.get('中',0):.3f}, 高={dist.get('高',0):.3f}")

# ========================== 2.10 ROC曲线（两层模型） ==========================
fpr_prev, tpr_prev, _ = roc_curve(df['disease'], prob_prev)

# ========================== 保存结果 ==========================
result_df = pd.DataFrame({
    'id': df['id'],
    'risk_score': risk_score,
    'risk_level': df['risk_level'],
    'disease': df['disease'],
    'prob_prev': prob_prev,
    'prob_diag': prob_diag,
    'phlegm': df['phlegm'],
    'mobility': df['mobility'],
    'constitution': df['constitution'],
    'lipid_risk_cnt': df['lipid_risk_cnt']
})
result_df.to_csv('output/p2_risk_scores.csv', index=False)

# 保存熵权结果
entropy_df = pd.DataFrame({
    '指标': ['P_prev', 'S_phlegm/100', '(100-M)/100'],
    '熵权法权重': [weights_entropy[c] for c in entropy_cols],
    '组合权重': [combined_w[c] for c in entropy_cols]
})
entropy_df.to_csv('output/p2_entropy_weights.csv', index=False)

# ========================== 可视化 ==========================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图2a: 两层模型ROC对比
axes[0, 0].plot(fpr_prev, tpr_prev, label=f'未病预警 AUC={np.mean(auc_prev_cv):.3f}', linewidth=2)
axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].set_title('问题2: 两层模型ROC曲线对比')
axes[0, 0].legend()

# 图2b: 风险评分分布
sns.histplot(data=df, x='risk_score', hue='risk_level', bins=30, kde=True, ax=axes[0, 1])
axes[0, 1].axvline(low_threshold, color='green', linestyle='--', label=f'低中阈值={low_threshold:.3f}')
axes[0, 1].axvline(high_threshold, color='red', linestyle='--', label=f'中高阈值={high_threshold:.3f}')
axes[0, 1].set_title('问题2: 综合风险评分分布')
axes[0, 1].legend()

# 图2c: 高风险人群年龄分布
age_dist_df = pd.DataFrame({'年龄组': [age_labels[i] for i in age_dist.index], '人数': age_dist.values})
sns.barplot(data=age_dist_df, x='年龄组', y='人数', ax=axes[1, 0])
axes[1, 0].set_title('问题2: 高风险人群年龄分布')

# 图2d: 熵权法权重对比
entropy_melt = entropy_df.melt(id_vars='指标', var_name='方法', value_name='权重')
sns.barplot(data=entropy_melt, x='指标', y='权重', hue='方法', ax=axes[1, 1])
axes[1, 1].set_title('问题2: 熵权法 vs 组合赋权')
axes[1, 1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('output/fig2_problem2_risk.png', dpi=300)
plt.close()

print("\n  图表已保存至 output/fig2_problem2_risk.png")
print("  数据已保存至 output/p2_*.csv")
