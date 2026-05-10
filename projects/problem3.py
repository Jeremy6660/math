"""
问题3：干预方案优化模型（动态逐月规划）
核心升级：调理分级x不再固定为初始级别，而是每月根据当前痰湿积分动态调整
包含：逐月模拟、样本1/2/3月度轨迹、全体278位患者优化、依从性衰减敏感性分析
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Heiti TC', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
os.makedirs('output', exist_ok=True)

# ========================== 数据读取 ==========================
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

print("=" * 60)
print("问题3：干预方案优化模型（动态逐月规划）")
print("=" * 60)

# ========================== 参数设定 ==========================
C_TREAT = {1: 30, 2: 80, 3: 130}
C_ACT = {1: 3, 2: 5, 3: 8}
WEEKS_PER_MONTH = 4
MONTHS = 6
MAX_COST = 2000

def get_treatment_level(phlegm_score):
    if phlegm_score <= 58:
        return 1
    elif phlegm_score <= 61:
        return 2
    else:
        return 3

def get_max_activity_intensity(age_group, mobility_score):
    if age_group in [1, 2]:
        y_max_age = 3
    elif age_group in [3, 4]:
        y_max_age = 2
    else:
        y_max_age = 1
    if mobility_score < 40:
        y_max_mob = 1
    elif mobility_score < 60:
        y_max_mob = 2
    else:
        y_max_mob = 3
    return min(y_max_age, y_max_mob)

def calc_monthly_drop_rate(y, f, age_group, mobility, compliance_params=None):
    """
    计算每月痰湿积分下降率（含依从性衰减）
    compliance_params: 可传入自定义衰减参数做敏感性分析
      默认: {'age5_factor': 0.5, 'age5_cap': 6.0, 'mob_low_factor': 0.6, 'mob_low_cap': 6.2, 'age34_factor': 0.8}
    """
    if compliance_params is None:
        compliance_params = {
            'age5_factor': 0.5, 'age5_cap': 6.0,
            'mob_low_factor': 0.6, 'mob_low_cap': 6.2,
            'age34_factor': 0.8
        }
    if f < 5:
        return 0.0
    if age_group == 5:
        effective_f = min(5 + (f - 5) * compliance_params['age5_factor'], compliance_params['age5_cap'])
    elif mobility < 40:
        effective_f = min(5 + (f - 5) * compliance_params['mob_low_factor'], compliance_params['mob_low_cap'])
    elif age_group in [3, 4]:
        effective_f = 5 + (f - 5) * compliance_params['age34_factor']
    else:
        effective_f = f
    base_drop = 0.03 * (y - 1)
    extra_drop = 0.01 * (effective_f - 5)
    return base_drop + extra_drop

def simulate_months(s0, y, f, age_group, mobility, months=MONTHS, compliance_params=None):
    """
    逐月模拟干预过程，调理分级x每月根据当前积分动态调整
    返回: {
        's_trajectory': [S0, S1, S2, S3, S4, S5, S6] (每月初的积分，S6为6个月末),
        'x_trajectory': [x1, x2, x3, x4, x5, x6] (每月的调理分级),
        'cost_trajectory': [c1, c2, c3, c4, c5, c6] (每月成本),
        'total_cost': 总成本,
        's_final': 最终积分
    }
    """
    s_current = s0
    s_trajectory = [s_current]
    x_trajectory = []
    cost_trajectory = []
    total_cost = 0
    r = calc_monthly_drop_rate(y, f, age_group, mobility, compliance_params)

    for _ in range(months):
        x = get_treatment_level(s_current)
        monthly_cost = C_TREAT[x] + WEEKS_PER_MONTH * f * C_ACT[y]
        total_cost += monthly_cost
        if total_cost > MAX_COST:
            return None  # 超支，非法方案
        x_trajectory.append(x)
        cost_trajectory.append(monthly_cost)
        s_current = s_current * (1 - r)
        s_trajectory.append(s_current)

    return {
        's_trajectory': s_trajectory,
        'x_trajectory': x_trajectory,
        'cost_trajectory': cost_trajectory,
        'total_cost': total_cost,
        's_final': s_current
    }

def optimize_intervention_dynamic(patient, compliance_params=None):
    """
    动态逐月规划优化：遍历所有合法(y,f)，逐月模拟，选择最终积分最小者
    """
    s0 = patient['phlegm']
    y_max = get_max_activity_intensity(patient['age_group'], patient['mobility'])

    best_plan = None
    best_final = float('inf')

    for y in range(1, y_max + 1):
        for f in range(1, 11):
            result = simulate_months(s0, y, f, patient['age_group'], patient['mobility'], compliance_params=compliance_params)
            if result is None:
                continue
            if result['s_final'] < best_final - 1e-6:
                best_final = result['s_final']
                best_plan = {
                    'id': patient['id'],
                    'y_level': y,
                    'freq': f,
                    's0': s0,
                    **result
                }
            elif abs(result['s_final'] - best_final) < 1e-6 and result['total_cost'] < best_plan['total_cost']:
                best_plan['y_level'] = y
                best_plan['freq'] = f
                best_plan['total_cost'] = result['total_cost']
                best_plan['s_trajectory'] = result['s_trajectory']
                best_plan['x_trajectory'] = result['x_trajectory']
                best_plan['cost_trajectory'] = result['cost_trajectory']
                best_plan['s_final'] = result['s_final']

    return best_plan

# ========================== 样本ID 1、2、3月度轨迹输出 ==========================
print("\n【1】样本ID 1、2、3 的最优干预方案（动态逐月规划）：\n")
sample_ids = [1, 2, 3]
plans_123 = []
for sid in sample_ids:
    p = df[df['id'] == sid].iloc[0]
    if p['constitution'] != 5:
        print(f"  注意：样本ID {sid} 不是痰湿体质（constitution={int(p['constitution'])}），仍按题目要求输出方案。\n")
    plan = optimize_intervention_dynamic(p)
    plans_123.append(plan)
    lvl_name = ('基础', '中度', '强化')
    print(f"样本ID {int(plan['id'])}: 痰湿积分={plan['s0']}, 年龄组={int(p['age_group'])}, 活动总分={p['mobility']}")
    print(f"  活动强度: {plan['y_level']}级 (单次{C_ACT[plan['y_level']]}元), 每周频率: {plan['freq']}次")
    print(f"  6个月总成本: {plan['total_cost']}元, 预期6个月后痰湿积分: {plan['s_final']:.2f} (下降{plan['s0']-plan['s_final']:.2f}分)")
    print(f"  {'月份':>4} {'月初积分':>8} {'调理分级':>8} {'月成本(元)':>10}")
    for m in range(MONTHS):
        print(f"  {m+1:>4} {plan['s_trajectory'][m]:>8.2f} {lvl_name[plan['x_trajectory'][m]-1]+'('+str(plan['x_trajectory'][m])+')':>8} {plan['cost_trajectory'][m]:>10}")
    print(f"  {'期末':>4} {plan['s_trajectory'][MONTHS]:>8.2f}")
    print()

# ========================== 全体痰湿体质患者优化（n=278） ==========================
print("【2】对全体痰湿体质患者（体质标签=5）执行优化，n=278")
phlegm_patients = df[df['constitution'] == 5].copy()
print(f"  确认：痰湿体质患者人数 = {len(phlegm_patients)}")

all_plans = []
for _, p in phlegm_patients.iterrows():
    all_plans.append(optimize_intervention_dynamic(p))

# 过滤掉None（理论上不应出现，因为至少存在低成本方案）
all_plans = [p for p in all_plans if p is not None]
plan_df = pd.DataFrame(all_plans)

phlegm_patients['opt_y'] = plan_df['y_level'].values
phlegm_patients['opt_f'] = plan_df['freq'].values
phlegm_patients['opt_cost'] = plan_df['total_cost'].values
phlegm_patients['opt_s_final'] = plan_df['s_final'].values
phlegm_patients['opt_s_drop'] = plan_df['s0'].values - plan_df['s_final'].values

print("\n【3】患者特征—最优方案匹配规律：")

# 按年龄组
print("\n(a) 按年龄组统计最优方案均值：")
age_summary = phlegm_patients.groupby('age_group').agg({
    'opt_y': 'mean',
    'opt_f': 'mean',
    'opt_cost': 'mean',
    'opt_s_final': 'mean',
    'opt_s_drop': 'mean'
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
    'opt_s_final': 'mean',
    'opt_s_drop': 'mean'
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
    'opt_s_final': 'mean',
    'opt_s_drop': 'mean'
}).round(2)
print(phlegm_summary)

# 成本-效果分析
print("\n【4】单位成本降分效果（按初始调理分级统计）：")
# 注意：由于动态调整，患者可能在干预过程中改变调理分级
# 这里按"初始调理分级"（基于初始积分）统计
phlegm_patients['initial_x'] = phlegm_patients['phlegm'].apply(get_treatment_level)
cost_effect = phlegm_patients.groupby('initial_x').agg({
    'opt_s_drop': 'mean',
    'opt_cost': 'mean'
}).round(2)
cost_effect['单位成本降分(分/元)'] = (cost_effect['opt_s_drop'] / cost_effect['opt_cost']).round(4)
cost_effect.index = ['基础(x=1)', '中度(x=2)', '强化(x=3)']
print(cost_effect)

# ========================== 依从性衰减敏感性分析 ==========================
print("\n【5】依从性衰减敏感性分析：")
sensitivity_configs = {
    '保守方案': {'age5_factor': 0.3, 'age5_cap': 5.5, 'mob_low_factor': 0.4, 'mob_low_cap': 5.5, 'age34_factor': 0.6},
    '基准方案': {'age5_factor': 0.5, 'age5_cap': 6.0, 'mob_low_factor': 0.6, 'mob_low_cap': 6.2, 'age34_factor': 0.8},
    '乐观方案': {'age5_factor': 0.7, 'age5_cap': 7.0, 'mob_low_factor': 0.8, 'mob_low_cap': 7.5, 'age34_factor': 0.9},
}

sens_results = []
for config_name, params in sensitivity_configs.items():
    sid_results = []
    for sid in sample_ids:
        p = df[df['id'] == sid].iloc[0]
        plan = optimize_intervention_dynamic(p, compliance_params=params)
        sid_results.append({
            'config': config_name,
            'id': sid,
            'freq': plan['freq'],
            'cost': plan['total_cost'],
            's_drop': plan['s0'] - plan['s_final']
        })
    # 计算该配置下278人的均值
    temp_plans = []
    for _, p in phlegm_patients.iterrows():
        temp_plans.append(optimize_intervention_dynamic(p, compliance_params=params))
    temp_plans = [p for p in temp_plans if p is not None]
    avg_cost = np.mean([p['total_cost'] for p in temp_plans])
    avg_drop = np.mean([p['s0'] - p['s_final'] for p in temp_plans])
    for sr in sid_results:
        sr['avg_cost_278'] = avg_cost
        sr['avg_drop_278'] = avg_drop
        sens_results.append(sr)

sens_df = pd.DataFrame(sens_results)
print("\n  敏感性分析结果：")
for config_name in sensitivity_configs:
    sub = sens_df[sens_df['config'] == config_name]
    print(f"\n  {config_name}:")
    print(f"    278人平均成本: {sub['avg_cost_278'].iloc[0]:.2f}元, 平均降分: {sub['avg_drop_278'].iloc[0]:.2f}分")
    for _, row in sub.iterrows():
        print(f"    ID{int(row['id'])}: 最优频率={int(row['freq'])}次/周, 成本={row['cost']:.0f}元, 降分={row['s_drop']:.2f}分")

# ========================== 保存结果 ==========================
phlegm_patients.to_csv('output/p3_phlegm_patients_results.csv', index=False)
sens_df.to_csv('output/p3_sensitivity_analysis.csv', index=False)

# 保存样本1,2,3的月度轨迹
for plan in plans_123:
    traj_df = pd.DataFrame({
        '月份': list(range(1, MONTHS+1)) + ['期末'],
        '月初积分': plan['s_trajectory'],
        '调理分级': plan['x_trajectory'] + ['--'],
        '月成本(元)': plan['cost_trajectory'] + [plan['total_cost']]
    })
    traj_df.to_csv(f"output/p3_id{int(plan['id'])}_trajectory.csv", index=False)

# ========================== 可视化 ==========================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 图3a: 样本1,2,3的干预历程折线图
for plan in plans_123:
    months_axis = list(range(MONTHS + 1))
    axes[0, 0].plot(months_axis, plan['s_trajectory'], marker='o', label=f"ID {int(plan['id'])}")
axes[0, 0].set_xlabel('月份')
axes[0, 0].set_ylabel('痰湿积分')
axes[0, 0].set_title('问题3: 样本ID 1/2/3 干预历程（痰湿积分变化）')
axes[0, 0].legend()
axes[0, 0].set_xticks(range(MONTHS + 1))
axes[0, 0].set_xticklabels(['初始'] + [f'{m}月' for m in range(1, MONTHS+1)])
axes[0, 0].grid(True, alpha=0.3)

# 图3b: 样本1,2,3的月度成本柱状图
width = 0.25
x_pos = np.arange(MONTHS)
for i, plan in enumerate(plans_123):
    axes[0, 1].bar(x_pos + i*width, plan['cost_trajectory'], width, label=f"ID {int(plan['id'])}")
axes[0, 1].set_xlabel('月份')
axes[0, 1].set_ylabel('月度成本（元）')
axes[0, 1].set_title('问题3: 样本ID 1/2/3 月度成本分布')
axes[0, 1].set_xticks(x_pos + width)
axes[0, 1].set_xticklabels([f'{m}月' for m in range(1, MONTHS+1)])
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 图3c: 最优频率分布（按年龄组）
sns.boxplot(data=phlegm_patients, x='age_group', y='opt_f', ax=axes[1, 0])
axes[1, 0].set_title('问题3: 最优训练频率按年龄组分布（n=278）')
axes[1, 0].set_xlabel('年龄组')
axes[1, 0].set_ylabel('最优频率（次/周）')

# 图3d: 敏感性分析对比
sens_pivot = sens_df.pivot_table(index='config', values=['avg_cost_278', 'avg_drop_278'], aggfunc='first')
sens_pivot = sens_pivot.reindex(['保守方案', '基准方案', '乐观方案'])
x_labels = sens_pivot.index
x_pos = np.arange(len(x_labels))
ax_twin = axes[1, 1].twinx()
bars = axes[1, 1].bar(x_pos - 0.2, sens_pivot['avg_cost_278'], 0.4, label='平均成本', color='steelblue')
line = ax_twin.plot(x_pos + 0.2, sens_pivot['avg_drop_278'], 'ro-', label='平均降分', linewidth=2, markersize=8)
axes[1, 1].set_xlabel('依从性衰减假设')
axes[1, 1].set_ylabel('平均成本（元）', color='steelblue')
ax_twin.set_ylabel('平均降分（分）', color='red')
axes[1, 1].set_title('问题3: 依从性衰减敏感性分析（278人平均）')
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(x_labels)
axes[1, 1].legend(loc='upper left')
ax_twin.legend(loc='upper right')

plt.tight_layout()
plt.savefig('output/fig3_problem3_optimization.png', dpi=300)
plt.close()

print("\n  图表已保存至 output/fig3_problem3_optimization.png")
print("  数据已保存至 output/p3_*.csv")
