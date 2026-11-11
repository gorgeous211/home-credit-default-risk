# %%
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import toad
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve


# 文件路径
DATA_PATH = Path.cwd() / "data"

# 主申请表
application_train = pd.read_csv(DATA_PATH / 'raw' / 'application_train.csv')

cols_eda = pd.read_csv(DATA_PATH / 'interim' / 'cols_eda.csv')

# 提取房屋相关列（标准化信息）
house_cols = cols_eda.loc[
    cols_eda['Description'].str.startswith('Normalized information'), 'Row'
].tolist()

# %%
# -- 特殊异常值替换与标记 --
# DAYS_EMPLOYED：365243 实际上是退休/无业人员的特殊标记，标记出来后替换为 NaN
application_train['FLAG_DAYS_EMPLOYED_ANOMALOUS'] = (application_train['DAYS_EMPLOYED'] == 365243).astype(int)
application_train['DAYS_EMPLOYED'] = application_train['DAYS_EMPLOYED'].replace(365243, np.nan)
application_train['CODE_GENDER'] = application_train['CODE_GENDER'].replace('XNA', 'F')
# 社交圈异常值裁剪
application_train.loc[application_train['OBS_30_CNT_SOCIAL_CIRCLE'] > 30, 'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan
application_train.loc[application_train['OBS_60_CNT_SOCIAL_CIRCLE'] > 30, 'OBS_60_CNT_SOCIAL_CIRCLE'] = np.nan

# %%
# -- 高阶单表特征衍生 --
def add_advanced_features(df, house_cols_list):
    """针对申请主表的高阶特征工程（包含还款能力、期限代理、时间交叉、EXT外部评分深挖及质量缺失指标）"""
    # 1. 还款能力与负债压力
    df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['FREE_INCOME_PROXY'] = df['AMT_INCOME_TOTAL'] - df['AMT_ANNUITY']
    df['ANNUITY_TO_CREDIT_RATIO'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    df['GOODS_TO_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    df['DOWN_PAYMENT_PROXY'] = df['AMT_GOODS_PRICE'] - df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])

    # 2. 期限代理与利率成本
    df['ESTIMATED_TENURE'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['ANNUITY_TO_GOODS_RATIO'] = df['AMT_ANNUITY'] / df['AMT_GOODS_PRICE']

    # 3. 稳定性与时间交叉比率
    df['EMPLOYED_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / (-df['DAYS_BIRTH'] / 365)
    df['ID_TO_BIRTH_RATIO'] = df['DAYS_ID_PUBLISH'] / df['DAYS_BIRTH']
    df['REG_TO_BIRTH_RATIO'] = df['DAYS_REGISTRATION'] / df['DAYS_BIRTH']
    df['PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['PHONE_TO_EMPLOYED_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED'].replace(0, np.nan)
    df['AGE_AT_EMPLOYMENT'] = (df['DAYS_BIRTH'] - df['DAYS_EMPLOYED']) / 365
    df['REG_TO_ID_RATIO'] = df['DAYS_REGISTRATION'] / df['DAYS_ID_PUBLISH'].replace(0, np.nan)

    # 4. 外部征信评分深挖 (单表提分核心)
    ext_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    df['EXT_SOURCE_MEAN'] = df[ext_cols].mean(axis=1)
    df['EXT_SOURCE_STD'] = df[ext_cols].std(axis=1)
    df['EXT_SOURCE_MIN'] = df[ext_cols].min(axis=1)
    df['EXT_SOURCE_MAX'] = df[ext_cols].max(axis=1)
    df['EXT_SOURCE_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['EXT_SOURCE_WEIGHTED'] = (
        df['EXT_SOURCE_1'] * 2 + df['EXT_SOURCE_2'] * 3 + df['EXT_SOURCE_3'] * 5
    )
    df['EXT_SOURCE_NAN_CNT'] = df[ext_cols].isna().sum(axis=1)
    df['EXT_MEAN_TO_BIRTH'] = df['EXT_SOURCE_MEAN'] / (-df['DAYS_BIRTH'] / 365)
    df['EXT_MEAN_TO_CREDIT_RATIO'] = df['EXT_SOURCE_MEAN'] / df['CREDIT_TO_INCOME_RATIO']

    # 5. 风险信号聚合与同源缺失标志位
    avg_house_cols = [col for col in house_cols_list if col.endswith('AVG') and col in df.columns]
    if avg_house_cols:
        df['HOUSE_INFO_MISSING_RATE'] = df[avg_house_cols].isna().mean(axis=1)

    doc_cols = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22) if f"FLAG_DOCUMENT_{i}" in df.columns]
    df['DOCUMENTS_PROVIDED_CNT'] = df[doc_cols].sum(axis=1)

    addr_cols = [
        'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY',
        'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION'
    ]
    df['ADDRESS_MISMATCH_MEAN'] = df[addr_cols].mean(axis=1)
    df['DEF_OBS_RATIO_30'] = df['DEF_30_CNT_SOCIAL_CIRCLE'] / df['OBS_30_CNT_SOCIAL_CIRCLE'].replace(0, np.nan)
    df['DEF_OBS_RATIO_60'] = df['DEF_60_CNT_SOCIAL_CIRCLE'] / df['OBS_60_CNT_SOCIAL_CIRCLE'].replace(0, np.nan)

    # 整行总缺失特征计数（质量与配合度代理）
    df['TOTAL_NAN_COUNT'] = df.isna().sum(axis=1)


# 执行特征衍生
add_advanced_features(application_train, house_cols)

# 6. 分组统计特征（同行业/同职业相对水平）
for grp_col in ['OCCUPATION_TYPE', 'ORGANIZATION_TYPE']:
    grp_stats = application_train.groupby(grp_col, observed=True)['AMT_INCOME_TOTAL'].agg(['mean']).reset_index()
    grp_stats = grp_stats.rename(columns={'mean': f'{grp_col}_INCOME_MEAN'})

    application_train = application_train.merge(grp_stats, on=grp_col, how='left')
    application_train[f'INCOME_TO_{grp_col}_RATIO'] = (
        application_train['AMT_INCOME_TOTAL'] / application_train[f'{grp_col}_INCOME_MEAN']
    )

del grp_col, grp_stats
del house_cols
del add_advanced_features

# %%
ignore_cols = ['SK_ID_CURR', 'TARGET']
all_feature_cols = [c for c in application_train.columns if c not in ignore_cols]  # 共158个特征
orig_num_cols = [
    c for c in all_feature_cols
    if c in application_train.select_dtypes(include='number').columns and c not in ignore_cols
]

# -- 粗筛 --
# 1.基础清洗：剔除极高同值特征，同值率大于0.95
high_primary_value_ratio = cols_eda.loc[
    cols_eda['primary_value_ratio'] > 0.95, 'Row'
].tolist()

# 衍生特征同值率
derived_cols = [col for col in all_feature_cols if col not in cols_eda['Row'].tolist()]
for d_col in derived_cols:
    val_counts = application_train[d_col].value_counts(normalize=True, dropna=False)
    if val_counts.iloc[0] > 0.95:
        high_primary_value_ratio.append(d_col)

del d_col, val_counts

# 2.稳定性过滤：剔除分布漂移严重的特征，psi > 0.25（原始特征基于 EDA 结果，衍生特征基于 toad 库计算）
high_psi = cols_eda.loc[cols_eda['psi'] > 0.25, 'Row'].tolist()

# 衍生特征稳定性评估：通过开发集与验证集切分，使用 toad 库计算 PSI
train_sub, test_sub = train_test_split(
    application_train,
    test_size=0.2,
    random_state=123,
    stratify=application_train['TARGET']
)

combiner = toad.transform.Combiner()
combiner.fit(train_sub[derived_cols], method='quantile', n_bins=10)
derived_psi = toad.metrics.PSI(test_sub[derived_cols], train_sub[derived_cols], combiner=combiner)

# 剔除 psi > 0.25 的衍生特征
high_psi_derived = derived_psi[derived_psi > 0.25].index.tolist()
high_psi.extend(high_psi_derived)

del train_sub, test_sub, combiner, high_psi_derived

# 3.多重共线性：对原始数值特征采用 Spearman 相关系数衡量单调相关性，阈值设置为0.95
features_corr = application_train[[
    col for col in orig_num_cols
    if col not in high_primary_value_ratio
    and col not in high_psi
]].corr(method='spearman')

result = features_corr.stack().reset_index()
result.columns = ['feature1', 'feature2', 'corr']
result = result[np.abs(result['corr']).between(0.95, 1, inclusive='left')]

psi_dict = dict(zip(cols_eda['Row'], cols_eda['psi']))
psi_dict.update(derived_psi.to_dict())
result['psi1'] = result['feature1'].map(psi_dict).fillna(0)
result['psi2'] = result['feature2'].map(psi_dict).fillna(0)
high_corr = result.loc[result['psi1'] > result['psi2'], 'feature1'].tolist()

del result, psi_dict, derived_psi

# 粗筛需要剔除的特征清单
delete_feature = set(high_primary_value_ratio + high_psi + high_corr)

# 初筛后的候选特征集（包含保留的优质原始特征与稳定衍生特征）
candidate_feature_cols = [
    col for col in all_feature_cols
    if col not in delete_feature
]

del orig_num_cols, high_primary_value_ratio, high_psi, features_corr
del ignore_cols, all_feature_cols, derived_cols, high_corr, delete_feature, cols_eda

# %%
# -- 精筛 --
# 使用五折交叉验证汇总 Gain，降低单次切分带来的重要性波动。
X = application_train[candidate_feature_cols].copy()
y = application_train['TARGET']

categorical_cols = X.select_dtypes(include=['object', 'category']).columns
X[categorical_cols] = X[categorical_cols].astype('category')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
gain_importance = np.zeros(X.shape[1], dtype=float)

params = {
    # 1. 树结构复杂度控制 (防过拟合)
    'max_depth': 5,             # 默认值: -1 (不限)。限制树深为5，严格约束特征交互阶数，防止拟合长尾稀疏噪声
    'num_leaves': 16,           # 默认值: 31。Leaf-wise策略核心参数，设为16远低于2^5=32，显著降低树复杂度与叶节点方差
    'min_child_samples': 50,    # 默认值: 20。单叶节点最小样本数设为50，保证叶子节点违约率统计的置信度，压制个别孤立坏账噪声

    # 2. 随机采样引入 (增强泛化与特征多样性)
    'subsample': 0.8,           # 默认值: 1.0。每棵树随机采样80%样本，打破弱特征对特定样本依赖，提高抗噪性
    'subsample_freq': 1,        # 默认值: 0。每1轮迭代均执行行采样；必须>0以激活subsample配置
    'colsample_bytree': 0.8,    # 默认值: 1.0。每棵树随机抽取80%特征，防止强特征(如EXT)垄断初始分裂，促使潜在优质特征充分展现

    # 3. 正则化约束 (稀疏化与平滑权重)
    'reg_alpha': 1.0,           # 默认值: 0.0。L1正则化，促使无有效增益的弱特征叶节点权重归零，有利于精筛阶段特征稀疏化
    'reg_lambda': 1.0,          # 默认值: 0.0。L2正则化，惩罚极端叶子权重，平滑预测输出，防止个别分箱预测过激

    # 4. 学习步长与迭代控制
    'learning_rate': 0.05,      # 默认值: 0.1。较小步长使梯度下降更平稳细腻，配合早停避免过早陷入局部次优
    'n_estimators': 1000,       # 默认值: 100。给予充裕迭代轮数上限，配合 early_stopping=50 动态截断，确保充分收敛

    # 5. 业务目标与评估指标
    'objective': 'binary',      # 默认值: 'regression'。信贷违约二分类任务(Logloss损失)
    'metric': 'auc',            # 默认值: ''。信贷样本极不平衡(好坏比11:1)，AUC衡量全局排序能力且不受阈值与样本先验概率影响
    'class_weight': None,       # 默认值: None。保持自然违约率(约8%)，确保预测概率反映真实违约风险，AUC指标对类别不平衡天然鲁棒

    # 6. 工程与计算配置
    'random_state': 123,        # 默认值: None。固定随机种子，保证数据切分、采样及特征重要性计算100%可复现
    'importance_type': 'gain',  # 默认值: 'split'。精筛核心！按分裂带来的实际信息增益而非分裂次数评估，避免偏向高基数特征
    'n_jobs': -1,               # 默认值: -1。充分利用多核CPU多线程并行计算
    'verbose': -1               # 默认值: 1/0。抑制迭代冗余输出，保持日志清爽
}

for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    fold_gain = model.booster_.feature_importance(importance_type='gain')
    fold_gain_sum = fold_gain.sum()
    if fold_gain_sum > 0:
        gain_importance += fold_gain / fold_gain_sum / skf.n_splits

importance_df = pd.DataFrame({
    'feature': X.columns,
    'gain_importance': gain_importance,
}).sort_values('gain_importance', ascending=False, ignore_index=True)

# 计算累计的 Gain，保留前 95% 累计重要性的核心特征
importance_df['cumulative_gain'] = importance_df['gain_importance'].cumsum()

# 筛选后的特征（包含刚好跨过 0.95 门槛的特征）
selected_feature_cols = importance_df.loc[
    importance_df['cumulative_gain'].shift(1, fill_value=0) < 0.95, 'feature'
].tolist()

print(f"\n精筛完成: 总候选特征 {len(candidate_feature_cols)} 个，保留核心特征 {len(selected_feature_cols)} 个。")
# 精筛完成: 总候选特征 95 个，保留核心特征 48 个。
print("\nTop 15 重要特征排名:")
print(importance_df.head(15)[['feature', 'gain_importance', 'cumulative_gain']])
#                     feature  gain_importance  cumulative_gain
# 0            EXT_SOURCE_MIN         0.217490         0.217490
# 1            EXT_SOURCE_MAX         0.135228         0.352718
# 2         ORGANIZATION_TYPE         0.087179         0.439897
# 3              EXT_SOURCE_3         0.058392         0.498289
# 4   ANNUITY_TO_CREDIT_RATIO         0.039752         0.538041
# 5          ESTIMATED_TENURE         0.036912         0.574954
# 6     GOODS_TO_CREDIT_RATIO         0.028353         0.603307
# 7              EXT_SOURCE_2         0.023420         0.626727
# 8   EMPLOYED_TO_BIRTH_RATIO         0.021166         0.647894
# 9           OCCUPATION_TYPE         0.020012         0.667905
# 10               DAYS_BIRTH         0.019550         0.687455
# 11           EXT_SOURCE_STD         0.015510         0.702965
# 12             EXT_SOURCE_1         0.015452         0.718417
# 13        EXT_MEAN_TO_BIRTH         0.014596         0.733013
# 14      EXT_SOURCE_WEIGHTED         0.014437         0.747450

# 保存特征重要性表
(DATA_PATH / 'interim').mkdir(parents=True, exist_ok=True)
importance_df.to_csv(DATA_PATH / 'interim' / 'lgb_feature_importance.csv', index=False)

# 清理精筛过程的中间变量与大型数据副本
del X, categorical_cols, candidate_feature_cols, gain_importance, importance_df
del fold, train_idx, valid_idx, X_train, X_valid, y_train, y_valid, model, fold_gain, fold_gain_sum


# %%
# -- 最终模型训练与本地验证 (基于精筛后的核心特征与超参数搜索) --
def calc_ks(y_true, y_prob):
    """计算二分类模型的 KS (Kolmogorov-Smirnov) 统计量"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return np.max(np.abs(tpr - fpr))

X_final = application_train[selected_feature_cols].copy()
cat_cols_final = X_final.select_dtypes(include=['object', 'category']).columns
X_final[cat_cols_final] = X_final[cat_cols_final].astype('category')

# ---------------------------------------------------------
# 1. 超参数确定与搜索规则设置
# ---------------------------------------------------------
# 规则说明：
# - learning_rate (0.05)、n_estimators (1000): 根据经验确定，较小步长配合早停确保充分平稳收敛
# - min_child_samples (50): 根据数据集规模（约 30.7 万样本）确定，单叶节点最小样本数取 50 保证违约率置信度并压制个别孤立噪声
# - max_depth、num_leaves、subsample、colsample_bytree、reg_alpha、reg_lambda: 通过 Optuna 算法进行搜索
#   * max_depth: [3, 4, 5]
#   * num_leaves: 与 max_depth 配合，严格小于 2^max_depth - 1
#   * subsample: [0.8, 0.9, 1.0] (整值采样比例)
#   * colsample_bytree: [0.8, 0.9, 1.0]
#   * reg_alpha、reg_lambda: 范围 1.0 ~ 1000.0 (log 对数均匀采样)，后续对搜索值过大给出特征与模型诊断提示

# ---------------------------------------------------------
# 2. Optuna 超参数搜索目标函数
# ---------------------------------------------------------
# 说明：
#   - objective() 是 Optuna 的核心评估函数。
#   - 每一次调用都对应一次 "trial"（一次试验/一次参数组合搜索）。
#   - Optuna 会自动在给定范围内采样超参数，并根据返回值决定下一轮如何搜索。
#   - 这里的目标是最大化验证 AUC，因此 study 的 direction='maximize'。
# ---------------------------------------------------------
def objective(trial):
    # 1) 设定当前 trial 要搜索的超参数及其候选空间
    #    `suggest_*` 是 Optuna 的固定接口，表示“在这个范围内抽样一个值”

    # 树的最大深度：范围 3~5，控制模型复杂度
    max_depth = trial.suggest_int('max_depth', 3, 5)

    # num_leaves 需要和 max_depth 配合使用，且必须严格小于 2^max_depth - 1
    # 这是 LightGBM 的常见约束，避免叶子节点过多导致过拟合
    max_leaves_bound = (2 ** max_depth) - 1
    num_leaves = trial.suggest_int('num_leaves', 3, max_leaves_bound - 1)

    # 每棵树采样比例，候选 [0.8, 0.9, 1.0]
    subsample = trial.suggest_categorical('subsample', [0.8, 0.9, 1.0])

    # 特征采样比例，候选 [0.8, 0.9, 1.0]
    colsample_bytree = trial.suggest_categorical('colsample_bytree', [0.8, 0.9, 1.0])

    # L1 正则项 reg_alpha：按对数尺度在 [1.0, 1000.0] 区间中搜索
    # 这样可以更均匀地覆盖小到大范围的正则强度
    reg_alpha = trial.suggest_float('reg_alpha', 1.0, 1000.0, log=True)

    # L2 正则项 reg_lambda：按对数尺度在 [1.0, 1000.0] 区间中搜索
    reg_lambda = trial.suggest_float('reg_lambda', 1.0, 1000.0, log=True)

    # 2) 将当前 trial 抽样得到的超参数组合整理成 LightGBM 模型参数
    #    这里的 learning_rate 和 n_estimators 是经验固定值，其他参数通过搜索优化
    trial_params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.05,        # 经验确定：较小学习率配合更多树提升稳定性
        'n_estimators': 1000,         # 经验确定：通过 early stopping 控制收敛，不盲目加大树数
        'min_child_samples': 50,      # 叶子节点最小样本数，基于数据量设定，减少极端离群噪声影响
        'max_depth': max_depth,       # 当前 trial 抽样得到的树深度
        'num_leaves': num_leaves,     # 当前 trial 抽样得到的叶子数
        'subsample': subsample,       # 当前 trial 抽样得到的样本采样比例
        'subsample_freq': 1,          # 每次采样频率，表示每轮都进行采样
        'colsample_bytree': colsample_bytree,  # 当前 trial 抽样得到的特征采样比例
        'reg_alpha': reg_alpha,       # 当前 trial 抽样得到的 L1 正则系数
        'reg_lambda': reg_lambda,     # 当前 trial 抽样得到的 L2 正则系数
        'random_state': 123,          # 固定随机种子，保证可复现
        'n_jobs': -1,                 # 使用全部 CPU 资源加速训练
        'verbose': -1                 # 关闭训练过程输出，避免冗余日志
    }

    # 3) 使用 3 折分层交叉验证收集全量 OOF (Out-of-Fold) 预测概率
    #    StratifiedKFold 可以保持每一折正负样本比例一致，适合不平衡数据集
    cv_tune = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
    oof_preds_tune = np.zeros(len(X_final))

    # 对每一折的数据执行训练并在验证折生成预测
    for tr_idx, va_idx in cv_tune.split(X_final, y):
        X_tr_t, y_tr_t = X_final.iloc[tr_idx], y.iloc[tr_idx]
        X_va_t, y_va_t = X_final.iloc[va_idx], y.iloc[va_idx]

        # 当前 iteration 使用当前 trial 参数创建 LightGBM 模型
        model_t = lgb.LGBMClassifier(**trial_params)

        # 训练模型，并使用验证集进行早停
        # early stopping 可以避免过拟合，并节省训练时间
        model_t.fit(
            X_tr_t, y_tr_t,
            eval_set=[(X_va_t, y_va_t)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
        )

        # 记录验证折预测概率至对应的 OOF 位置
        oof_preds_tune[va_idx] = model_t.predict_proba(X_va_t)[:, 1]

    # 4) 基于全量 OOF 预测概率与真实标签统一计算全局 AUC
    #    相比折平均，全局 OOF AUC 额外考察了跨折预测概率尺度的校准与一致性，评估更严谨
    return roc_auc_score(y, oof_preds_tune)

# ---------------------------------------------------------
# 3. 启动 Optuna 超参数搜索
# ---------------------------------------------------------
# 说明：
#   - `study.optimize()` 会自动循环执行 n_trials 次 objective(trial)
#   - 每个 trial 都会产生一组参数组合，并评估其效能
#   - TPE（Tree-structured Parzen Estimator）是一种典型的贝叶斯优化算法
# ---------------------------------------------------------
print("启动 Optuna 超参数搜索 (TPE 算法)...")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 创建优化研究对象：目标是最大化 AUC
# sampler=TPESampler(seed=123) 可以保证搜索过程可复现
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=123))

# 执行 30 次试验（30 个 trial）
# 每次 trial 都会调用一次 objective()，并根据返回值更新搜索策略
study.optimize(objective, n_trials=30, show_progress_bar=False)

print(f"\n超参数搜索完成！搜索阶段最佳 OOF 验证 AUC: {study.best_value:.5f}")
print("搜索获得的最优超参数组合:")
for k, v in study.best_params.items():
    if isinstance(v, float):
        print(f"  - {k}: {v:.4f}")
    else:
        print(f"  - {k}: {v}")

del k, v

# 检查 reg_alpha 与 reg_lambda 合理性
best_alpha = study.best_params['reg_alpha']
best_lambda = study.best_params['reg_lambda']

if best_alpha > 50.0:
    print(f"\n⚠️ [特征提示] 搜索得到的 reg_alpha 较大 ({best_alpha:.2f} > 50)。比较好的模型上 reg_alpha 不应过大，过大说明模型强烈依赖 L1 正则截断弱特征，建议进一步审查特征共线性与噪声特征！")
else:
    print(f"\n✓ [特征检查正常] reg_alpha = {best_alpha:.2f}，处于较小稳健区间。")

if best_lambda > 100.0:
    print(f"⚠️ [模型提示] 搜索得到的 reg_lambda 较大 ({best_lambda:.2f} > 100)，说明叶节点权重存在极端值并受到强 L2 惩罚。")
else:
    print(f"✓ [模型检查正常] reg_lambda = {best_lambda:.2f}，处于平稳区间。")

# 超参数搜索完成！搜索阶段最佳 OOF 验证 AUC: 0.76576
# 搜索获得的最优超参数组合:
#   - max_depth: 4
#   - num_leaves: 11
#   - subsample: 0.9000
#   - colsample_bytree: 0.8000
#   - reg_alpha: 6.6883
#   - reg_lambda: 1.5031

# ✓ [特征检查正常] reg_alpha = 6.69，处于较小稳健区间。
# ✓ [模型检查正常] reg_lambda = 1.50，处于平稳区间。

# %%
# ---------------------------------------------------------
# 2. 组装最终最优超参数并进行 5 折本地交叉验证与 OOF 评估
# ---------------------------------------------------------
final_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,       # 经验确定
    'n_estimators': 1000,        # 经验确定
    'min_child_samples': 50,     # 数据集规模确定
    'max_depth': study.best_params['max_depth'],
    'num_leaves': study.best_params['num_leaves'],
    'subsample': study.best_params['subsample'],
    'subsample_freq': 1,
    'colsample_bytree': study.best_params['colsample_bytree'],
    'reg_alpha': study.best_params['reg_alpha'],
    'reg_lambda': study.best_params['reg_lambda'],
    'random_state': 123,
    'n_jobs': -1,
    'verbose': -1
}

print(f"\n开始使用搜索出的最佳超参数在精筛后的 {len(selected_feature_cols)} 个核心特征上进行 5 折完整训练与本地验证...")

oof_preds = np.zeros(len(application_train))
fold_aucs = []
fold_kss = []

for fold, (train_idx, valid_idx) in enumerate(skf.split(X_final, y), 1):
    X_tr, y_tr = X_final.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X_final.iloc[valid_idx], y.iloc[valid_idx]

    clf = lgb.LGBMClassifier(**final_params)
    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )

    val_pred = clf.predict_proba(X_va)[:, 1]
    oof_preds[valid_idx] = val_pred

    fold_auc = roc_auc_score(y_va, val_pred)
    fold_ks = calc_ks(y_va, val_pred)
    fold_aucs.append(fold_auc)
    fold_kss.append(fold_ks)
    print(f"Fold {fold} | AUC: {fold_auc:.5f} | KS: {fold_ks:.5f}")

cv_auc = roc_auc_score(y, oof_preds)
cv_ks = calc_ks(y, oof_preds)
print(f"各折平均 (Mean +/- Std): AUC = {np.mean(fold_aucs):.5f} (+/- {np.std(fold_aucs):.5f}) | KS = {np.mean(fold_kss):.5f} (+/- {np.std(fold_kss):.5f})")
print(f"本地验证集 (5-Fold CV OOF): AUC = {cv_auc:.5f} | KS = {cv_ks:.5f}")

# 各折平均 (Mean +/- Std): AUC = 0.76677 (+/- 0.00265) | KS = 0.39847 (+/- 0.00390)
# 本地验证集 (5-Fold CV OOF): AUC = 0.76675 | KS = 0.39703

# 清理最终训练与验证过程中的临时变量与特征集副本
del fold, train_idx, valid_idx, X_tr, y_tr, X_va, y_va, clf, val_pred
del X_final, cat_cols_final, skf, params, final_params, y, calc_ks, study, objective
