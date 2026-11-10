# %%
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold


# 文件路径
DATA_PATH = Path.cwd() / "data"

# 主申请表
application_train = pd.read_csv(DATA_PATH / 'raw' / 'application_train.csv')
application_test = pd.read_csv(DATA_PATH / 'raw' / 'application_test.csv')

cols_eda = pd.read_csv(DATA_PATH / 'interim' / 'cols_eda.csv')

# %%
# -- 特殊异常值替换 --
# DAYS_EMPLOYED：将 365243 替换为 np.nan
application_train['DAYS_EMPLOYED'] = application_train['DAYS_EMPLOYED'].replace(365243, np.nan)

# CODE_GENDER：将 4 个 'XNA' 样本替换为众数 'F'
application_train['CODE_GENDER'] = application_train['CODE_GENDER'].replace('XNA', 'F')

# OBS_30_CNT_SOCIAL_CIRCLE、OBS_60_CNT_SOCIAL_CIRCLE 将30以上的值替换为np.nan
# 这两个字段的 50% 分位数是 0，75% 分位数也才仅仅是 2，均值只有 1.42
# 突然出现某个人有344和348个社交圈关联人，从信贷业务常识来看几乎不可能是真实的人际社交圈，大概率数据错误
application_train.loc[
    application_train['OBS_30_CNT_SOCIAL_CIRCLE'] > 30, 'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan

application_train.loc[
    application_train['OBS_60_CNT_SOCIAL_CIRCLE'] > 30, 'OBS_60_CNT_SOCIAL_CIRCLE'] = np.nan

application_test['DAYS_EMPLOYED'] = application_test['DAYS_EMPLOYED'].replace(365243, np.nan)
application_test['CODE_GENDER'] = application_test['CODE_GENDER'].replace('XNA', 'F')
application_test.loc[
    application_test['OBS_30_CNT_SOCIAL_CIRCLE'] > 30, 'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan
application_test.loc[
    application_test['OBS_60_CNT_SOCIAL_CIRCLE'] > 30, 'OBS_60_CNT_SOCIAL_CIRCLE'] = np.nan

# %%
# -- 特征衍生 --


# %%
ignore_cols = ['SK_ID_CURR', 'TARGET']
all_feature_cols = [c for c in application_train.columns if c not in ignore_cols]
all_num_cols = [c for c in application_train.select_dtypes(include='number').columns if c not in ignore_cols]

# -- 粗筛 --
# 1.基础清洗
# 剔除极高同值特征，同值率大于0.95
high_primary_value_ratio = cols_eda.loc[
    cols_eda['primary_value_ratio'] > 0.95, 'Row'
].tolist()

# 2.稳定性过滤
# 剔除在训练集和测试集/时间外样本上分布漂移严重的特征，psi > 0.25
high_psi = cols_eda.loc[cols_eda['psi'] >= 0.25, 'Row'].tolist()

# 3.多重共线性
# 进入第 3 步的所有特征，都已经通过了质量和稳定性检验
# Pearson相关系数只能捕捉线性关系
# 而树模型对单调非线性变换不敏感，采用Spearman相关系数衡量特征间的单调相关性
# lightgbm 对共线性的容忍度远高于逻辑回归，阈值设置为0.95
features_corr = application_train[[
    col for col in all_num_cols
    if col not in high_primary_value_ratio
    and col not in high_psi
]].corr(method='spearman')

# 筛选出相关系数大于0.95的特征
result = features_corr.stack().reset_index()
result.columns = ['feature1', 'feature2', 'corr']
result = result[np.abs(result['corr']).between(0.95, 1, inclusive='left')]
# 保留psi较小者
psi_dict = dict(zip(cols_eda['Row'], cols_eda['psi']))
result['psi1'] = result['feature1'].map(psi_dict)
result['psi2'] = result['feature2'].map(psi_dict)
high_corr = result.loc[result['psi1'] > result['psi2'], 'feature1'].tolist()

# 三种情况一共需要删除的特征
delete_feature = set(high_primary_value_ratio + high_psi + high_corr)
# 一共需要删除54个特征

# 初筛后的特征
candidate_feature_cols = [col for col in all_feature_cols
                        if col not in delete_feature]
del all_num_cols
del high_primary_value_ratio
del high_psi
del features_corr, result, psi_dict, high_corr
del delete_feature

# %%
# -- 精筛 --
# 使用五折交叉验证汇总 Gain，降低单次切分带来的重要性波动。
# 重要性在每折内归一化后再取平均，避免不同折的树数/增益尺度影响比较。
X = application_train[candidate_feature_cols].copy()
y = application_train['TARGET']

# LightGBM可直接处理缺失值；将字符串列转换为category类型。
categorical_cols = X.select_dtypes(include=['object', 'category']).columns
X[categorical_cols] = X[categorical_cols].astype('category')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
gain_importance = np.zeros(X.shape[1], dtype=float)

params = { 
    # 复杂度控制参数
    # 浅树更倾向于捕捉全局性特征分布差异，避免深树在叶节点上对个别样本强行切分
    'max_depth': 5,             # 默认-1，树的最大深度， 一般在3、4、5这三个数里挑一个
    'num_leaves': 16,           # 默认31，树的最大叶子数，取<=2^max_depth-1=2^5-1=31
    'min_child_samples': 50,    # 默认20，单个叶子节点上所需的最小样本数
                                # 调大该参数，防止叶节点样本数过少导致的局部过拟合
                                
    # 采样与扰动参数
    'subsample': 0.8,           # 默认1.0，行采样比例（样本子集比例）
    'subsample_freq': 1,        # 默认0，行采样频率。0 表示不启用 bagging；若设为k，则每k轮迭代采样一次
    'colsample_bytree': 0.8,    # 默认1.0，列采样比例，每棵树构建时采样的特征比例
                                # 调小该参数，防止少数强漂移特征在每棵树中都占据绝对主导地位
                                # 强制模型去探索其他次要特征的漂移情况
    
    # 正则化与迭代参数
    'reg_alpha': 1.0,           # 默认0.0，L1正则化系数
    'reg_lambda': 1.0,          # 默认0.0，L2正则化系数
    'learning_rate': 0.05,      # 默认0.1，调小该参数提升模型上限与泛化能力
    'n_estimators': 1000,       # 默认100，迭代轮数（基学习器的最大数量），
                                # 由于调小learning_rate并且加入了早停机制，调大该参数
    # 任务与目标参数
    'objective': 'binary',      # 默认None，binary代表二分类任务
    'metric': 'auc',            # 确保早停锁定在区分度最高的那一轮
    'class_weight': None,       # 默认None，去掉可防少数类噪声放大特征 Gain
    'random_state': 123,        # 默认None，随机种子
    'importance_type': 'gain',  # 默认'split'，特征重要性计算方式，'gain'表示按信息增益计算
    'n_jobs': -1,               # 默认None，-1代表使用全部CPU核心
}

for train_idx, valid_idx in skf.split(X, y):
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

# 计算累计的Gain，只保留重要性达到0.95的特征
importance_df['cumulative_gain'] = importance_df['gain_importance'].cumsum()
# 筛选后的特征
candidate_feature_cols = importance_df.loc[
    importance_df['cumulative_gain'] > 0.95, 'feature'
].tolist()

# %%


# %%


