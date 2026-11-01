# %%
# 逻辑回归评分卡通常只保留 10~20 个核心特征。
# 如果塞入大量弱特征，不仅无法提升模型的 KS 和 AUC，反而会引入噪声、增加共线性风险，
# 导致逻辑回归权重系数（Coefficient）失真甚至符号反转。

# %%
# 导入库
from pathlib import Path

import numpy as np
import pandas as pd
import toad
from optbinning import BinningProcess

# 文件路径
DATA_PATH = Path().cwd() / 'data'

# 主申请表
application_train = pd.read_csv(DATA_PATH / 'raw' / 'application_train.csv')
application_test = pd.read_csv(DATA_PATH / 'raw' / 'application_test.csv')

# eda结果
cols_eda = pd.read_csv(DATA_PATH / 'interim' / 'cols_eda.csv')
model_features_summary = pd.read_csv(DATA_PATH / 'interim' / 'model_features_summary.csv')

# %%
model_features_summary = model_features_summary[['Row', 'Description','iv', 'psi']]

house_cols = cols_eda.loc[
    cols_eda['Description'].str.startswith('Normalized information'), 'Row'
    ].tolist()

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
train_fe = application_train[model_features_summary['Row'].tolist() + ['TARGET']].copy()
test_fe = application_test[model_features_summary['Row'].tolist()].copy()

# %%
# == 构造衍生特征 ==
def add_derived_features(df_raw, df_fe, house_cols):
    # -- 还款能力与负债压力 --
    df_fe['CREDIT_TO_INCOME_RATIO'] = \
        df_raw['AMT_CREDIT'] / df_raw['AMT_INCOME_TOTAL']
    df_fe['ANNUITY_TO_INCOME_RATIO'] = \
        df_raw['AMT_ANNUITY'] / df_raw['AMT_INCOME_TOTAL']
    df_fe['FREE_INCOME_PROXY'] = \
        df_raw['AMT_INCOME_TOTAL'] - df_raw['AMT_ANNUITY']
    df_fe['ANNUITY_TO_CREDIT_RATIO'] = \
        df_raw['AMT_ANNUITY'] / df_raw['AMT_CREDIT']
    df_fe['GOODS_TO_CREDIT_RATIO'] = \
        df_raw['AMT_GOODS_PRICE'] / df_raw['AMT_CREDIT']
    df_fe['DOWN_PAYMENT_PROXY'] = \
        df_raw['AMT_GOODS_PRICE'] - df_raw['AMT_CREDIT']
    df_fe['INCOME_PER_PERSON'] = \
        df_raw['AMT_INCOME_TOTAL'] / df_raw['CNT_FAM_MEMBERS']
    df_fe['INCOME_PER_CHILD'] = \
        df_raw['AMT_INCOME_TOTAL'] / (1 + df_raw['CNT_CHILDREN'])

    # -- 稳定性与年龄交叉比率 --
    df_fe['EMPLOYED_TO_BIRTH_RATIO'] = \
        df_raw['DAYS_EMPLOYED'] / df_raw['DAYS_BIRTH']
    df_fe['CAR_TO_BIRTH_RATIO'] = \
        df_raw['OWN_CAR_AGE'] / (-df_raw['DAYS_BIRTH'] / 365)
    df_fe['ID_TO_BIRTH_RATIO'] = \
        df_raw['DAYS_ID_PUBLISH'] / df_raw['DAYS_BIRTH']

    # -- 外部征信评分 --
    df_fe['EXT_SOURCE_MEAN'] = \
        df_raw[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)

    # -- 风险信号聚合与同源缺失标志位 --
    df_fe['HOUSE_INFO_MISSING_RATE'] = \
        df_raw[[
            col for col in house_cols
            if col.endswith('AVG')
        ]].isna().mean(axis=1)
    df_fe['DOCUMENTS_PROVIDED_CNT'] = \
        df_raw[[
            f"FLAG_DOCUMENT_{i}"
            for i in range(2, 22)
        ]].sum(axis=1)
    df_fe['ADDRESS_MISMATCH_MEAN'] = \
        df_raw[[
            'REG_CITY_NOT_LIVE_CITY',
            'REG_CITY_NOT_WORK_CITY',
            'LIVE_CITY_NOT_WORK_CITY',
            'REG_REGION_NOT_LIVE_REGION',
            'REG_REGION_NOT_WORK_REGION',
            'LIVE_REGION_NOT_WORK_REGION'
        ]].mean(axis=1)
    df_fe['DEF_OBS_RATIO_30'] = \
        df_raw['DEF_30_CNT_SOCIAL_CIRCLE'] / df_raw['OBS_30_CNT_SOCIAL_CIRCLE'].replace(0, np.nan)
    df_fe['DEF_OBS_RATIO_60'] = \
        df_raw['DEF_60_CNT_SOCIAL_CIRCLE'] / df_raw['OBS_60_CNT_SOCIAL_CIRCLE'].replace(0, np.nan)


add_derived_features(application_train, train_fe, house_cols)
add_derived_features(application_test, test_fe, house_cols)

del application_train, application_test, house_cols

# %%
# 对17个衍生特征进行分箱，计算iv和psi
feature_cols = train_fe.columns.tolist()[-17:]

# IV 值计算
binning_fit_params = {
    col: {"monotonic_trend": "auto"} 
    for col in feature_cols
}

binning_process = BinningProcess(
    variable_names=feature_cols,
    max_n_bins=8,
    min_bin_size=0.02,
    binning_fit_params=binning_fit_params,
    selection_criteria=None,
    n_jobs=-1
)

binning_process.fit(
    train_fe[feature_cols],
    train_fe['TARGET']
)

iv_summary = binning_process.summary().rename(columns={'name': 'feature'})

iv_summary = iv_summary.sort_values(by='iv', ascending=False).reset_index(drop=True)

# PSI 计算
train_bins = pd.DataFrame(
    binning_process.transform(
        train_fe[feature_cols],
        metric="bins"
    ),
    columns=feature_cols,
)

test_bins = pd.DataFrame(
    binning_process.transform(
        test_fe[feature_cols],
        metric="bins"
    ),
    columns=feature_cols,
)

psi_summary = toad.metrics.PSI(
    test_bins[feature_cols],
    train_bins[feature_cols]
).sort_values(ascending=False)

low_iv = iv_summary.loc[iv_summary['iv'] < 0.02, 'feature'].tolist()
high_psi = psi_summary[psi_summary > 0.25].index.tolist()

# 筛选后的衍生特征
feature_cols = [col for col in feature_cols
                if col not in low_iv
                if col not in high_psi]

delete_cols = list(set(low_iv + high_psi))

del train_bins, test_bins, test_fe, low_iv, high_psi

# %%
# 对特征以及筛选后的衍生特征进行相关性分析
# 计算Pearson相关系数矩阵
train_fe = train_fe.drop(columns=delete_cols)
train_corr = train_fe.select_dtypes(include='number').corr()

mask = np.triu(np.ones(train_corr.shape), k=1).astype(bool)
result = train_corr.where(mask).stack()
result[np.abs(result) > 0.8]
# DAYS_EMPLOYED           EMPLOYED_TO_BIRTH_RATIO   -0.954679
# DAYS_ID_PUBLISH         ID_TO_BIRTH_RATIO         -0.843506
# OWN_CAR_AGE             CAR_TO_BIRTH_RATIO         0.929369
# REG_CITY_NOT_WORK_CITY  ADDRESS_MISMATCH_MEAN      0.887885

# 找出这四组特征的iv值
cols_eda.loc[cols_eda['Row'] == 'DAYS_EMPLOYED', 'iv']  # 0.113994
iv_summary.loc[iv_summary['feature'] == 'EMPLOYED_TO_BIRTH_RATIO', 'iv']  # 0.09272
# 保留DAYS_EMPLOYED，删除EMPLOYED_TO_BIRTH_RATIO

cols_eda.loc[cols_eda['Row'] == 'DAYS_ID_PUBLISH', 'iv']  # 0.040167
iv_summary.loc[iv_summary['feature'] == 'ID_TO_BIRTH_RATIO', 'iv']  #  0.023977
# 保留DAYS_ID_PUBLISH，删除ID_TO_BIRTH_RATIO

cols_eda.loc[cols_eda['Row'] == 'OWN_CAR_AGE', 'iv']  # 0.022061
iv_summary.loc[iv_summary['feature'] == 'CAR_TO_BIRTH_RATIO', 'iv']  # 0.030711
# 保留CAR_TO_BIRTH_RATIO，删除OWN_CAR_AGE

cols_eda.loc[cols_eda['Row'] == 'REG_CITY_NOT_WORK_CITY', 'iv']  # 0.032207
iv_summary.loc[iv_summary['feature'] == 'ADDRESS_MISMATCH_MEAN', 'iv']  # 0.028941
# 保留REG_CITY_NOT_WORK_CITY，删除ADDRESS_MISMATCH_MEAN

# 结果同步到 train_fe
train_fe = train_fe.drop(columns=['EMPLOYED_TO_BIRTH_RATIO',
                                  'ID_TO_BIRTH_RATIO',
                                  'OWN_CAR_AGE',
                                  'ADDRESS_MISMATCH_MEAN'])
feature_cols = train_fe.columns.tolist()

del mask, result, train_corr
del delete_cols

# %%
# 输出特征工程加工结果
train_fe.to_csv(DATA_PATH / 'processed' / 'train_fe.csv', index=False)
