# %%
# -- 单箱样本约束 --
# 单箱样本占比原则上 >= 5%（全量样本 307511 时，至少约 15376 个样本）。
# 每个分类或分箱中的坏样本数建议不少于 20~30 个；样本过少时应合并分箱，避免 WOE 估计不稳定。

# -- WOE 分箱质量 --
# 1. 组间差异：相邻分类/分箱的 |ΔWOE| 建议 >= 0.10，严格标准可取 0.15；差异过小时考虑合并。
# 2. 单箱幅度：|WOE| 通常以 0.10~1.50 为较合理范围。
#    - |WOE| 长期接近 0（如 -0.05~+0.05）：该箱与总体违约率接近，区分信息有限。
#    - |WOE| 过大（如 > 2.0，尤其 > 3.0）：检查箱内样本/坏样本是否过少，并考虑合并分箱。
# 3. 趋势与稳定性：有序变量的 WOE 应单调递增或递减；无序变量应检查各箱排序在训练集、测试集及 OOT 样本间是否一致。
#    不应出现训练集与测试集（或跨时间样本）间 WOE 方向反转。

# %%
from pathlib import Path

import numpy as np
import pandas as pd
import toad

from optbinning import BinningProcess
from sklearn.model_selection import train_test_split

# 文件路径
DATA_PATH = Path().cwd() / 'data'

# 主申请表
train_fe = pd.read_csv(DATA_PATH / 'processed' / 'train_fe.csv')

# eda结果
cols_eda = pd.read_csv(DATA_PATH / 'interim' / 'cols_eda.csv')

# train_fe 拆分为X和y
X = train_fe.drop(columns='TARGET').copy()
y = train_fe['TARGET'].copy()

del train_fe

# 在类别特征中，NAME_EDUCATION_TYPE是一个有序类别特征，将其映射为数值特征，强制单调
# 学历从低 1 到高 5
edu_mapping = {
    'Lower secondary': 1,
    'Secondary / secondary special': 2,
    'Incomplete higher': 3,
    'Higher education': 4,
    'Academic degree': 5
}

X['NAME_EDUCATION_TYPE'] = X['NAME_EDUCATION_TYPE'].map(edu_mapping)

del edu_mapping

features = X.columns.tolist()
num_features = X.select_dtypes(include='number').columns.tolist()
cat_features = X.select_dtypes(include='object').columns.tolist()

# %%
# 1.样本切分
# 按y的比例分层随机划分训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=123,
    shuffle=True,
    stratify=y
)

# 2.精细分箱与 WOE 编码
# 数值特征：强制单调 (auto_asc_desc 自动判断单调增或单调减，杜绝 U 型或锯齿波动)
# 类别特征：optbinning 会自动按各分类的 Bad Rate 排序并合并相似风险分类

features = X.columns.tolist()
num_features = X.select_dtypes(include='number').columns.tolist()
cat_features = X.select_dtypes(include='object').columns.tolist()

u_shape_cols = ["AMT_GOODS_PRICE",
                "AMT_ANNUITY",
                "ANNUITY_TO_CREDIT_RATIO",
                "HOUSE_INFO_MISSING_RATE",
                ]

# min_bin_n_event=30：落实注释中“每个分箱坏样本数不少于 20~30 个”的约束
binning_fit_params = {}
for col in num_features:
    if col in u_shape_cols:
        binning_fit_params[col] = {
            "monotonic_trend": "auto",            # 允许1个合理峰值/谷值拐点
            "max_n_bins": 4,                      # 限制4箱以内，防多拐点锯齿
            "min_bin_n_event": 30,                # 单箱至少30个坏样本
        }
    else:
        binning_fit_params[col] = {
            "monotonic_trend": "auto_asc_desc",   # 严格单调递增或递减
            "min_bin_n_event": 30,
        }

for col in cat_features:
    binning_fit_params[col] = {
        "min_bin_n_event": 30,
    }
del col

# 实例化精细分箱器
fine_binning_process = BinningProcess(
    variable_names=features,
    categorical_variables=cat_features,
    
    max_n_prebins=30,                             # 预分箱数提升至 30，提高切点精度
    min_prebin_size=0.02,                         # 预分箱最小占比 2% (需小于 min_bin_size)
    
    min_n_bins=2,
    max_n_bins=5,
    min_bin_size=0.05,
    
    max_pvalue=0.05,
    max_pvalue_policy="consecutive",
    
    binning_fit_params=binning_fit_params,
    selection_criteria=None,
    n_jobs=-1
)

fine_binning_process.fit(X_train, y_train)

binning_summary = fine_binning_process.summary().rename(columns={'name': 'feature'})

# 发现 HOUSETYPE_MODE、EMERGENCYSTATE_MODE 最终求解只有一个分箱
X.HOUSETYPE_MODE.value_counts(dropna=False, normalize=True)
# HOUSETYPE_MODE
# NaN                 0.501761
# block of flats      0.489423
# specific housing    0.004875
# terraced house      0.003941
# Name: proportion, dtype: float64

X.EMERGENCYSTATE_MODE.value_counts(dropna=False, normalize=True)
# EMERGENCYSTATE_MODE
# No     0.518446
# NaN    0.473983
# Yes    0.007570
# Name: proportion, dtype: float64

# 可以发现，HOUSETYPE_MODE、EMERGENCYSTATE_MODE 除缺失与最高频类别外，其余类别总和不足 5%
# 最终只得到正常和缺失两个分箱
# 特征池中已有HOUSE_INFO_MISSING_RATE，故不需要保留这两个房屋特征

def get_woe_summary(column):
    optb = fine_binning_process.get_binned_variable(column)
    bt = optb.binning_table
    df_bin = bt.build()
    
    return df_bin

woe_summary = {}
for col in features:
    woe_summary[col] = get_woe_summary(col)

X_train_woe = pd.DataFrame(
    fine_binning_process.transform(X_train, metric="woe", metric_missing="empirical"),
    columns=features,
    index=X_train.index
)

X_valid_woe = pd.DataFrame(
    fine_binning_process.transform(X_valid, metric="woe", metric_missing="empirical"),
    columns=features,
    index=X_valid.index
)

# 更新移除这两个特征后的X_train_woe和X_valid_woe
X_train_woe = X_train_woe.drop(columns=['HOUSETYPE_MODE', 'EMERGENCYSTATE_MODE'])
X_valid_woe = X_valid_woe.drop(columns=['HOUSETYPE_MODE', 'EMERGENCYSTATE_MODE'])

features = X_train_woe.columns.tolist()
num_features = X_train_woe.select_dtypes(include='number').columns.tolist()
cat_features = X_train_woe.select_dtypes(include='object').columns.tolist()

# %%
# optbinning woe 定义为 ln(good%/bad%)，即好坏比的对数：好客户占比越高，WOE 越大；违约率越高，WOE 越负
# 当预测目标为违约概率P(Y=1)时，所有正常入模特征的逻辑回归系数都必须严格为负，如果某个特征出现了正系数，说明发生了共线性符号反转，必须剔除

# -- WOE相关性筛选 --
binning_summary = binning_summary[~binning_summary['feature'].isin(['HOUSETYPE_MODE', 'EMERGENCYSTATE_MODE'])]

train_woe_corr = X_train_woe.corr()
result = train_woe_corr.stack().reset_index()
result.columns = ['val1', 'val2', 'corr']
result = result[np.abs(result['corr']).between(0.8, 1, inclusive='left')]

# 保留iv值最大的一个特征
iv_dict = dict(zip(binning_summary['feature'], binning_summary['iv']))
result['iv1'] = result['val1'].map(iv_dict)
result['iv2'] = result['val2'].map(iv_dict)

delete_feature = result.loc[result['iv1'] < result['iv2'], 'val1'].tolist()
delete_feature = list(set(delete_feature))
del result

features = [col for col in features if col not in delete_feature]

# -- 