# %%
# 库
from pathlib import Path

import numpy as np
import pandas as pd

# 文件路径
DATA_PATH = Path().cwd() / 'data'

# 主申请表
application_train = pd.read_csv(DATA_PATH / 'raw' / 'application_train.csv')
application_test = pd.read_csv(DATA_PATH / 'raw' / 'application_test.csv')

# 字段含义
cols_meaning = pd.read_csv(DATA_PATH / 'HomeCredit_columns_description.csv',
                           usecols=['Table', 'Row', 'Description'])

# %%
"一、数据概况"
# 形状
print(application_train.shape)
# (307511, 122)
# application_train 样本量为307511，含有122个字段

print(application_test.shape)
# (48744, 121)
# application_test 样本量为48744，含有121个字段

# 检查测试集的字段是否和训练集一致
print(set(application_test.columns).issubset(set(application_train.columns)))
# True

# 数据类型
cols_train_types = application_train.dtypes
cols_test_types = application_test.dtypes

# 检查测试集字段类型是否和训练集一致
print(cols_train_types.drop('TARGET').equals(cols_test_types))
# True

""" SK_ID_CURR 需要转换为 str, 其余变量不需要转换
 application_train['SK_ID_CURR'] = application_train['SK_ID_CURR'].astype(str) """
del cols_test_types, cols_train_types

# 重复样本，重复主键检查
print(application_train.duplicated().any())
# False
print(application_train.duplicated(subset='SK_ID_CURR').any())
# False
print(application_test.duplicated().any())
# False
# -> 无重复

# %%
"二、目标变量分析"
# label=1 违约；label=0 正常
# 好坏样本数量
print(application_train['TARGET'].value_counts())
# TARGET
# 0    282686
# 1     24825
# Name: count, dtype: int64

# 违约率
print(application_train['TARGET'].mean())
# 0.08072881945686496
# -> 样本存在类别不平衡

# %%
"三、数据质量检查"

# 缺失率
missing_cols_rate = application_train.isna().mean().reset_index(name='missing_rate')
missing_cols_rate = missing_cols_rate[missing_cols_rate['missing_rate'] > 0].reset_index(drop=True)

missing_cols = missing_cols_rate['index'].tolist()

missing_cols_rate.sort_values(by='missing_rate', ascending=False).head(20)
#                        index  missing_rate
# 41           COMMONAREA_MEDI      0.698723
# 13            COMMONAREA_AVG      0.698723
# 27           COMMONAREA_MODE      0.698723
# 49  NONLIVINGAPARTMENTS_MEDI      0.694330
# 35  NONLIVINGAPARTMENTS_MODE      0.694330
# 21   NONLIVINGAPARTMENTS_AVG      0.694330
# 51        FONDKAPREMONT_MODE      0.683862
# 33     LIVINGAPARTMENTS_MODE      0.683550
# 47     LIVINGAPARTMENTS_MEDI      0.683550
# 19      LIVINGAPARTMENTS_AVG      0.683550
# 31            FLOORSMIN_MODE      0.678486
# 45            FLOORSMIN_MEDI      0.678486
# 17             FLOORSMIN_AVG      0.678486
# 26          YEARS_BUILD_MODE      0.664978
# 40          YEARS_BUILD_MEDI      0.664978
# 12           YEARS_BUILD_AVG      0.664978
# 3                OWN_CAR_AGE      0.659908
# 18              LANDAREA_AVG      0.593767
# 46             LANDAREA_MEDI      0.593767
# 32             LANDAREA_MODE      0.593767

# 注意到 COMMONAREA_ NONLIVINGAPARTMENTS_ LIVINGAPARTMENTS_ FLOORSMIN_ 等缺失率相同
# 可能是相同的缺失模式

# 进一步探究缺失模式
same_missing_partten = []
for rate, group in missing_cols_rate.groupby('missing_rate'):
    if len(group) > 1:
        temp_list = group['index'].tolist()
        temp_df = application_train[temp_list]
        if len(temp_df.isna().T.drop_duplicates()) == 1:
            same_missing_partten.append(temp_list)
        else:
            print(f"{temp_list}的缺失模式不同")

del rate, group, temp_df, temp_list
# -> 相同缺失率的字段缺失模式都相同

missing_cols_rate.groupby('missing_rate').head(1).sort_values(by='missing_rate', ascending=False)
#                           index  missing_rate
# 13               COMMONAREA_AVG      0.698723
# 21      NONLIVINGAPARTMENTS_AVG      0.694330
# 51           FONDKAPREMONT_MODE      0.683862
# 19         LIVINGAPARTMENTS_AVG      0.683550
# 17                FLOORSMIN_AVG      0.678486
# 12              YEARS_BUILD_AVG      0.664978
# 3                   OWN_CAR_AGE      0.659908
# 18                 LANDAREA_AVG      0.593767
# 10             BASEMENTAREA_AVG      0.585160
# 6                  EXT_SOURCE_1      0.563811
# 22            NONLIVINGAREA_AVG      0.551792
# 14                ELEVATORS_AVG      0.532960
# 54           WALLSMATERIAL_MODE      0.508408
# 9                APARTMENTS_AVG      0.507497
# 15                ENTRANCES_AVG      0.503488
# 20               LIVINGAREA_AVG      0.501933
# 52               HOUSETYPE_MODE      0.501761
# 16                FLOORSMAX_AVG      0.497608
# 11  YEARS_BEGINEXPLUATATION_AVG      0.487810
# 53               TOTALAREA_MODE      0.482685
# 55          EMERGENCYSTATE_MODE      0.473983
# 4               OCCUPATION_TYPE      0.313455
# 8                  EXT_SOURCE_3      0.198253
# 61   AMT_REQ_CREDIT_BUREAU_HOUR      0.135016
# 2               NAME_TYPE_SUITE      0.004201
# 56     OBS_30_CNT_SOCIAL_CIRCLE      0.003320
# 7                  EXT_SOURCE_2      0.002146
# 1               AMT_GOODS_PRICE      0.000904
# 0                   AMT_ANNUITY      0.000039
# 5               CNT_FAM_MEMBERS      0.000007
# 60       DAYS_LAST_PHONE_CHANGE      0.000003

# 缺失率分析
default_rate_of_missing = []
for col in missing_cols:
    temp_df = application_train[[col, 'TARGET']]
    default_rate = temp_df.loc[temp_df[col].isna(), 'TARGET'].mean()
    default_rate_of_missing.append(default_rate)

default_rate_of_ = []
for col in missing_cols:
    temp_df = application_train[[col, 'TARGET']]
    default_rate = temp_df.loc[temp_df[col].notna(), 'TARGET'].mean()
    default_rate_of_.append(default_rate)

del col, temp_df, default_rate

missing_analysis = pd.DataFrame(
    {'missing_cols': missing_cols,
     'missing_rate': missing_cols_rate['missing_rate'],
     'default_rate_of_missing': default_rate_of_missing,
     'default_rate_of_': default_rate_of_}
)

del default_rate_of_, default_rate_of_missing

missing_analysis['default_rate_diff'] = missing_analysis['default_rate_of_missing'] - \
    missing_analysis['default_rate_of_']

for rate, group in missing_analysis.groupby('missing_rate'):
    if len(group) > 1:
        temp_list = group['missing_cols']
        temp_s = group['default_rate_of_missing']
        if temp_s.max() > temp_s.min():
            print(f"{temp_list}的坏样本率不同")

del temp_list, temp_s, group, rate
# 坏样本率都相同

# 输出待添加 _miss 特征的列
# missing_analysis.to_csv(DATA_PATH / 'interim' / 'missing_cols.csv')

# %%
