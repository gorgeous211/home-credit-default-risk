# %%
# 库
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest

sns.set_theme(font_scale=0.5)

# 文件路径
DATA_PATH = Path().cwd() / 'data'

# 主申请表
application_train = pd.read_csv(DATA_PATH / 'raw' / 'application_train.csv')
application_test = pd.read_csv(DATA_PATH / 'raw' / 'application_test.csv')

# 字段含义
cols_meaning = pd.read_csv(DATA_PATH / 'HomeCredit_columns_description.csv',
                           usecols=['Table', 'Row', 'Description'])
cols_meaning = cols_meaning[cols_meaning['Table'] == 'application_{train|test}.csv']

# %%
"一、数据概况与数据质量"
# -- 数据规模 --
print(application_train.shape)
# (307511, 122)
# application_train 样本量为307511，含有122个字段

print(application_test.shape)
# (48744, 121)
# application_test 样本量为48744，含有121个字段

# 检查测试集的字段是否和训练集一致
print(set(application_test.columns).issubset(set(application_train.columns)))
# True

# -- 数据类型 --
cols_train_types = application_train.dtypes
cols_test_types = application_test.dtypes

# 检查测试集字段类型是否和训练集一致
print(cols_train_types.drop('TARGET').equals(cols_test_types))
# True

""" SK_ID_CURR 需要转换为 str, 其余变量不需要转换
 application_train['SK_ID_CURR'] = application_train['SK_ID_CURR'].astype(str) """

# -- 重复与唯一性 --
print(application_train.duplicated().any())
# False
print(application_train.duplicated(subset='SK_ID_CURR').any())
# False
print(application_test.duplicated().any())
# False
# -> 无重复

del cols_test_types, cols_train_types

# -- 标签分布 --
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

# -- 缺失率情况 --
missing_summary = application_train.isna().mean().reset_index(name='missing_rate')
missing_summary = missing_summary[missing_summary['missing_rate'] > 0].reset_index(drop=True)

missing_columns = missing_summary['index'].tolist()

missing_summary.sort_values(by='missing_rate', ascending=False).head(20)
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
same_missing_pattern_groups = []

for rate, group in missing_summary.groupby('missing_rate'):
    
    if len(group) > 1:
        temp_list = group['index'].tolist()
        temp_df = application_train[temp_list]
        
        if len(temp_df.isna().T.drop_duplicates()) == 1:
            same_missing_pattern_groups.append(temp_list)
        else:
            print(f"{temp_list}的缺失模式不同")

del rate, group, temp_df, temp_list

# -> 相同缺失率的字段缺失模式都相同

missing_summary.groupby('missing_rate').head(1).sort_values(by='missing_rate', ascending=False)
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

missing_summary = missing_summary.merge(
    cols_meaning.drop(columns='Table'),
    left_on='index',
    right_on='Row',
    how='left')

missing_summary = missing_summary.drop(columns='index')

# -> 缺失率较高的字段大部分是与房屋相关
# -> 说明一部分申请人缺少相关房屋信息，而不是数据随机缺失

# 去除房屋相关字段后，其余高缺失率变量
missing_summary.loc[
    ~missing_summary['Description'].str.startswith('Normalized'),
    ['Row', 'missing_rate']
].sort_values(by='missing_rate', ascending=False)

#                            Row  missing_rate
# 3                  OWN_CAR_AGE      0.659908
# 4              OCCUPATION_TYPE      0.313455
# 66  AMT_REQ_CREDIT_BUREAU_YEAR      0.135016
# 61  AMT_REQ_CREDIT_BUREAU_HOUR      0.135016
# 65   AMT_REQ_CREDIT_BUREAU_QRT      0.135016
# 64   AMT_REQ_CREDIT_BUREAU_MON      0.135016
# 63  AMT_REQ_CREDIT_BUREAU_WEEK      0.135016
# 62   AMT_REQ_CREDIT_BUREAU_DAY      0.135016
# 2              NAME_TYPE_SUITE      0.004201
# 58    OBS_60_CNT_SOCIAL_CIRCLE      0.003320
# 59    DEF_60_CNT_SOCIAL_CIRCLE      0.003320
# 57    DEF_30_CNT_SOCIAL_CIRCLE      0.003320
# 56    OBS_30_CNT_SOCIAL_CIRCLE      0.003320
# 1              AMT_GOODS_PRICE      0.000904
# 0                  AMT_ANNUITY      0.000039
# 5              CNT_FAM_MEMBERS      0.000007
# 60      DAYS_LAST_PHONE_CHANGE      0.000003

# 输出缺失率排名图
sns.barplot(
    data=missing_summary.sort_values(by='missing_rate', ascending=False).head(30),
    x='missing_rate',
    y='missing_cols')

sns.barplot(
    data=missing_summary.loc[
        ~missing_summary['Description'].str.startswith('Normalized'),
        ['Row', 'missing_rate']
        ].sort_values(by='missing_rate', ascending=False),
    x='missing_rate',
    y='Row')

# 缺失率分析
default_rate_when_missing = []     # 缺失组违约率
missing_numbers_when_missing = []  # 缺失组样本量
default_numbers_when_missing = []  # 缺失组违约样本率

for col in missing_columns:
    temp_df = application_train[[col, 'TARGET']]
    
    default_rate = temp_df.loc[temp_df[col].isna(), 'TARGET'].mean()
    missing_number = len(temp_df[temp_df[col].isna()])
    default_number = temp_df.loc[temp_df[col].isna(), 'TARGET'].sum()
    
    default_rate_when_missing.append(default_rate)
    missing_numbers_when_missing.append(missing_number)
    default_numbers_when_missing.append(default_number)

default_rate_when_present = []     # 正常组违约率
missing_numbers_when_present = []  # 正常组样本量
default_numbers_when_present = []  # 正常组违约样本量

for col in missing_columns:
    temp_df = application_train[[col, 'TARGET']]
    
    default_rate = temp_df.loc[temp_df[col].notna(), 'TARGET'].mean()
    missing_number = len(temp_df[temp_df[col].notna()])
    default_number = temp_df.loc[temp_df[col].notna(), 'TARGET'].sum()
    
    default_rate_when_present.append(default_rate)
    missing_numbers_when_present.append(missing_number)
    default_numbers_when_present.append(default_number)

del col, temp_df, default_rate, default_number

missing_impact_summary = pd.DataFrame(
    {'missing_cols': missing_columns,
     'missing_rate': missing_summary['missing_rate'],
     
     'default_rate_when_missing': default_rate_when_missing,
     'missing_numbers_when_missing': missing_numbers_when_missing,
     'default_numbers_when_missing': default_numbers_when_missing,
     
     'default_rate_when_present': default_rate_when_present,
     'missing_numbers_when_present': missing_numbers_when_present,
     'default_numbers_when_present': default_numbers_when_present,
     
     'default_rate_diff': [default_rate_when_missing[i] -
                           default_rate_when_present[i]
                           for i in range(len(default_rate_when_missing))]}
)

del default_rate_when_missing, missing_numbers_when_missing, default_numbers_when_missing,\
    default_rate_when_present, missing_numbers_when_present, default_numbers_when_present

temp_list = ['missing_rate',
             'default_rate_when_missing',
             'default_rate_when_present',
             'default_rate_diff']
missing_impact_summary[temp_list] = missing_impact_summary[temp_list].map(lambda x: f"{x:.2%}")
# map对元素逐项变化，apply对series变化

del temp_list

for rate, group in missing_impact_summary.groupby('missing_rate'):
    if len(group) > 1:
        temp_list = group['missing_cols']
        temp_s = group['default_rate_when_missing']
        if temp_s.max() > temp_s.min():
            print(f"{temp_list}的坏样本率不同")

del temp_list, temp_s, group, rate
# 坏样本率都相同

del missing_columns, missing_summary

# 进一步分析缺失状态与 TARGET 的关系
# 检验缺失组的违约率与非缺失组的违约率差异是否显著，设显著性水平为0.05
# 两样本比例之差的假设检验

def run_ztest_row(row):
    col_default_n_missing = 'default_numbers_when_missing'
    col_n_missing = 'missing_numbers_when_missing'
    col_default_n_present = 'default_numbers_when_present'
    col_n_present = 'missing_numbers_when_present'
    
    n_default_miss = row[col_default_n_missing]
    n_default_pres = row[col_default_n_present]
    
    n_miss = row[col_n_missing]
    n_pres = row[col_n_present]
    
    stat, pval = proportions_ztest(
        count=[n_default_miss, n_default_pres],
        nobs=[n_miss, n_pres],
        alternative='two-sided')
    
    return pd.Series({'z_stat': stat, 'p_value': pval})

missing_impact_summary[['z_stat', 'p_value']] = missing_impact_summary.apply(run_ztest_row, axis=1)

# 检验结果除
# 	missing_cols	missing_rate
# 7	EXT_SOURCE_2	0.21%
# 60	DAYS_LAST_PHONE_CHANGE	0.00%
# 1	AMT_GOODS_PRICE	0.09%
# 5	CNT_FAM_MEMBERS	0.00%
# 0	AMT_ANNUITY	0.00%

# 以外，差值均非常显著，说明缺失对于违约是一个强信号
# 后续特征工程添加显著列 _miss 特征
# 非显著列采用中位数填充或者增加missing类

# 对于房屋相关字段，缺失模式相同，考虑后续特征工程只添加一个_miss变量，避免信息重复

# 输出缺失值分析的结果
missing_impact_summary.to_csv(DATA_PATH / 'interim' / 'missing_impact_summary.csv')
