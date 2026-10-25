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

"数据规模、目标变量、缺失值、重复值、异常值、训练集和测试集差异。"

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

"SK_ID_CURR 需要转换为 str, 其余变量不需要转换"
application_train['SK_ID_CURR'] = application_train['SK_ID_CURR'].astype(str) 

# -- 重复与唯一性 --
print(application_train.duplicated().any())
# False
print(application_train.duplicated(subset='SK_ID_CURR').any())
# False
print(application_test.duplicated().any())
# False
# -> 无重复

del cols_test_types, cols_train_types

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

del rate, group, temp_df, temp_list, same_missing_pattern_groups

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

del col, temp_df, default_rate, default_number, missing_number 

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
del missing_impact_summary

# -- 异常值分析 --
# 针对连续性数值变量
def get_outlier(s):
    Q_1 = s.quantile(0.25)
    Q_3 = s.quantile(0.75)
    IQR = Q_3 - Q_1

    outlier = s[(s < Q_1 - 1.5 * IQR)
                | (s > Q_3 + 1.5 * IQR)]
    return outlier

nm_df = application_train.select_dtypes(include='float')
outliers_dict = {col: get_outlier(nm_df[col])
                 for col in nm_df.columns}

del nm_df

# 从 outliers_dict 中数量最多的开始分析

# %%
"二、目标变量分析"

"违约率、类别不平衡、不同特征下的违约率变化。"

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
"三、单变量分布分析"

"年龄、收入、贷款金额、工作年限、家庭情况等变量的分布。"

'''

第一类：二元变量（0/1）
1.看 0 和 1 各有多少条记录。
2.比较 0 和 1 两组的违约率。
3.检查是否有类别极少，导致违约率不可靠的情况。

第二类：连续数值变量
1.看描述性统计，了解取值范围和分散程度。
2.看直方图，了解数据集中在哪里、是否偏态。
3.检查异常值和特殊编码。
4.再分析不同数值范围对应的违约率。

第三类：多类别变量
1.看各类别的样本数和占比。
2.比较各类别的违约率。
3.检查稀有类别和缺失值。

'''
# 筛选 int 类型的字段：包含0/1分类变量，离散数值变量
discrete_values_columns = application_train.select_dtypes(include='int').columns

discrete_values_frequency = {}
for col in discrete_values_columns:
    discrete_values_frequency[col] = application_train[col].value_counts()

del col
# DAYS_EMPLOYED 存在异常编码 55374

# 存在取值极度稀有情况
# FLAG_MOBIL	count
# 1	307510
# 0	1
# FLAG_DOCUMENT_2	count
# 0	307498
# 1	13
# FLAG_DOCUMENT_4	count
# 0	307486
# 1	25

# 重点检查 FLAG_DOCUMENT_ 的取值情况（0/1）
FLAG_frequency = pd.DataFrame({
    'columns': [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)],
    'frequency': [f"{application_train[f'FLAG_DOCUMENT_{i}'].mean():.4%}"
                  for i in range(2, 22)]
})

# 统计每一列取1的频率
# 	columns	frequency
# 0	FLAG_DOCUMENT_2	0.0042%
# 1	FLAG_DOCUMENT_3	71.0023%
# 2	FLAG_DOCUMENT_4	0.0081%
# 3	FLAG_DOCUMENT_5	1.5115%
# 4	FLAG_DOCUMENT_6	8.8055%
# 5	FLAG_DOCUMENT_7	0.0192%
# 6	FLAG_DOCUMENT_8	8.1376%
# 7	FLAG_DOCUMENT_9	0.3896%
# 8	FLAG_DOCUMENT_10	0.0023%
# 9	FLAG_DOCUMENT_11	0.3912%
# 10	FLAG_DOCUMENT_12	0.0007%
# 11	FLAG_DOCUMENT_13	0.3525%
# 12	FLAG_DOCUMENT_14	0.2936%
# 13	FLAG_DOCUMENT_15	0.1210%
# 14	FLAG_DOCUMENT_16	0.9928%
# 15	FLAG_DOCUMENT_17	0.0267%
# 16	FLAG_DOCUMENT_18	0.8130%
# 17	FLAG_DOCUMENT_19	0.0595%
# 18	FLAG_DOCUMENT_20	0.0507%
# 19	FLAG_DOCUMENT_21	0.0335%

# 小于0.1%：FLAG_DOCUMENT_2、4、7、10、12、17、19、20、21
# 0.1%到2%：FLAG_DOCUMENT_5、9、11、13、14、15、16、18
# 大于2%：FLAG_DOCUMENT_3、6、8

# 对FLAG_DOCUMENT_3、6、8稳定的类别变量进一步分析不同类别违约率大小
for col in ['FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8']:
    default_rate_1 = application_train.loc[application_train[col] == 1, 'TARGET'].mean()
    default_rate_0 = application_train.loc[application_train[col] == 0, 'TARGET'].mean()
    print(f"\n{col}取值为1时，违约率：{default_rate_1:.2%}")
    print(f"{col}取值为0时，违约率：{default_rate_0:.2%}")

# FLAG_DOCUMENT_3取值为1时，违约率：8.84%
# FLAG_DOCUMENT_3取值为0时，违约率：6.18%

# FLAG_DOCUMENT_6取值为1时，违约率：5.57%
# FLAG_DOCUMENT_6取值为0时，违约率：8.31%

# FLAG_DOCUMENT_8取值为1时，违约率：7.34%
# FLAG_DOCUMENT_8取值为0时，违约率：8.14%

# 存在明显差异，可能提供违约信息，在特征工程重点考虑

# 小于0.1%的，变量不稳定，暂时保留
# 对于0.1%到2%的，特征工程考虑构建一个组合变量

del FLAG_frequency, col, default_rate_1, default_rate_0

# 接着分析 FLAG_EMP_PHONE 等变量
for col in ['FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']:
    default_rate_1 = application_train.loc[application_train[col] == 1, 'TARGET'].mean()
    default_rate_0 = application_train.loc[application_train[col] == 0, 'TARGET'].mean()
    print(f"\n{col}取值为1时，违约率：{default_rate_1:.2%}")
    print(f"{col}取值为0时，违约率：{default_rate_0:.2%}")

del col, default_rate_1, default_rate_0
# FLAG_EMP_PHONE取值为1时，违约率：8.66%
# FLAG_EMP_PHONE取值为0时，违约率：5.40%

# FLAG_WORK_PHONE取值为1时，违约率：9.63%
# FLAG_WORK_PHONE取值为0时，违约率：7.69%

# FLAG_CONT_MOBILE取值为1时，违约率：8.07%
# FLAG_CONT_MOBILE取值为0时，违约率：7.84%

# FLAG_PHONE取值为1时，违约率：7.04%
# FLAG_PHONE取值为0时，违约率：8.48%

# FLAG_EMAIL取值为1时，违约率：7.88%
# FLAG_EMAIL取值为0时，违约率：8.08%

# 以上是对0/1编码后的分类变量的分析
# 下面是对离散数值变量进行分析

# 重点分析 DAYS_ 类离散数值变量
# 描述性统计与绘制直方图
(-application_train['DAYS_BIRTH'] / 365).describe()
# count    307511.000000
# mean         43.936973
# std          11.956133
# min          20.517808
# 25%          34.008219
# 50%          43.150685
# 75%          53.923288
# max          69.120548
# Name: DAYS_BIRTH, dtype: float64

sns.histplot(x=-discrete_values_frequency['DAYS_BIRTH'].index / 365,
             weights=discrete_values_frequency['DAYS_BIRTH'].values,
             binwidth=1,
             kde=True)

application_train['DAYS_EMPLOYED'] = application_train['DAYS_EMPLOYED'].replace(365243, np.nan)
(-application_train['DAYS_EMPLOYED'] / 365).describe()
# count    252137.000000
# mean          6.531971
# std           6.406466
# min          -0.000000
# 25%           2.101370
# 50%           4.515068
# 75%           8.698630
# max          49.073973
# Name: DAYS_EMPLOYED, dtype: float64

# 删除 365243
sns.histplot(x=-discrete_values_frequency['DAYS_EMPLOYED'].drop(365243).index / 365,
             weights=discrete_values_frequency['DAYS_EMPLOYED'].drop(365243).values,
             binwidth=1,
             kde=True)
# 图像为典型的右偏分布

(-application_train['DAYS_ID_PUBLISH'] / 365).describe()
# count    307511.000000
# mean          8.203294
# std           4.135481
# min           0.000000
# 25%           4.712329
# 50%           8.915068
# 75%          11.778082
# max          19.717808
# Name: DAYS_ID_PUBLISH, dtype: float64

sns.histplot(x=-discrete_values_frequency['DAYS_ID_PUBLISH'].index / 365,
             weights=discrete_values_frequency['DAYS_ID_PUBLISH'].values,
             binwidth=1,
             kde=True)

# 进行分箱判断与 TARGET 的关系
# 分箱方法有业务含义分箱、等宽分箱、等频分箱

# DAYS_BIRTH 取值范围为20~70，按业务含义分箱，20~29，30~39一个箱子
bins_birth = [20, 30, 40, 50, 60, np.inf]
labels_birth = ['20~29岁', '30~39岁', '40~49岁', '50岁~59岁', '60岁以上']

birth_bins = pd.DataFrame({'bins_birth': labels_birth})
application_train['birth_bins'] = pd.cut(
    -application_train['DAYS_BIRTH'] / 365,
    bins=bins_birth,
    labels=labels_birth,
    right=False)

# 统计每个箱子的违约率
birth_bins['default_rate'] = birth_bins['bins_birth'].map(
    application_train.groupby('birth_bins')['TARGET'].mean().map(lambda x: f"{x:.2%}")
)
birth_bins['population'] = birth_bins['bins_birth'].map(
    application_train.groupby('birth_bins').size()
)
birth_bins['pop_default'] = birth_bins['bins_birth'].map(
    application_train.groupby('birth_bins')['TARGET'].sum()
)

# 	bins_birth	default_rate	population	pop_default
# 0	20~29岁	11.46%	45000	5155
# 1	30~39岁	9.58%	82299	7885
# 2	40~49岁	7.66%	76543	5860
# 3	50岁~59岁	6.13%	68074	4173
# 4	60岁以上	4.92%	35595	1752

# 各组的样本量充足，违约率和年龄是单调递减的关系

del bins_birth, labels_birth, birth_bins

# DAYS_EMPLOYED 将特殊编码365243单独一个箱子，采用业务含义分箱，
bins_employed = [0, 1, 3, 5, 10, np.inf]
labels_employed = ['少于1年', '1~3年', '3~5年', '5~10年', '10年以上']

employed_bins = pd.DataFrame({'employed_bins': labels_employed + ['特殊编码组']})
application_train['employed_bins'] = pd.cut(
    (-application_train['DAYS_EMPLOYED'] / 365),
    bins=bins_employed,
    labels=labels_employed,
    right=False)

application_train['employed_bins'] = \
    application_train['employed_bins'].cat.add_categories(['特殊编码组']).fillna('特殊编码组')

# 统计每个分箱的违约率
employed_bins['default_rate'] = employed_bins['employed_bins'].map(
    application_train.groupby('employed_bins')['TARGET'].mean().map(lambda x: f"{x:.2%}")
)
employed_bins['population'] = employed_bins['employed_bins'].map(
    application_train.groupby('employed_bins').size()
)
employed_bins['pop_default'] = employed_bins['employed_bins'].map(
    application_train.groupby('employed_bins')['TARGET'].sum()
)

# 	employed_bins	default_rate	population	pop_default
# 0	少于1年	10.98%	27904	3065
# 1	1~3年	11.07%	61489	6805
# 2	3~5年	9.67%	46874	4535
# 3	5~10年	7.37%	64890	4785
# 4	10年以上	5.19%	50980	2645
# 5	特殊编码组	5.40%	55374	2990

# 各组的样本量充足，可以看出随着工作年限增长，违约率是下降的，但是不是一个严格的线性关系

del bins_employed, labels_employed, employed_bins

# 最后分析 DAYS_ID_PUBLISH
# 采用业务含义分箱
bins_publish = [0, 2, 5, 10, 15, np.inf]
label_publish = ['0~2年', '2~5年', '5~10年', '10~15年', '15年以上']

publish_bins = pd.DataFrame({'bins_publish': label_publish})
application_train['publish_bins'] = pd.cut(
    -application_train['DAYS_ID_PUBLISH'] / 365,
    bins=bins_publish,
    labels=label_publish,
    right=False)

# 统计每个箱子的违约率
publish_bins['default_rate'] = publish_bins['bins_publish'].map(
    application_train.groupby('publish_bins')['TARGET'].mean().map(lambda x: f"{x:.2%}")
)
publish_bins['population'] = publish_bins['bins_publish'].map(
    application_train.groupby('publish_bins').size()
)
publish_bins['pop_default'] = publish_bins['bins_publish'].map(
    application_train.groupby('publish_bins')['TARGET'].sum()
)

# 	bins_publish	default_rate	population	pop_default
# 0	0~2年	10.17%	30735	3125
# 1	2~5年	9.69%	51036	4946
# 2	5~10年	8.46%	92710	7843
# 3	10~15年	6.72%	130151	8751
# 4	15年以上	5.56%	2879	160

# 各组样本量充足，可以看出越临近申请日更改id文件的申请人违约率越高

del bins_publish, label_publish, publish_bins
del discrete_values_columns, discrete_values_frequency

# 筛选 float 类型的字段，即连续型数值变量
continuous_values_columns = application_train.select_dtypes(include='float').columns

