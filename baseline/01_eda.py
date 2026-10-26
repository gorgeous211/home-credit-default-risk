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

'''SK_ID_CURR 转换为 str
   CNT_FAM_MEMBERS 转换为 int
   OWN_CAR_AGE 转换为 int
   其余变量不需要转换'''

application_train['SK_ID_CURR'] = application_train['SK_ID_CURR'].astype(str)
application_train['CNT_FAM_MEMBERS'] = application_train['CNT_FAM_MEMBERS'].astype('Int64')
application_train['OWN_CAR_AGE'] = application_train['OWN_CAR_AGE'].astype('Int64')


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

" -> 相同缺失率的字段缺失模式都相同"

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

''' -> 缺失率较高的字段大部分是与房屋相关
    -> 说明一部分申请人缺少相关房屋信息，而不是数据随机缺失 '''

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
" 坏样本率都相同"

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

# 	missing_cols	missing_rate
# 7	EXT_SOURCE_2	0.21%
# 60	DAYS_LAST_PHONE_CHANGE	0.00%
# 1	AMT_GOODS_PRICE	0.09%
# 5	CNT_FAM_MEMBERS	0.00%
# 0	AMT_ANNUITY	0.00%

''' 这些变量的缺失率非常小，并且缺失组与正常组的违约率无显著差异，后续用中位数填充
    其余差异显著，考虑在特征工程阶段添加 _is_missing 列，相同缺失模式的只添加一列 '''

missing_impact_summary['is_significant'] = missing_impact_summary['p_value'] < 0.05

# 除此以外，其余变量两组差值均非常显著，说明缺失对于违约是一个强信号
# 后续特征工程添加显著列 _miss 特征

# 对于房屋相关字段，缺失模式相同，考虑后续特征工程只添加一个_miss变量，避免信息重复

# 输出缺失值分析的结果
missing_impact_summary.to_csv(DATA_PATH / 'interim' / 'missing_impact_summary.csv')
del missing_impact_summary

# -- 异常值分析 --
# 针对数值变量
def get_outlier(s):
    Q_1 = s.quantile(0.25)
    Q_3 = s.quantile(0.75)
    IQR = Q_3 - Q_1

    outlier = s[(s < Q_1 - 1.5 * IQR)
                | (s > Q_3 + 1.5 * IQR)]
    return outlier

nm_df = application_train.select_dtypes(exclude='object')
outliers_dict = {col: get_outlier(nm_df[col])
                 for col in nm_df.columns}

del nm_df

# DAYS_EMPLOYED 存在异常值365243

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

# 新建处理汇总表
discrete_values_summary = pd.DataFrame({'columns': discrete_values_columns,
                                        'process': 'keep'})

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

# FLAG_MOBIL 删除，几乎0方差
discrete_values_summary.loc[
    discrete_values_summary['columns'] == 'FLAG_MOBIL',
    'process'] = 'delete'

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

# 小于 0.01%：DOCUMENT_2、4、10、12，考虑删除
# 0.01%–0.1%：DOCUMENT_7、17、19、20、21，考虑删除
# 0.1%–1%：DOCUMENT_9、11、13、14、15、18，结合违约率进一步判断
# 1%–10%：DOCUMENT_5、6、8、16，保留
# 大于 10%：DOCUMENT_3，保留

discrete_values_summary.loc[
    discrete_values_summary['columns'].isin([
        f"FLAG_DOCUMENT_{i}"
        for i in [2, 4, 10, 12, 7, 17, 19, 20, 21]
    ]), 'process'] = 'delete'

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

# 一般处理函数
def bins_stats(bins: list, labels: list, column: str, df: pd.DataFrame) -> pd.DataFrame():
    if column.startswith('DAYS_'):
        data = -application_train[column] / 365
    else:
        data = application_train[column]

    application_train[f"{column}_bins"] = pd.cut(
        x=data,
        bins=bins,
        labels=labels,
        right=False
    )
    
    if application_train[f"{column}_bins"].isna().any():
        application_train[f"{column}_bins"] = \
            application_train[f"{column}_bins"].cat.add_categories(['特殊编码组']).fillna('特殊编码组')
    
    # 统计不同箱子违约率
    df['default_rate'] = df['bins'].map(
        application_train.groupby(f"{column}_bins")['TARGET'].mean().map(lambda x: f"{x:.2%}")
    )
    # 统计不同箱子样本量
    df['population'] = df['bins'].map(
        application_train.groupby(f"{column}_bins").size()
    )
    # 统计不同箱子的违约样本量
    df['pop_default'] = df['bins'].map(
        application_train.groupby(f"{column}_bins")['TARGET'].sum()
    )
    
    return df

# DAYS_BIRTH 取值范围为20~70，按业务含义分箱，20~29，30~39一个箱子
bins_birth = [20, 30, 40, 50, 60, np.inf]
labels_birth = ['20~29岁', '30~39岁', '40~49岁', '50岁~59岁', '60岁以上']

DAYS_BIRTH_summary = pd.DataFrame({'bins': labels_birth})

DAYS_BIRTH_summary = bins_stats(bins=bins_birth,
                                labels=labels_birth,
                                column='DAYS_BIRTH',
                                df=DAYS_BIRTH_summary)

# 	bins	default_rate	population	pop_default
# 0	20~29岁	11.46%	45000	5155
# 1	30~39岁	9.58%	82299	7885
# 2	40~49岁	7.66%	76543	5860
# 3	50岁~59岁	6.13%	68074	4173
# 4	60岁以上	4.92%	35595	1752

# 各组的样本量充足，违约率和年龄是单调递减的关系

del bins_birth, labels_birth

# DAYS_EMPLOYED 将特殊编码365243单独一个箱子，采用业务含义分箱，
bins_employed = [0, 1, 3, 5, 10, np.inf]
labels_employed = ['少于1年', '1~3年', '3~5年', '5~10年', '10年以上']

DAYS_EMPLOYED_summary = pd.DataFrame({'bins': labels_employed + ['特殊编码组']})

DAYS_EMPLOYED_summary = bins_stats(bins_employed,
                                   labels_employed,
                                   'DAYS_EMPLOYED',
                                   DAYS_EMPLOYED_summary)

# 	employed_bins	default_rate	population	pop_default
# 0	少于1年	10.98%	27904	3065
# 1	1~3年	11.07%	61489	6805
# 2	3~5年	9.67%	46874	4535
# 3	5~10年	7.37%	64890	4785
# 4	10年以上	5.19%	50980	2645
# 5	特殊编码组	5.40%	55374	2990

# 各组的样本量充足，可以看出随着工作年限增长，违约率是下降的，但是不是一个严格的线性关系

del bins_employed, labels_employed

# 最后分析 DAYS_ID_PUBLISH
# 采用业务含义分箱
bins_publish = [0, 2, 5, 10, 15, np.inf]
label_publish = ['0~2年', '2~5年', '5~10年', '10~15年', '15年以上']

DAYS_ID_PUBLISH_summary = pd.DataFrame({'bins': label_publish})

DAYS_ID_PUBLISH_summary = bins_stats(bins=bins_publish,
                                     labels=label_publish,
                                     column='DAYS_ID_PUBLISH',
                                     df=DAYS_ID_PUBLISH_summary)

del bins_publish, label_publish

# 	bins_publish	default_rate	population	pop_default
# 0	0~2年	10.17%	30735	3125
# 1	2~5年	9.69%	51036	4946
# 2	5~10年	8.46%	92710	7843
# 3	10~15年	6.72%	130151	8751
# 4	15年以上	5.56%	2879	160

# 各组样本量充足，可以看出越临近申请日更改id文件的申请人违约率越高

del DAYS_BIRTH_summary, DAYS_EMPLOYED_summary, DAYS_ID_PUBLISH_summary

# 分析 CNT_CHILDREN 这一字段
# CNT_CHILDREN	count
# 0	215371
# 1	61119
# 2	26749
# 3	3717
# 4	429
# 5	84
# 6	21
# 7	7
# 8	2
# 9	2
# 10	2
# 11	1
# 12	2
# 14	3
# 19	2

application_train['CNT_CHILDREN'].describe()
# count    307511.000000
# mean          0.417052
# std           0.722121
# min           0.000000
# 25%           0.000000
# 50%           0.000000
# 75%           1.000000
# max          19.000000
# Name: CNT_CHILDREN, dtype: float64

# 有少数的极端值，按业务含义分箱处理
bins_children = [0, 1, 2, 3, np.inf]
labels_children = ['0个孩子', '1个孩子', '2个孩子', '3个孩子及以上']

CNT_CHILDREN_summary = pd.DataFrame({'bins': labels_children})
CNT_CHILDREN_summary = bins_stats(bins_children,
                                  labels_children,
                                  'CNT_CHILDREN',
                                  CNT_CHILDREN_summary)

# 	bins	default_rate	population	pop_default
# 0	0个孩子	7.71%	215371	16609
# 1	1个孩子	8.92%	61119	5454
# 2	2个孩子	8.72%	26749	2333
# 3	3个孩子及以上	10.04%	4272	429

# 各组样本量充足，可以看出，违约率随孩子数量增长而增长

# 分析 CNT_FAM_MEMBERS 字段
# CNT_FAM_MEMBERS	count
# 1	67847
# 2	158357
# 3	52601
# 4	24697
# 5	3478
# 6	408
# 7	81
# 8	20
# 9	6
# 10	3
# 11	1
# 12	2
# 13	1
# 14	2
# 15	1
# 16	2
# 20	2

application_train['CNT_FAM_MEMBERS'].describe()
# count    307509.0
# mean     2.152665
# std      0.910682
# min           1.0
# 25%           2.0
# 50%           2.0
# 75%           3.0
# max          20.0
# Name: CNT_FAM_MEMBERS, dtype: Float64
bins_fam = [1, 2, 3, 4, 5, np.inf]
labels_fam = ['1人', '2人', '3人', '4人', '5人及以上']

CNT_FAM_MEMBERS_summary = pd.DataFrame({'bins': labels_fam})
CNT_FAM_MEMBERS_summary = bins_stats(bins_fam,
                                     labels_fam,
                                     'CNT_FAM_MEMBERS',
                                     CNT_FAM_MEMBERS_summary)

# 	bins	default_rate	population	pop_default
# 0	1人	8.36%	67847	5675
# 1	2人	7.58%	158357	12009
# 2	3人	8.76%	52601	4608
# 3	4人	8.65%	24697	2136
# 4	5人及以上	9.91%	4007	397

del discrete_values_columns, discrete_values_frequency


# 筛选 float 类型的字段，即连续型数值变量
continuous_values_columns = application_train.select_dtypes(include='float').columns

# 先分析第一类变量：收入、贷款金额及个人基本信息
financial_basic_features_columns = continuous_values_columns[: 6]
financial_basic_features = application_train[financial_basic_features_columns]

# 描述性统计
financial_basic_features_stats = financial_basic_features.describe()
# 	AMT_INCOME_TOTAL	AMT_CREDIT	AMT_ANNUITY	AMT_GOODS_PRICE	REGION_POPULATION_RELATIVE	DAYS_REGISTRATION
# count	307511.0	307511.0	307499.0	307233.0	307511.0	307511.0
# mean	168797.9192969845	599025.9997057016	27108.573909183444	538396.2074288895	0.020868112057780035	-4986.120327538419
# std	237123.14627885626	402490.77699585486	14493.737315118333	369446.46054005757	0.013831280122704704	3522.8863209630713
# min	25650.0	45000.0	1615.5	40500.0	0.00029	-24672.0
# 25%	112500.0	270000.0	16524.0	238500.0	0.010006	-7479.5
# 50%	147150.0	513531.0	24903.0	450000.0	0.01885	-4504.0
# 75%	202500.0	808650.0	34596.0	679500.0	0.028663	-2010.0
# max	117000000.0	4050000.0	258025.5	4050000.0	0.072508	0.0

# 绘制直方图
sns.histplot(
        data=financial_basic_features,
        x='AMT_INCOME_TOTAL',
        kde=True,
        log_scale=True)

sns.histplot(
        data=financial_basic_features,
        x='AMT_CREDIT',
        kde=True)

sns.histplot(
        data=financial_basic_features,
        x='AMT_ANNUITY',
        kde=True)

# 均为右偏的长尾分布，其中AMT_INCOME_TOTAL 右偏非常强烈

# 绘制箱线图
sns.boxplot(data=financial_basic_features, y="AMT_INCOME_TOTAL")

sns.boxplot(data=financial_basic_features, y="AMT_CREDIT")

sns.boxplot(data=financial_basic_features, y="AMT_ANNUITY")

# 分箱进一步分析与违约率的关系
def get_5_qcut(column):
    application_train[f"{column}_bins"] = pd.qcut(
        application_train[column],
        q=5,
        labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    )
    
    default_rate = application_train.groupby([f"{column}_bins"])['TARGET'].mean()
    
    return default_rate.map(lambda x: f"{x:.2%}")

get_5_qcut('AMT_INCOME_TOTAL')
# AMT_INCOME_TOTAL_bins
# Q1    8.21%
# Q2    8.59%
# Q3    8.68%
# Q4    8.06%
# Q5    6.52%
# Name: TARGET, dtype: object
get_5_qcut('AMT_CREDIT')
# AMT_CREDIT_bins
# Q1     7.24%
# Q2     9.17%
# Q3    10.05%
# Q4     7.85%
# Q5     6.08%
# Name: TARGET, dtype: object
get_5_qcut('AMT_ANNUITY')
# AMT_ANNUITY_bins
# Q1    6.95%
# Q2    8.58%
# Q3    8.79%
# Q4    9.34%
# Q5    6.70%
# Name: TARGET, dtype: object