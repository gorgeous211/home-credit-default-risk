# %%
# 风控建模中最重要的8类变量

# 1. 还款能力变量（核心）
# 反映客户有没有钱还款，是信用风险最基础的判断维度。
# - 收入：月收入、工资流水、收入稳定性
# - 负债：当前负债、月还款额、负债收入比（DTI）
# - 现金流：月均流入、月均流出、账户余额、现金流波动
# - 资产：存款、房产、可变现资产
# 业务逻辑： 收入不稳定、负债过高、现金流持续恶化，通常意味着违约风险上升。

# 2. 历史信用表现变量（核心）
# 反映客户过去是否有履约意愿和履约能力。
# - 历史逾期次数、最大逾期天数
# - 历史 M1、M2、M3 逾期记录
# - 近 3/6/12 个月逾期次数
# - 历史结清情况、核销情况
# - 信用账户数量、信贷历史长度
# 业务逻辑： 近期发生的严重逾期，通常比很久以前的一次轻微逾期更值得关注。逾期的严重程度、频率和距今时间都很重要。

# 3. 多头借贷与负债压力变量
# 反映客户是否在多个机构借款，以及是否存在过度借贷。
# - 当前在贷机构数
# - 近 7/30/90 天贷款申请次数
# - 近 30 天新增借款次数
# - 总授信额度、已用额度、额度使用率
# - 多平台借款频率、短期借款增长率
# 业务逻辑： 短期内频繁申请贷款、多个平台同时新增借款，可能意味着资金压力增加。
# 注意：多头不一定等于高风险，要结合客户收入、存量负债和借款用途判断。

# 4. 申请行为与借款行为变量
# 反映客户在什么情况下申请贷款，以及借款行为是否异常。
# - 申请时间、申请频率
# - 申请金额、期限、选择的产品
# - 重复申请、短时间多次修改申请信息
# - 额度使用和支用行为
# - 借款后是否迅速再次借款
# 业务逻辑： 单次申请金额突然升高、短期反复申请，可能反映资金需求发生变化，但也可能是正常的消费或经营需求。

# 5. 身份真实性与反欺诈变量
# 反映客户是否真实、申请行为是否存在欺诈风险。
# - 设备关联账号数
# - IP、设备、手机号、身份证的关联关系
# - 注册到申请的时间间隔
# - 设备环境异常、模拟器或自动化行为
# - 身份信息与外部数据的一致性
# 业务逻辑： 同一设备关联大量无明显合理关系的申请人，可能存在团伙欺诈或批量注册风险。
# 这一类变量对反欺诈模型尤其重要，对纯粹的违约概率模型则要看数据和业务场景。

# 6. 账户与交易行为变量
# 反映客户近期财务状况、消费模式和行为变化。
# - 日均余额、余额波动率
# - 入账金额、出账金额、净现金流
# - 交易频次、交易金额分布
# - 工资入账规律、异常大额交易
# - 账户活跃度、交易对手集中度
# 业务逻辑： 比起单一时点的余额，持续多月的现金流趋势可能更有解释力。收入突然下降、余额持续减少等变化值得关注。

# 7. 产品与借款条件变量
# 反映客户承担的还款压力，以及不同产品本身的风险差异。
# - 借款金额、期限、利率
# - 月供金额、还款方式
# - 额度、首借或复借
# - 抵押物价值、贷款价值比（LTV）
# - 产品类型、渠道、资金方
# 业务逻辑： 相同的客户在不同贷款金额、期限和月供下，可能面临不同的违约风险。
# 特别注意：产品、渠道等变量可能捕捉到的是业务准入规则或历史客群差异，不一定是客户自身的风险特征。

# 8. 时间变化与趋势变量
# 这是很多建模人员容易低估的一类变量。
# - 近 7/30/90 天申请次数
# - 近 3 个月负债增长率
# - 最近一次逾期距今天数
# - 收入环比变化
# - 近期与历史平均行为的偏离程度
# 业务逻辑： 客户当前的风险状态可能比几年前的静态画像更重要。
# 例如，同样有 5 笔贷款，一个客户负债稳定且按时还款，另一个客户近一个月突然新增 4 笔贷款，两者的风险可能完全不同。

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
# cols_meaning = pd.read_csv(DATA_PATH / 'HomeCredit_columns_description.csv',
#                            usecols=['Table', 'Row', 'Description'],
#                            encoding="latin1")
# cols_meaning = cols_meaning[cols_meaning['Table'] == 'application_{train|test}.csv']

# 缺失率报告
cols_missinig_report = pd.read_csv(DATA_PATH / 'interim' / 'missing_impact_summary.csv')

# 随着eda阶段逐步更新
cols_eda = pd.read_csv(DATA_PATH / 'interim' / 'cols_eda.csv')

# %%
# 两样本比例之差检验函数
def run_ztest_row(row):
    col_default_n_missing = 'default_numbers_when_missing'
    col_n_missing = 'numbers_when_missing'
    col_default_n_present = 'default_numbers_when_present'
    col_n_present = 'numbers_when_present'
    
    n_default_miss = row[col_default_n_missing]
    n_default_pres = row[col_default_n_present]
    
    n_miss = row[col_n_missing]
    n_pres = row[col_n_present]
    
    stat, pval = proportions_ztest(
        count=[n_default_miss, n_default_pres],
        nobs=[n_miss, n_pres],
        alternative='two-sided')
    
    return pd.Series({'z_stat': stat, 'p_value': pval})


# 异常值函数
def get_outlier(s):
    Q_1 = s.quantile(0.25)
    Q_3 = s.quantile(0.75)
    IQR = Q_3 - Q_1

    outlier = s[(s < Q_1 - 1.5 * IQR)
                | (s > Q_3 + 1.5 * IQR)]
    return outlier


# 离散变量分箱函数
def bins_stats(bins: list, labels: list, column: str, df: pd.DataFrame) -> pd.DataFrame:
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


# 连续变量5等分等频分箱函数
def get_5_qcut(column):
    application_train[f"{column}_bins"] = pd.qcut(
        application_train[column],
        q=5,
        labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
    )
    
    default_rate = application_train.groupby([f"{column}_bins"])['TARGET'].mean()
    
    return default_rate.map(lambda x: f"{x:.2%}")

# 字段不同类别的违约率
def diff_type_default_rate(column):
    col_default_rate = {}
    col_dtypes = application_train[column].value_counts().index
    for i in col_dtypes:
        default_rate = application_train.loc[
            application_train[column] == i, 'TARGET'].mean()
        col_default_rate[i] = f"{default_rate:.2%}"
    return col_default_rate.items()

# %%
"一、数据概况与数据质量"

"数据规模、目标变量、缺失值、重复值、异常值、训练集和测试集差异。"

# -- 数据规模 --
# print(application_train.shape)
# (307511, 122)
# application_train 样本量为307511，含有122个字段

# print(application_test.shape)
# (48744, 121)
# application_test 样本量为48744，含有121个字段

# 检查测试集的字段是否和训练集一致
# print(set(application_test.columns).issubset(set(application_train.columns)))
# True

# -- 数据类型 --
cols_train_types = application_train.dtypes.rename('dtypes')
cols_test_types = application_test.dtypes.rename('dtypes')

# 添加数据类型到 cols_eda
# cols_eda = cols_meaning.copy()
# cols_eda['dtypes'] = cols_eda['Row'].map(cols_train_types)

# 检查测试集字段类型是否和训练集一致
# print(cols_train_types.drop('TARGET').equals(cols_test_types))
# True

'''SK_ID_CURR 转换为 str
   CNT_FAM_MEMBERS 转换为 int
   OWN_CAR_AGE 转换为 int
   其余变量不需要转换'''

# 结论更新到 cols_eda 中
# cols_eda['to_dtypes'] = np.nan
# cols_eda.loc[cols_eda['Row'] == 'SK_ID_CURR', 'to_dtypes'] = 'str'
# cols_eda.loc[cols_eda['Row'] == 'CNT_FAM_MEMBERS', 'to_dtypes'] = 'int'
# cols_eda.loc[cols_eda['Row'] == 'OWN_CAR_AGE', 'to_dtypes'] = 'int'

application_train['SK_ID_CURR'] = application_train['SK_ID_CURR'].astype(str)
application_train['CNT_FAM_MEMBERS'] = application_train['CNT_FAM_MEMBERS'].astype('Int64')
application_train['OWN_CAR_AGE'] = application_train['OWN_CAR_AGE'].astype('Int64')


# -- 重复与唯一性 --
# print(application_train.duplicated().any())
# False
# print(application_train.duplicated(subset='SK_ID_CURR').any())
# False
# print(application_test.duplicated().any())
# False
" -> 无重复"

del cols_train_types, cols_test_types

# -- 缺失率情况 --
missing_summary = application_train.isna().mean().reset_index()
missing_summary.columns = ['Row', 'missing_rate']

# 没有缺失的列
present_columns = missing_summary.loc[missing_summary['missing_rate'] == 0, 'Row'].tolist()

# 缺失的列
missing_summary = missing_summary[missing_summary['missing_rate'] > 0].reset_index(drop=True)
missing_columns = missing_summary['Row'].tolist()

# missing_summary.sort_values(by='missing_rate', ascending=False).head(20)
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

'''注意到 COMMONAREA_ NONLIVINGAPARTMENTS_ LIVINGAPARTMENTS_ FLOORSMIN_ 等缺失率相同
    可能是相同的缺失模式'''

# 进一步探究缺失模式
# same_missing_pattern_groups = []

# for rate, group in missing_summary.groupby('missing_rate'):
    
#     if len(group) > 1:
#         temp_list = group['index'].tolist()
#         temp_df = application_train[temp_list]
        
#         if len(temp_df.isna().T.drop_duplicates()) == 1:
#             same_missing_pattern_groups.append(temp_list)
#         else:
#             print(f"{temp_list}的缺失模式不同")

# del rate, group, temp_df, temp_list, same_missing_pattern_groups

" -> 相同缺失率的字段缺失模式都相同"

# missing_summary.groupby('missing_rate').head(1).sort_values(by='missing_rate', ascending=False)
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
    cols_eda[['Row', 'Description']],
    on='Row',
    how='left')

''' -> 缺失率较高的字段大部分是与房屋相关
    -> 说明一部分申请人缺少相关房屋信息，而不是数据随机缺失 '''

# 去除房屋相关字段后，其余高缺失率变量
# missing_summary.loc[
#     ~missing_summary['Description'].str.startswith('Normalized information'),
#     ['Row', 'missing_rate']
# ].sort_values(by='missing_rate', ascending=False)

#                            Row  missing_rate
# 3                  OWN_CAR_AGE      0.659908
# 6                 EXT_SOURCE_1      0.563811
# 4              OCCUPATION_TYPE      0.313455
# 8                 EXT_SOURCE_3      0.198253
# 66  AMT_REQ_CREDIT_BUREAU_YEAR      0.135016
# 65   AMT_REQ_CREDIT_BUREAU_QRT      0.135016
# 64   AMT_REQ_CREDIT_BUREAU_MON      0.135016
# 63  AMT_REQ_CREDIT_BUREAU_WEEK      0.135016
# 62   AMT_REQ_CREDIT_BUREAU_DAY      0.135016
# 61  AMT_REQ_CREDIT_BUREAU_HOUR      0.135016
# 2              NAME_TYPE_SUITE      0.004201
# 59    DEF_60_CNT_SOCIAL_CIRCLE      0.003320
# 57    DEF_30_CNT_SOCIAL_CIRCLE      0.003320
# 58    OBS_60_CNT_SOCIAL_CIRCLE      0.003320
# 56    OBS_30_CNT_SOCIAL_CIRCLE      0.003320
# 7                 EXT_SOURCE_2      0.002146
# 1              AMT_GOODS_PRICE      0.000904
# 0                  AMT_ANNUITY      0.000039
# 5              CNT_FAM_MEMBERS      0.000007
# 60      DAYS_LAST_PHONE_CHANGE      0.000003

# 输出缺失率排名图
sns.barplot(
    data=missing_summary.sort_values(by='missing_rate', ascending=False).head(30),
    x='missing_rate',
    y='Row')

sns.barplot(
    data=missing_summary.loc[
        ~missing_summary['Description'].str.startswith('Normalized'),
        ['Row', 'missing_rate']
        ].sort_values(by='missing_rate', ascending=False),
    x='missing_rate',
    y='Row')

# 缺失率分析
default_rate_when_missing = []     # 缺失组违约率
numbers_when_missing = []  # 缺失组样本量
default_numbers_when_missing = []  # 缺失组违约样本量

for col in missing_columns:
    temp_df = application_train[[col, 'TARGET']]
    
    # 计算违约率
    default_rate = temp_df.loc[temp_df[col].isna(), 'TARGET'].mean()
    # 计算样本量
    number = len(temp_df[temp_df[col].isna()])
    # 计算违约样本量
    default_number = temp_df.loc[temp_df[col].isna(), 'TARGET'].sum()
    
    default_rate_when_missing.append(default_rate)
    numbers_when_missing.append(number)
    default_numbers_when_missing.append(default_number)

default_rate_when_present = []     # 正常组违约率
numbers_when_present = []  # 正常组样本量
default_numbers_when_present = []  # 正常组违约样本量

for col in missing_columns:
    temp_df = application_train[[col, 'TARGET']]
    
    default_rate = temp_df.loc[temp_df[col].notna(), 'TARGET'].mean()
    number = len(temp_df[temp_df[col].notna()])
    default_number = temp_df.loc[temp_df[col].notna(), 'TARGET'].sum()
    
    default_rate_when_present.append(default_rate)
    numbers_when_present.append(number)
    default_numbers_when_present.append(default_number)

del col, temp_df, default_rate, number, default_number

# 汇总各项指标，输出汇总表
missing_impact_summary = pd.DataFrame(
    {'missing_cols': missing_columns,
     'missing_rate': missing_summary['missing_rate'],
     
     'default_rate_when_missing': default_rate_when_missing,
     'numbers_when_missing': numbers_when_missing,
     'default_numbers_when_missing': default_numbers_when_missing,
     
     'default_rate_when_present': default_rate_when_present,
     'numbers_when_present': numbers_when_present,
     'default_numbers_when_present': default_numbers_when_present,
     
     'default_rate_diff': [default_rate_when_missing[i] -
                           default_rate_when_present[i]
                           for i in range(len(default_rate_when_missing))]}
)

del default_rate_when_missing, numbers_when_missing, default_numbers_when_missing
del default_rate_when_present, numbers_when_present, default_numbers_when_present

temp_list = ['missing_rate',
             'default_rate_when_missing',
             'default_rate_when_present',
             'default_rate_diff']
missing_impact_summary[temp_list] = missing_impact_summary[temp_list].map(lambda x: f"{x:.5%}")
# map对元素逐项变化，apply对series变化

for rate, group in missing_impact_summary.groupby('missing_rate'):
    if len(group) > 1:
        temp_list_ = group['missing_cols']
        temp_s = group['default_rate_when_missing']
        if temp_s.max() > temp_s.min():
            print(f"{temp_list}的坏样本率不同")

del temp_list_, temp_s, group, rate

"相同缺失率的字段，坏样本率都相同"

# 更新到 cols_eda 中
# cols_eda = pd.merge(
#     cols_eda,
#     missing_impact_summary[temp_list + ['missing_cols']],
#     left_on='Row',
#     right_on='missing_cols',
#     how='left'
# )
    
# del cols_eda['missing_cols']
del temp_list

# 初步输出 cols_eda
# cols_eda.to_csv(DATA_PATH / 'interim' / 'cols_eda.csv')

# 进一步分析缺失状态与 TARGET 的关系
# 检验缺失组的违约率与非缺失组的违约率差异是否显著，设显著性水平为0.05
# 两样本比例之差的假设检验

missing_impact_summary[['z_stat', 'p_value']] = missing_impact_summary.apply(run_ztest_row, axis=1)

#     missing_cols            missing_rate
# 7   EXT_SOURCE_2            0.21%
# 60  DAYS_LAST_PHONE_CHANGE  0.00%
# 1   AMT_GOODS_PRICE         0.09%
# 5   CNT_FAM_MEMBERS         0.00%
# 0   AMT_ANNUITY             0.00%

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
"三、单变量分析"

"年龄、收入、贷款金额、工作年限、家庭情况等变量。"

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

cols_eda['is_delete'] = np.nan

"筛选 int 类型的字段：包含0/1分类变量，离散数值变量"
discrete_values_columns = application_train.select_dtypes(include='int').columns

discrete_values_frequency = {}
for col in discrete_values_columns:
    discrete_values_frequency[col] = application_train[col].value_counts()

del col

# DAYS_EMPLOYED 存在异常编码 55374

# 存在取值极度稀有情况
# FLAG_MOBIL       count
# 1                307510
# 0                1
# FLAG_DOCUMENT_2  count
# 0                307498
# 1                13
# FLAG_DOCUMENT_4  count
# 0                307486
# 1                25

# FLAG_MOBIL 删除，几乎0方差
# cols_eda.loc[cols_eda['Row'] == 'FLAG_MOBIL', 'is_delete'] = 'delete'

# 重点检查 FLAG_DOCUMENT_ 的取值情况（0/1）
FLAG_frequency = pd.DataFrame({
    'columns': [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)],
    'frequency': [f"{application_train[f'FLAG_DOCUMENT_{i}'].mean():.4%}"
                  for i in range(2, 22)]
})

# 统计每一列取1的频率
#     columns           frequency
# 0   FLAG_DOCUMENT_2   0.0042%
# 1   FLAG_DOCUMENT_3   71.0023%
# 2   FLAG_DOCUMENT_4   0.0081%
# 3   FLAG_DOCUMENT_5   1.5115%
# 4   FLAG_DOCUMENT_6   8.8055%
# 5   FLAG_DOCUMENT_7   0.0192%
# 6   FLAG_DOCUMENT_8   8.1376%
# 7   FLAG_DOCUMENT_9   0.3896%
# 8   FLAG_DOCUMENT_10  0.0023%
# 9   FLAG_DOCUMENT_11  0.3912%
# 10  FLAG_DOCUMENT_12  0.0007%
# 11  FLAG_DOCUMENT_13  0.3525%
# 12  FLAG_DOCUMENT_14  0.2936%
# 13  FLAG_DOCUMENT_15  0.1210%
# 14  FLAG_DOCUMENT_16  0.9928%
# 15  FLAG_DOCUMENT_17  0.0267%
# 16  FLAG_DOCUMENT_18  0.8130%
# 17  FLAG_DOCUMENT_19  0.0595%
# 18  FLAG_DOCUMENT_20  0.0507%
# 19  FLAG_DOCUMENT_21  0.0335%

# 小于 0.01%：DOCUMENT_2、4、10、12，考虑删除
# 0.01%–0.1%：DOCUMENT_7、17、19、20、21，考虑删除
# 0.1%–1%：DOCUMENT_9、11、13、14、15、18，结合违约率进一步判断
# 1%–10%：DOCUMENT_5、6、8、16，保留
# 大于 10%：DOCUMENT_3，保留

# cols_eda.loc[cols_eda['Row'].isin(
#     [f"FLAG_DOCUMENT_{i}"
#     for i in [2, 4, 10, 12, 7, 17, 19, 20, 21]]),
#     'is_delete'] = 'delete'

# cols_eda.loc[cols_eda['Row'].isin(
#     [f"FLAG_DOCUMENT_{i}"
#     for i in [9, 11, 13, 14, 15, 18]]),
#     'is_delete'] = 'consider'

# cols_eda.loc[cols_eda['Row'].isin(
#     [f"FLAG_DOCUMENT_{i}"
#     for i in [5, 6, 8, 16, 3]]),
#     'is_delete'] = 'keep'

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

# 结果保存到 cols_eda
# cols_eda.loc[cols_eda['Row'].isin(
#     ['FLAG_EMP_PHONE',
#      'FLAG_WORK_PHONE',
#      'FLAG_CONT_MOBILE',
#      'FLAG_PHONE',
#      'FLAG_EMAIL']), 'is_delete'] = 'keep'

# 输出更新后的cols_eda
cols_eda.to_csv(DATA_PATH / 'interim' / 'cols_eda.csv', index=False)


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

DAYS_BIRTH_summary = pd.DataFrame({'bins': labels_birth})

DAYS_BIRTH_summary = bins_stats(bins=bins_birth,
                                labels=labels_birth,
                                column='DAYS_BIRTH',
                                df=DAYS_BIRTH_summary)

#    bins       default_rate  population  pop_default
# 0  20~29岁    11.46%        45000       5155
# 1  30~39岁    9.58%         82299       7885
# 2  40~49岁    7.66%         76543       5860
# 3  50岁~59岁  6.13%         68074       4173
# 4  60岁以上   4.92%         35595       1752

# 各组的样本量充足，违约率和年龄是单调递减的关系

cols_eda.loc[cols_eda['Row'] == 'DAYS_BIRTH', 'is_delete'] = 'keep'

del bins_birth, labels_birth

# DAYS_EMPLOYED 将特殊编码365243单独一个箱子，采用业务含义分箱，
bins_employed = [0, 1, 3, 5, 10, np.inf]
labels_employed = ['少于1年', '1~3年', '3~5年', '5~10年', '10年以上']

DAYS_EMPLOYED_summary = pd.DataFrame({'bins': labels_employed + ['特殊编码组']})

DAYS_EMPLOYED_summary = bins_stats(bins_employed,
                                   labels_employed,
                                   'DAYS_EMPLOYED',
                                   DAYS_EMPLOYED_summary)

#    employed_bins  default_rate  population  pop_default
# 0  少于1年        10.98%        27904       3065
# 1  1~3年          11.07%        61489       6805
# 2  3~5年          9.67%         46874       4535
# 3  5~10年         7.37%         64890       4785
# 4  10年以上       5.19%         50980       2645
# 5  特殊编码组     5.40%         55374       2990

# 各组的样本量充足，可以看出随着工作年限增长，违约率是下降的，但是不是一个严格的线性关系

cols_eda.loc[cols_eda['Row'] == 'DAYS_EMPLOYED', 'is_delete'] = 'keep'

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

cols_eda.loc[cols_eda['Row'] == 'DAYS_ID_PUBLISH', 'is_delete'] = 'keep'

del bins_publish, label_publish

#    bins_publish  default_rate  population  pop_default
# 0  0~2年         10.17%        30735       3125
# 1  2~5年         9.69%         51036       4946
# 2  5~10年        8.46%         92710       7843
# 3  10~15年       6.72%         130151      8751
# 4  15年以上      5.56%         2879        160

# 各组样本量充足，可以看出越临近申请日更改id文件的申请人违约率越高

del DAYS_BIRTH_summary, DAYS_EMPLOYED_summary, DAYS_ID_PUBLISH_summary

# 分析 CNT_CHILDREN 这一字段
# CNT_CHILDREN  count
# 0             215371
# 1             61119
# 2             26749
# 3             3717
# 4             429
# 5             84
# 6             21
# 7             7
# 8             2
# 9             2
# 10            2
# 11            1
# 12            2
# 14            3
# 19            2

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

#    bins           default_rate  population  pop_default
# 0  0个孩子        7.71%         215371      16609
# 1  1个孩子        8.92%         61119       5454
# 2  2个孩子        8.72%         26749       2333
# 3  3个孩子及以上  10.04%        4272        429

# 各组样本量充足，可以看出，违约率随孩子数量增长而增长

cols_eda.loc[cols_eda['Row'] == 'CNT_CHILDREN', 'is_delete'] = 'keep'

# 分析 CNT_FAM_MEMBERS 字段
# CNT_FAM_MEMBERS  count
# 1                67847
# 2                158357
# 3                52601
# 4                24697
# 5                3478
# 6                408
# 7                81
# 8                20
# 9                6
# 10               3
# 11               1
# 12               2
# 13               1
# 14               2
# 15               1
# 16               2
# 20               2

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

#    bins       default_rate  population  pop_default
# 0  1人        8.36%         67847       5675
# 1  2人        7.58%         158357      12009
# 2  3人        8.76%         52601       4608
# 3  4人        8.65%         24697       2136
# 4  5人及以上  9.91%         4007        397

cols_eda.loc[cols_eda['Row'] == 'CNT_FAM_MEMBERS', 'is_delete'] = 'keep'

del discrete_values_columns, discrete_values_frequency

cols_eda.to_csv(DATA_PATH / 'interim' / 'cols_eda.csv', index=False)

"筛选 float 类型的字段，即连续型数值变量"
continuous_values_columns = application_train.select_dtypes(include='float').columns

# 先分析第一类变量
# 1.收入、贷款金额及个人基本信息
financial_basic_features_columns = continuous_values_columns[: 6]

# 描述性统计
financial_basic_features_stats = application_train[financial_basic_features_columns].describe()
#        AMT_INCOME_TOTAL    AMT_CREDIT          AMT_ANNUITY         AMT_GOODS_PRICE     REGION_POPULATION_RELATIVE  DAYS_REGISTRATION
# count  307511.0            307511.0            307499.0            307233.0            307511.0                    307511.0
# mean   168797.9192969845   599025.9997057016   27108.573909183444  538396.2074288895   0.020868112057780035        -4986.120327538419
# std    237123.14627885626  402490.77699585486  14493.737315118333  369446.46054005757  0.013831280122704704        3522.8863209630713
# min    25650.0             45000.0             1615.5              40500.0             0.00029                     -24672.0
# 25%    112500.0            270000.0            16524.0             238500.0            0.010006                    -7479.5
# 50%    147150.0            513531.0            24903.0             450000.0            0.01885                     -4504.0
# 75%    202500.0            808650.0            34596.0             679500.0            0.028663                    -2010.0
# max    117000000.0         4050000.0           258025.5            4050000.0           0.072508                    0.0

# 绘制直方图
sns.histplot(
    data=application_train,
    x='AMT_INCOME_TOTAL',
    kde=True,
    log_scale=True)

sns.histplot(
    data=application_train,
    x='AMT_CREDIT',
    kde=True)

sns.histplot(
    data=application_train,
    x='AMT_ANNUITY',
    kde=True)

sns.histplot(
    data=application_train,
    x='AMT_GOODS_PRICE',
    kde=True)

# 均为右偏的长尾分布，其中AMT_INCOME_TOTAL 右偏非常强烈

# 绘制箱线图
sns.boxplot(data=application_train, y="AMT_INCOME_TOTAL")
sns.boxplot(data=application_train, y="AMT_CREDIT")
sns.boxplot(data=application_train, y="AMT_ANNUITY")
sns.boxplot(data=application_train, y="AMT_GOODS_PRICE")

# 分箱进一步分析与违约率的关系
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
get_5_qcut('AMT_GOODS_PRICE')
# AMT_GOODS_PRICE_bins
# Q1     8.01%
# Q2     8.79%
# Q3    10.38%
# Q4     7.58%
# Q5     5.73%
# Name: TARGET, dtype: object

# 以上四个变量对应的分组违约率均呈现非单调变化，这些变量与违约风险之间可能存在非线性关联

# 进一步分析这四个变量间相关性，计算相关系数矩阵
AMT_corr = application_train[
    ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
    ].corr().round(2)

#                   AMT_INCOME_TOTAL  AMT_CREDIT   AMT_ANNUITY      AMT_GOODS_PRICE
# AMT_INCOME_TOTAL  1.0         0.16         0.19             0.16
# AMT_CREDIT        0.16        1.0          0.77             0.99
# AMT_ANNUITY       0.19        0.77         1.0              0.78
# AMT_GOODS_PRICE   0.16        0.99         0.78             1.0

# 可以发现 AMT_CREDIT 和 AMT_GOODS_PRICE 相关系数极高，存在重复信息
# AMT_GOODS_PRICE 和 AMT_ANNUITY，AMT_CREDIT 和 AMT_ANNUITY 也有较强的线性相关关系

# 分析 REGION_POPULATION_RELATIVE
get_5_qcut('REGION_POPULATION_RELATIVE')
# REGION_POPULATION_RELATIVE_bins
# Q1    8.94%
# Q2    7.92%
# Q3    9.55%
# Q4    7.73%
# Q5    6.00%
# Name: TARGET, dtype: object
# 可能与违约率存在非线性关系

# 分析 DAYS_REGISTRATION
(-application_train['DAYS_REGISTRATION'] / 365).describe()
# count    307511.000000
# mean         13.660604
# std           9.651743
# min          -0.000000
# 25%           5.506849
# 50%          12.339726
# 75%          20.491781
# max          67.594521
# Name: DAYS_REGISTRATION, dtype: float64

# 分箱进一步分析
DAYS_REGISTRATION_bins = [0, 2, 5, 10, 20, np.inf]
DAYS_REGISTRATION_labels = ['0-2年', '2-5年', '5-10年', '10-20年', '20年以上']
DAYS_REGISTRATION_summary = pd.DataFrame({'bins': DAYS_REGISTRATION_labels})

bins_stats(DAYS_REGISTRATION_bins,
           DAYS_REGISTRATION_labels,
           'DAYS_REGISTRATION',
           DAYS_REGISTRATION_summary)

#    bins      default_rate  population  pop_default
# 0  0-2年     9.64%         32432       3127
# 1  2-5年     8.80%         39133       3445
# 2  5-10年    8.72%         54618       4762
# 3  10-20年   8.31%         100662      8370
# 4  20年以上  6.35%         80666       5121

# DAYS_REGISTRATION 呈现出比较清晰的违约率随登记年限增加而总体下降的趋势，而且各组样本量都比较充足
del DAYS_REGISTRATION_bins, DAYS_REGISTRATION_labels, DAYS_REGISTRATION_summary
del financial_basic_features_columns


# 2.分析外部信用评分变量
ext_score_cols = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
#    missing_rate           Row
# 8  0.19825307062186392    EXT_SOURCE_3
# 7  0.0021462646864665006  EXT_SOURCE_2
# 6  0.5638107254699832     EXT_SOURCE_1

# 描述性统计
application_train['EXT_SOURCE_1'].describe()
# count    134133.000000
# mean          0.502130
# std           0.211062
# min           0.014568
# 25%           0.334007
# 50%           0.505998
# 75%           0.675053
# max           0.962693
# Name: EXT_SOURCE_1, dtype: float64

# 分布比较对称
sns.histplot(
    data=application_train,
    x='EXT_SOURCE_1',
    )

application_train['EXT_SOURCE_2'].describe()
# count    3.068510e+05
# mean     5.143927e-01
# std      1.910602e-01
# min      8.173617e-08
# 25%      3.924574e-01
# 50%      5.659614e-01
# 75%      6.636171e-01
# max      8.549997e-01
# Name: EXT_SOURCE_2, dtype: float64

sns.histplot(
    data=application_train,
    x='EXT_SOURCE_2',
    )

application_train['EXT_SOURCE_3'].describe()
# count    246546.000000
# mean          0.510853
# std           0.194844
# min           0.000527
# 25%           0.370650
# 50%           0.535276
# 75%           0.669057
# max           0.896010
# Name: EXT_SOURCE_3, dtype: float64

sns.histplot(
    data=application_train,
    x='EXT_SOURCE_3',
    )

# 分箱，观察与TARGET的关系
EXT_SOURCE_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
EXT_SOURCE_labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']

EXT_SOURCE_1_summary = pd.DataFrame({'bins': EXT_SOURCE_labels + ['特殊编码组']})
EXT_SOURCE_1_summary = bins_stats(bins=EXT_SOURCE_bins,
                                  labels=EXT_SOURCE_labels,
                                  column='EXT_SOURCE_1',
                                  df=EXT_SOURCE_1_summary)

#    bins        default_rate  population  pop_default
# 0  0-0.2       18.23%        11661       2126
# 1  0.2-0.4     10.31%        34542       3562
# 2  0.4-0.6     6.42%         39702       2550
# 3  0.6-0.8     4.16%         36986       1540
# 4  0.8-1.0     2.46%         11242       276
# 5  特殊编码组  8.52%         173378      14771

EXT_SOURCE_2_summary = pd.DataFrame({'bins': EXT_SOURCE_labels + ['特殊编码组']})
EXT_SOURCE_2_summary = bins_stats(bins=EXT_SOURCE_bins,
                                  labels=EXT_SOURCE_labels,
                                  column='EXT_SOURCE_2',
                                  df=EXT_SOURCE_2_summary)

#    bins        default_rate  population  pop_default
# 0  0-0.2       18.84%        27865       5251
# 1  0.2-0.4     11.62%        51078       5933
# 2  0.4-0.6     7.79%         98869       7706
# 3  0.6-0.8     4.57%         128470      5873
# 4  0.8-1.0     1.76%         569         10
# 5  特殊编码组  7.88%         660         52

EXT_SOURCE_3_summary = pd.DataFrame({'bins': EXT_SOURCE_labels + ['特殊编码组']})
EXT_SOURCE_3_summary = bins_stats(bins=EXT_SOURCE_bins,
                                  labels=EXT_SOURCE_labels,
                                  column='EXT_SOURCE_3',
                                  df=EXT_SOURCE_3_summary)

#    bins        default_rate  population  pop_default
# 0  0-0.2       20.94%        19514       4087
# 1  0.2-0.4     11.78%        52072       6133
# 2  0.4-0.6     6.52%         80777       5266
# 3  0.6-0.8     4.00%         84924       3396
# 4  0.8-1.0     2.87%         9259        266
# 5  特殊编码组  9.31%         60965       5677

# 评分越高，违约率越低，而且下降趋势非常明显

application_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].corr()

#               EXT_SOURCE_1  EXT_SOURCE_2  EXT_SOURCE_3
# EXT_SOURCE_1      1.000000      0.213982      0.186846
# EXT_SOURCE_2      0.213982      1.000000      0.109167
# EXT_SOURCE_3      0.186846      0.109167      1.000000

# 三者的相关性不强


# 3.房产及居住变量
# 由前面的分析可知，房产相关变量的缺失模式相同，进行相关性分析
house_cols = cols_eda.loc[
    cols_eda['Description'].str.startswith('Normalized information'),
    'Row'
].tolist()

house_corr = application_train[house_cols].select_dtypes(include=['number']).corr().round(2)
upper_mask = np.triu(np.ones(house_corr.shape), k=1).astype(bool)
house_corr = house_corr.where(upper_mask)

# 查看0.9以上的
temp_df = house_corr.stack().reset_index()
result = temp_df[temp_df[0].between(0.9, 1)]

#      level_0                      level_1                       0
# 9    APARTMENTS_AVG               LIVINGAPARTMENTS_AVG          0.94
# 10   APARTMENTS_AVG               LIVINGAREA_AVG                0.91
# 13   APARTMENTS_AVG               APARTMENTS_MODE               0.97
# 23   APARTMENTS_AVG               LIVINGAPARTMENTS_MODE         0.93
# 27   APARTMENTS_AVG               APARTMENTS_MEDI               1.0
# 37   APARTMENTS_AVG               LIVINGAPARTMENTS_MEDI         0.94
# 38   APARTMENTS_AVG               LIVINGAREA_MEDI               0.91
# 55   BASEMENTAREA_AVG             BASEMENTAREA_MODE             0.97
# 69   BASEMENTAREA_AVG             BASEMENTAREA_MEDI             0.99
# 96   YEARS_BEGINEXPLUATATION_AVG  YEARS_BEGINEXPLUATATION_MODE  0.97
# 110  YEARS_BEGINEXPLUATATION_AVG  YEARS_BEGINEXPLUATATION_MEDI  0.99
# 136  YEARS_BUILD_AVG              YEARS_BUILD_MODE              0.99

# 查看0.8到0.9
result = temp_df[temp_df[0].between(0.8, 0.9, inclusive='left')]
#      level_0               level_1                0
# 4    APARTMENTS_AVG        ELEVATORS_AVG          0.84
# 18   APARTMENTS_AVG        ELEVATORS_MODE         0.82
# 24   APARTMENTS_AVG        LIVINGAREA_MODE        0.89
# 32   APARTMENTS_AVG        ELEVATORS_MEDI         0.84
# 41   APARTMENTS_AVG        TOTALAREA_MODE         0.89
# 204  ELEVATORS_AVG         LIVINGAPARTMENTS_AVG   0.81
# 205  ELEVATORS_AVG         LIVINGAREA_AVG         0.87
# 208  ELEVATORS_AVG         APARTMENTS_MODE        0.81
# 218  ELEVATORS_AVG         LIVINGAPARTMENTS_MODE  0.8
# 219  ELEVATORS_AVG         LIVINGAREA_MODE        0.84
# 222  ELEVATORS_AVG         APARTMENTS_MEDI        0.83
# 232  ELEVATORS_AVG         LIVINGAPARTMENTS_MEDI  0.81
# 233  ELEVATORS_AVG         LIVINGAREA_MEDI        0.87
# 236  ELEVATORS_AVG         TOTALAREA_MODE         0.84
# 375  LIVINGAPARTMENTS_AVG  LIVINGAREA_AVG         0.88
# 389  LIVINGAPARTMENTS_AVG  LIVINGAREA_MODE        0.85
# 397  LIVINGAPARTMENTS_AVG  ELEVATORS_MEDI         0.81
# 403  LIVINGAPARTMENTS_AVG  LIVINGAREA_MEDI        0.88
# 406  LIVINGAPARTMENTS_AVG  TOTALAREA_MODE         0.85
# 409  LIVINGAREA_AVG        APARTMENTS_MODE        0.89
# 414  LIVINGAREA_AVG        ELEVATORS_MODE         0.85
# 419  LIVINGAREA_AVG        LIVINGAPARTMENTS_MODE  0.87
# 428  LIVINGAREA_AVG        ELEVATORS_MEDI         0.87
# 433  LIVINGAREA_AVG        LIVINGAPARTMENTS_MEDI  0.88
# 501  APARTMENTS_MODE       ELEVATORS_MODE         0.83
# 515  APARTMENTS_MODE       ELEVATORS_MEDI         0.81
# 521  APARTMENTS_MODE       LIVINGAREA_MEDI        0.89
# 524  APARTMENTS_MODE       TOTALAREA_MODE         0.86
# 631  ELEVATORS_MODE        LIVINGAPARTMENTS_MODE  0.81
# 632  ELEVATORS_MODE        LIVINGAREA_MODE        0.86
# 635  ELEVATORS_MODE        APARTMENTS_MEDI        0.83
# 645  ELEVATORS_MODE        LIVINGAPARTMENTS_MEDI  0.8
# 646  ELEVATORS_MODE        LIVINGAREA_MEDI        0.86

# 说明房产及居住变量之间存在很强的相关性，APARTMENTS_AVG APARTMENTS_MEDI的相关系数达到了1

del house_cols, house_corr, temp_df, upper_mask, result

# 4.社交圈变量

# 描述性统计
application_train['OBS_30_CNT_SOCIAL_CIRCLE'].describe()
# count    306490.000000
# mean          1.422245
# std           2.400989
# min           0.000000
# 25%           0.000000
# 50%           0.000000
# 75%           2.000000
# max         348.000000
# Name: OBS_30_CNT_SOCIAL_CIRCLE, dtype: float64

# 存在一个非常大的离群点384，去除之后绘图
sns.histplot(
    data=application_train[
        application_train['OBS_30_CNT_SOCIAL_CIRCLE'] != 348],
    x='OBS_30_CNT_SOCIAL_CIRCLE'
)
# 明显的强烈右偏

application_train['DEF_30_CNT_SOCIAL_CIRCLE'].describe()
# count    306490.000000
# mean          0.143421
# std           0.446698
# min           0.000000
# 25%           0.000000
# 50%           0.000000
# 75%           0.000000
# max          34.000000
# Name: DEF_30_CNT_SOCIAL_CIRCLE, dtype: float64

sns.histplot(
    data=application_train,
    x='DEF_30_CNT_SOCIAL_CIRCLE'
)
# 右偏

application_train['OBS_60_CNT_SOCIAL_CIRCLE'].describe()
# count    306490.000000
# mean          1.405292
# std           2.379803
# min           0.000000
# 25%           0.000000
# 50%           0.000000
# 75%           2.000000
# max         344.000000
# Name: OBS_60_CNT_SOCIAL_CIRCLE, dtype: float64

application_train['DEF_60_CNT_SOCIAL_CIRCLE'].describe()
# count    306490.000000
# mean          0.100049
# std           0.362291
# min           0.000000
# 25%           0.000000
# 50%           0.000000
# 75%           0.000000
# max          24.000000
# Name: DEF_60_CNT_SOCIAL_CIRCLE, dtype: float64

sns.histplot(
    data=application_train,
    x='DEF_60_CNT_SOCIAL_CIRCLE'
)

# 观察到的社交圈人数，合并长尾计数，观察与 TARGET 的关系
OBS_social_bins = [0, 1, 2, 3, 4, 5, np.inf]
OBS_social_labels = ['0人', '1人', '2人', '3人', '4人', '5人及以上']

OBS_30_CNT_SOCIAL_CIRCLE_summary = pd.DataFrame(
    {'bins': OBS_social_labels + ['特殊编码组']})
OBS_30_CNT_SOCIAL_CIRCLE_summary = bins_stats(
    bins=OBS_social_bins,
    labels=OBS_social_labels,
    column='OBS_30_CNT_SOCIAL_CIRCLE',
    df=OBS_30_CNT_SOCIAL_CIRCLE_summary)

#    bins        default_rate  population  pop_default
# 0  0人         7.90%         163910      12950
# 1  1人         7.98%         48783       3895
# 2  2人         8.25%         29808       2458
# 3  3人         8.24%         20322       1675
# 4  4人         8.82%         14143       1248
# 5  5人及以上   8.68%         29524       2563
# 6  特殊编码组  3.53%         1021        36

# 特殊编码组的违约率显著小于一般的违约率大小，是一个很强的风险信号
# 针对这种不严格单调的情况，后续评分卡模型的特征工程可以进一步进行箱子的合并

OBS_60_CNT_SOCIAL_CIRCLE_summary = pd.DataFrame(
    {'bins': OBS_social_labels + ['特殊编码组']})
OBS_60_CNT_SOCIAL_CIRCLE_summary = bins_stats(
    bins=OBS_social_bins,
    labels=OBS_social_labels,
    column='OBS_60_CNT_SOCIAL_CIRCLE',
    df=OBS_60_CNT_SOCIAL_CIRCLE_summary)

#    bins        default_rate  population  pop_default
# 0  0人         7.91%         164666      13021
# 1  1人         7.97%         48870       3894
# 2  2人         8.22%         29766       2448
# 3  3人         8.26%         20215       1670
# 4  4人         8.83%         13946       1232
# 5  5人及以上   8.70%         29027       2524
# 6  特殊编码组  3.53%         1021        36

# OBS_30_CNT_SOCIAL_CIRCLE_summary 和 OBS_60_CNT_SOCIAL_CIRCLE_summary
# 0人违约率约为7.9%，4人时约为8.8%，5人及以上略有下降，整体呈弱正相关但不严格单调
# 针对这种不严格单调的情况，后续评分卡模型的特征工程可以进一步进行箱子的合并

# 特殊编码组的违约率显著小于一般的违约率大小，是一个很强的风险信号

# 违约人数分布更稀疏，将2人及以上合并后观察与 TARGET 的关系
DEF_social_bins = [0, 1, 2, np.inf]
DEF_social_labels = ['0人', '1人', '2人及以上']

DEF_30_CNT_SOCIAL_CIRCLE_summary = pd.DataFrame(
    {'bins': DEF_social_labels + ['特殊编码组']})
DEF_30_CNT_SOCIAL_CIRCLE_summary = bins_stats(
    bins=DEF_social_bins,
    labels=DEF_social_labels,
    column='DEF_30_CNT_SOCIAL_CIRCLE',
    df=DEF_30_CNT_SOCIAL_CIRCLE_summary)

#    bins        default_rate  population  pop_default
# 0  0人         7.78%         271324      21098
# 1  1人         10.15%        28328       2874
# 2  2人及以上    11.95%        6838        817
# 3  特殊编码组   3.53%         1021        36

# 特殊编码组的违约率显著小于一般的违约率大小，是一个很强的风险信号

DEF_60_CNT_SOCIAL_CIRCLE_summary = pd.DataFrame(
    {'bins': DEF_social_labels + ['特殊编码组']})
DEF_60_CNT_SOCIAL_CIRCLE_summary = bins_stats(
    bins=DEF_social_bins,
    labels=DEF_social_labels,
    column='DEF_60_CNT_SOCIAL_CIRCLE',
    df=DEF_60_CNT_SOCIAL_CIRCLE_summary)

#    bins        default_rate  population  pop_default
# 0  0人         7.83%         280721      21994
# 1  1人         10.52%        21841       2297
# 2  2人及以上    12.68%        3928        498
# 3  特殊编码组   3.53%         1021        36

# 特殊编码组的违约率显著小于一般的违约率大小，是一个很强的风险信号

# DEF_30_CNT_SOCIAL_CIRCLE_summary 和 DEF_60_CNT_SOCIAL_CIRCLE_summary
# 违约人数越多，申请人自身违约率越高，存在较明显的正相关关系

# 5.电话变化与征信查询次数

# 分析 DAYS_LAST_PHONE_CHANGE，将距上次电话变更天数转换成年
(-application_train['DAYS_LAST_PHONE_CHANGE'] / 365).describe()
# count    307510.000000
# mean          2.637969
# std           2.265229
# min          -0.000000
# 25%           0.750685
# 50%           2.073973
# 75%           4.301370
# max          11.758904
# Name: DAYS_LAST_PHONE_CHANGE, dtype: float64

DAYS_LAST_PHONE_CHANGE_bins = [0, 1, 2, 5, np.inf]
DAYS_LAST_PHONE_CHANGE_labels = ['0~1年', '1~2年', '2~5年', '5年以上']
DAYS_LAST_PHONE_CHANGE_summary = pd.DataFrame(
    {'bins': DAYS_LAST_PHONE_CHANGE_labels + ['特殊编码组']})
DAYS_LAST_PHONE_CHANGE_summary = bins_stats(
    bins=DAYS_LAST_PHONE_CHANGE_bins,
    labels=DAYS_LAST_PHONE_CHANGE_labels,
    column='DAYS_LAST_PHONE_CHANGE',
    df=DAYS_LAST_PHONE_CHANGE_summary)

#    bins        default_rate  population  pop_default
# 0  0~1年       9.58%         92451       8854
# 1  1~2年       9.30%         57641       5363
# 2  2~5年       7.27%         105929      7698
# 3  5年以上     5.65%         51489       2910
# 4  特殊编码组  0.00%         1           0

# 距上次电话变更时间越久，违约率整体越低，关系较清晰

# AMT_REQ_CREDIT_BUREAU 系列变量的相关系数矩阵
AMT_REQ_CREDIT_BUREAU_columns = [
    'AMT_REQ_CREDIT_BUREAU_HOUR',
    'AMT_REQ_CREDIT_BUREAU_DAY',
    'AMT_REQ_CREDIT_BUREAU_WEEK',
    'AMT_REQ_CREDIT_BUREAU_MON',
    'AMT_REQ_CREDIT_BUREAU_QRT',
    'AMT_REQ_CREDIT_BUREAU_YEAR'
]

AMT_REQ_CREDIT_BUREAU_corr = application_train[
    AMT_REQ_CREDIT_BUREAU_columns
].corr().round(2)

#                          HOUR   DAY  WEEK   MON   QRT  YEAR
# HOUR                     1.00  0.23  0.00 -0.00 -0.00 -0.00
# DAY                      0.23  1.00  0.22 -0.01 -0.00 -0.00
# WEEK                     0.00  0.22  1.00 -0.01 -0.02  0.02
# MON                     -0.00 -0.01 -0.01  1.00 -0.01 -0.00
# QRT                     -0.00 -0.00 -0.02 -0.01  1.00  0.08
# YEAR                    -0.00 -0.00  0.02 -0.00  0.08  1.00

# 整体相关性较弱，HOUR 与 DAY、DAY 与 WEEK 的相关系数分别为0.23和0.22

application_train['AMT_REQ_CREDIT_BUREAU_HOUR'].describe()
# count    265992.000000
# mean          0.006402
# std           0.083849
# min           0.000000
# 25%           0.000000
# 50%           0.000000
# 75%           0.000000
# max           4.000000
# Name: AMT_REQ_CREDIT_BUREAU_HOUR, dtype: float64

application_train['AMT_REQ_CREDIT_BUREAU_DAY'].describe()
# count    265992.000000
# mean          0.007000
# std           0.110757
# min           0.000000
# 25%           0.000000
# 50%           0.000000
# 75%           0.000000
# max           9.000000
# Name: AMT_REQ_CREDIT_BUREAU_DAY, dtype: float64

application_train['AMT_REQ_CREDIT_BUREAU_WEEK'].describe()
# count    265992.000000
# mean          0.034362
# std           0.204685
# min           0.000000
# 25%           0.000000
# 50%           0.000000
# 75%           0.000000
# max           8.000000
# Name: AMT_REQ_CREDIT_BUREAU_WEEK, dtype: float64

application_train['AMT_REQ_CREDIT_BUREAU_MON'].describe()
# count    265992.000000
# mean          0.267395
# std           0.916002
# min           0.000000
# 25%           0.000000
# 50%           0.000000
# 75%           0.000000
# max          27.000000
# Name: AMT_REQ_CREDIT_BUREAU_MON, dtype: float64

application_train['AMT_REQ_CREDIT_BUREAU_QRT'].describe()
# count    265992.000000
# mean          0.265474
# std           0.794056
# min           0.000000
# 25%           0.000000
# 50%           0.000000
# 75%           0.000000
# max         261.000000
# Name: AMT_REQ_CREDIT_BUREAU_QRT, dtype: float64

application_train['AMT_REQ_CREDIT_BUREAU_YEAR'].describe()
# count    265992.000000
# mean          1.899974
# std           1.869295
# min           0.000000
# 25%           0.000000
# 50%           1.000000
# 75%           3.000000
# max          25.000000
# Name: AMT_REQ_CREDIT_BUREAU_YEAR, dtype: float64

# 小时、天、周查询次数的非零样本较少，合并为0次和1次及以上
AMT_REQ_CREDIT_BUREAU_short_bins = [0, 1, np.inf]
AMT_REQ_CREDIT_BUREAU_short_labels = ['0次', '1次及以上']

AMT_REQ_CREDIT_BUREAU_HOUR_summary = pd.DataFrame(
    {'bins': AMT_REQ_CREDIT_BUREAU_short_labels + ['特殊编码组']})
AMT_REQ_CREDIT_BUREAU_HOUR_summary = bins_stats(
    bins=AMT_REQ_CREDIT_BUREAU_short_bins,
    labels=AMT_REQ_CREDIT_BUREAU_short_labels,
    column='AMT_REQ_CREDIT_BUREAU_HOUR',
    df=AMT_REQ_CREDIT_BUREAU_HOUR_summary)

#    bins        default_rate  population  pop_default
# 0  0次         7.72%         264366      20402
# 1  1次及以上    8.06%         1626        131
# 2  特殊编码组   10.34%        41519       4292

AMT_REQ_CREDIT_BUREAU_DAY_summary = pd.DataFrame(
    {'bins': AMT_REQ_CREDIT_BUREAU_short_labels + ['特殊编码组']})
AMT_REQ_CREDIT_BUREAU_DAY_summary = bins_stats(
    bins=AMT_REQ_CREDIT_BUREAU_short_bins,
    labels=AMT_REQ_CREDIT_BUREAU_short_labels,
    column='AMT_REQ_CREDIT_BUREAU_DAY',
    df=AMT_REQ_CREDIT_BUREAU_DAY_summary)

#    bins        default_rate  population  pop_default
# 0  0次         7.71%         264503      20392
# 1  1次及以上    9.47%         1489        141
# 2  特殊编码组   10.34%        41519       4292

AMT_REQ_CREDIT_BUREAU_WEEK_summary = pd.DataFrame(
    {'bins': AMT_REQ_CREDIT_BUREAU_short_labels + ['特殊编码组']})
AMT_REQ_CREDIT_BUREAU_WEEK_summary = bins_stats(
    bins=AMT_REQ_CREDIT_BUREAU_short_bins,
    labels=AMT_REQ_CREDIT_BUREAU_short_labels,
    column='AMT_REQ_CREDIT_BUREAU_WEEK',
    df=AMT_REQ_CREDIT_BUREAU_WEEK_summary)

#    bins        default_rate  population  pop_default
# 0  0次         7.72%         257456      19865
# 1  1次及以上    7.83%         8536        668
# 2  特殊编码组   10.34%        41519       4292

# 月度、季度查询次数按0、1、2、3次及以上分箱
AMT_REQ_CREDIT_BUREAU_period_bins = [0, 1, 2, 3, np.inf]
AMT_REQ_CREDIT_BUREAU_period_labels = ['0次', '1次', '2次', '3次及以上']

AMT_REQ_CREDIT_BUREAU_MON_summary = pd.DataFrame(
    {'bins': AMT_REQ_CREDIT_BUREAU_period_labels + ['特殊编码组']})
AMT_REQ_CREDIT_BUREAU_MON_summary = bins_stats(
    bins=AMT_REQ_CREDIT_BUREAU_period_bins,
    labels=AMT_REQ_CREDIT_BUREAU_period_labels,
    column='AMT_REQ_CREDIT_BUREAU_MON',
    df=AMT_REQ_CREDIT_BUREAU_MON_summary)

#    bins        default_rate  population  pop_default
# 0  0次         7.79%         222233      17315
# 1  1次         7.72%         33147       2560
# 2  2次         7.04%         5386        379
# 3  3次及以上    5.34%         5226        279
# 4  特殊编码组   10.34%        41519       4292

AMT_REQ_CREDIT_BUREAU_QRT_summary = pd.DataFrame(
    {'bins': AMT_REQ_CREDIT_BUREAU_period_labels + ['特殊编码组']})
AMT_REQ_CREDIT_BUREAU_QRT_summary = bins_stats(
    bins=AMT_REQ_CREDIT_BUREAU_period_bins,
    labels=AMT_REQ_CREDIT_BUREAU_period_labels,
    column='AMT_REQ_CREDIT_BUREAU_QRT',
    df=AMT_REQ_CREDIT_BUREAU_QRT_summary)

#    bins        default_rate  population  pop_default
# 0  0次         7.85%         215417      16905
# 1  1次         6.58%         33862       2227
# 2  2次         8.35%         14412       1203
# 3  3次及以上    8.60%         2301        198
# 4  特殊编码组   10.34%        41519       4292

# 年度查询次数保留0到4次，5次及以上合并长尾
AMT_REQ_CREDIT_BUREAU_YEAR_bins = [0, 1, 2, 3, 4, 5, np.inf]
AMT_REQ_CREDIT_BUREAU_YEAR_labels = [
    '0次', '1次', '2次', '3次', '4次', '5次及以上'
]
AMT_REQ_CREDIT_BUREAU_YEAR_summary = pd.DataFrame(
    {'bins': AMT_REQ_CREDIT_BUREAU_YEAR_labels + ['特殊编码组']})
AMT_REQ_CREDIT_BUREAU_YEAR_summary = bins_stats(
    bins=AMT_REQ_CREDIT_BUREAU_YEAR_bins,
    labels=AMT_REQ_CREDIT_BUREAU_YEAR_labels,
    column='AMT_REQ_CREDIT_BUREAU_YEAR',
    df=AMT_REQ_CREDIT_BUREAU_YEAR_summary)

#    bins        default_rate  population  pop_default
# 0  0次         7.13%         71801       5123
# 1  1次         7.33%         63405       4650
# 2  2次         8.10%         50192       4068
# 3  3次         7.96%         33628       2676
# 4  4次         8.26%         20714       1710
# 5  5次及以上    8.78%         26252       2306
# 6  特殊编码组   10.34%        41519       4292

# HOUR、DAY、WEEK的非零组样本较少；MON、QRT违约率非单调，YEAR总体随查询次数增加而上升但并非严格单调
# 保留缺失独立分箱，其余考虑合并分箱

"筛选类别变量"
object_columns = application_train.select_dtypes(include='object').columns

# 分析 NAME_CONTRACT_TYPE 字段，该字段无缺失值
application_train['NAME_CONTRACT_TYPE'].value_counts()
# NAME_CONTRACT_TYPE
# Cash loans         278232
# Revolving loans     29279
# Name: count, dtype: int64

# 分析不同类别的违约率
diff_type_default_rate('NAME_CONTRACT_TYPE')
# ('Cash loans', '8.35%'), ('Revolving loans', '5.48%')

# NAME_TYPE_SUITE
application_train['NAME_TYPE_SUITE'].value_counts()
# NAME_TYPE_SUITE
# Unaccompanied      248526
# Family              40149
# Spouse, partner     11370
# Children             3267
# Other_B              1770
# Other_A               866
# Group of people       271
# Name: count, dtype: int64

diff_type_default_rate('NAME_TYPE_SUITE')
# ('Unaccompanied', '8.18%'), ('Family', '7.49%'), ('Spouse, partner', '7.87%'),
# ('Children', '7.38%'), ('Other_B', '9.83%'), ('Other_A', '8.78%'), ('Group of people', '8.49%')

# NAME_INCOME_TYPE
application_train['NAME_INCOME_TYPE'].value_counts()
# NAME_INCOME_TYPE
# Working                 158774
# Commercial associate     71617
# Pensioner                55362
# State servant            21703
# Unemployed                  22
# Student                     18
# Businessman                 10
# Maternity leave              5
# Name: count, dtype: int64
diff_type_default_rate('NAME_INCOME_TYPE')
# ('Working', '9.59%'), ('Commercial associate', '7.48%'), ('Pensioner', '5.39%'),
# ('State servant', '5.75%'), ('Unemployed', '36.36%'), ('Student', '0.00%'),
# ('Businessman', '0.00%'), ('Maternity leave', '40.00%')

# 部分种类的数量太少，后续特征工程进一步将类别合并

# CODE_GENDER
application_train['CODE_GENDER'].value_counts()
# CODE_GENDER
# F      202448
# M      105059
# XNA         4
# Name: count, dtype: int64

diff_type_default_rate('CODE_GENDER')
# ('F', '7.00%'), ('M', '10.14%'), ('XNA', '0.00%')

# FLAG_OWN_CAR
application_train['FLAG_OWN_CAR'].value_counts()
# FLAG_OWN_CAR
# N    202924
# Y    104587
# Name: count, dtype: int64

diff_type_default_rate('FLAG_OWN_CAR')
# ('N', '8.50%'), ('Y', '7.24%')

# FLAG_OWN_REALTY
application_train['FLAG_OWN_REALTY'].value_counts()
# FLAG_OWN_REALTY
# Y    213312
# N     94199
# Name: count, dtype: int64

diff_type_default_rate('FLAG_OWN_REALTY')
# ('Y', '7.96%'), ('N', '8.32%')

# NAME_EDUCATION_TYPE
application_train['NAME_EDUCATION_TYPE'].value_counts()
# NAME_EDUCATION_TYPE
# Secondary / secondary special    218391
# Higher education                  74863
# Incomplete higher                 10277
# Lower secondary                    3816
# Academic degree                     164
# Name: count, dtype: int64

diff_type_default_rate('NAME_EDUCATION_TYPE')
# ('Secondary / secondary special', '8.94%'), ('Higher education', '5.36%'),
# ('Incomplete higher', '8.48%'), ('Lower secondary', '10.93%'),
# ('Academic degree', '1.83%')

# NAME_FAMILY_STATUS
application_train['NAME_FAMILY_STATUS'].value_counts()
# NAME_FAMILY_STATUS
# Married                  196432
# Single / not married      45444
# Civil marriage            29775
# Separated                 19770
# Widow                     16088
# Unknown                       2
# Name: count, dtype: int64

diff_type_default_rate('NAME_FAMILY_STATUS')
# ('Married', '7.56%'), ('Single / not married', '9.81%'),
# ('Civil marriage', '9.94%'), ('Separated', '8.19%'),
# ('Widow', '5.82%'), ('Unknown', '0.00%')

# 以上变量均无缺失值。男性违约率高于女性；有车客户的违约率较低，是否有房产的差异较小。
# 教育程度和家庭状态的违约率存在差异，但 Academic degree、XNA 和 Unknown 等类别样本极少，
# 其违约率不稳定，不宜单独据此下结论，后续特征工程可考虑合并稀有类别。
