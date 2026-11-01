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
import toad

from optbinning import BinningProcess

# 文件路径
DATA_PATH = Path.cwd() / "data"

# 主申请表
application_train = pd.read_csv(DATA_PATH / 'raw' / 'application_train.csv')
application_test = pd.read_csv(DATA_PATH / 'raw' / 'application_test.csv')

# cols_meaning = pd.read_csv(
#     DATA_PATH / 'HomeCredit_columns_description.csv',
#     usecols=['Table', 'Row', 'Description'],
#     encoding='latin1'
# )
# cols_meaning = cols_meaning.loc[
#     cols_meaning['Table'] == 'application_{train|test}.csv'
# ].reset_index(drop=True)

cols_eda = pd.read_csv(DATA_PATH / 'interim' / 'cols_eda.csv')

# %%
"一、样本整体盘点与客群对齐（Sample & Population Check）"

# -- 数据规模 --
application_train.shape
# (307511, 122)
# application_train 样本量为307511，含有122个字段

application_test.shape
# (48744, 121)
# application_test 样本量为48744，含有121个字段

# 检查测试集字段是否恰好等于训练集业务特征字段
set(application_test.columns) == (set(application_train.columns) - {'TARGET'})
# True
# 字段一致

# -- 重复与唯一性 --
application_train.duplicated().any()
# False
application_train.duplicated(subset='SK_ID_CURR').any()
# False
application_test.duplicated().any()
# False
" -> 无重复样本、无重复主键"

# %%
"二、目标变量探索与好坏定义（Target EDA）"

# 当前数据已给定 TARGET：1=违约，0=正常
application_train['TARGET'].isna().sum()
# 0
set(application_train['TARGET'].dropna().unique()).issubset({0, 1})
# True

# 好坏样本数量
application_train['TARGET'].value_counts(dropna=False)
# TARGET
# 0    282686
# 1     24825
# Name: count, dtype: int64

# 整体坏账率
f"{application_train['TARGET'].mean():.2%}"
# 8.07%
# -> 样本存在类别不平衡

# %%
"三、数据质量与单变量清洗（Univariate & Data Quality）"

# -- 数据类型与字段说明 --
cols_train_types = application_train.dtypes.rename('dtypes')
cols_test_types = application_test.dtypes.rename('dtypes')

# cols_eda = cols_meaning.copy()
cols_eda['dtypes'] = cols_eda['Row'].map(cols_train_types)

# 检查训练集与测试集业务特征字段类型是否一致
cols_train_types.drop('TARGET').equals(cols_test_types)
# True

# 主键不作为业务特征，标签也不进入单变量特征质量筛查
feature_columns = [
    column for column in application_train.columns
    if column not in {'SK_ID_CURR', 'TARGET'}
]
numeric_feature_columns = application_train[feature_columns].select_dtypes(
    include='number'
).columns.tolist()

# -- 形成单变量质量汇总：缺失率、最大同值率、分位数 --
# 阈值只用于待审查清单：缺失率 >90%、最大同值率 >95% 优先复核；不能脱离业务含义直接删特征。
univariate_summary = pd.DataFrame(index=feature_columns)
univariate_summary.index.name = 'Row'
univariate_summary['missing_rate'] = application_train[feature_columns].isna().mean()
univariate_summary['primary_value_ratio'] = pd.Series({
    column: application_train[column].value_counts(dropna=False).iloc[0] / len(application_train)
    for column in feature_columns
})

numeric_moments = application_train[numeric_feature_columns].agg(
    ['count', 'mean', 'std']
).T
univariate_summary = univariate_summary.join(numeric_moments)

# -- 异常值识别 --
# 计算数值字段的0.01、0.05、0.5、0.95和0.99分位数
numeric_quantiles = application_train[numeric_feature_columns].quantile(
    [0.01, 0.05, 0.50, 0.95, 0.99]
).T
numeric_quantiles.columns = ['p1', 'p5', 'p50', 'p95', 'p99']
univariate_summary = univariate_summary.join(numeric_quantiles)

cols_eda = cols_eda.merge(
    univariate_summary.reset_index(),
    on='Row',
    how='left'
)

# 在 DAYS_EMPLOYED 中发现 365243 特殊编码

# -- 缺失率分析 --
missing_summary = application_train.isna().mean().reset_index()
missing_summary.columns = ['Row', 'missing_rate']

# 没有缺失的列
present_columns = missing_summary.loc[missing_summary['missing_rate'] == 0, 'Row'].tolist()

# 缺失的列
# 保留完整汇总表，另用非零缺失率子集进行后续缺失值分析
missing_summary_nonzero = missing_summary[
    missing_summary['missing_rate'] > 0
].reset_index(drop=True)
missing_columns = missing_summary_nonzero['Row'].tolist()

missing_summary_nonzero.sort_values(by='missing_rate', ascending=False).head(20)
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

# 缺失率相同的字段组（由当前 application_train 统计，按缺失率降序）
# 缺失率相同只表示缺失比例相同，缺失位置是否一致由下方逻辑继续核验
# 69.8723%: COMMONAREA_AVG, COMMONAREA_MODE, COMMONAREA_MEDI
# 69.4330%: NONLIVINGAPARTMENTS_AVG, NONLIVINGAPARTMENTS_MODE, NONLIVINGAPARTMENTS_MEDI
# 68.3549%: LIVINGAPARTMENTS_AVG, LIVINGAPARTMENTS_MODE, LIVINGAPARTMENTS_MEDI
# 67.8486%: FLOORSMIN_AVG, FLOORSMIN_MODE, FLOORSMIN_MEDI
# 66.4978%: YEARS_BUILD_AVG, YEARS_BUILD_MODE, YEARS_BUILD_MEDI
# 59.3767%: LANDAREA_AVG, LANDAREA_MODE, LANDAREA_MEDI
# 58.5160%: BASEMENTAREA_AVG, BASEMENTAREA_MODE, BASEMENTAREA_MEDI
# 55.1792%: NONLIVINGAREA_AVG, NONLIVINGAREA_MODE, NONLIVINGAREA_MEDI
# 53.2960%: ELEVATORS_AVG, ELEVATORS_MODE, ELEVATORS_MEDI
# 50.7497%: APARTMENTS_AVG, APARTMENTS_MODE, APARTMENTS_MEDI
# 50.3488%: ENTRANCES_AVG, ENTRANCES_MODE, ENTRANCES_MEDI
# 50.1933%: LIVINGAREA_AVG, LIVINGAREA_MODE, LIVINGAREA_MEDI
# 49.7608%: FLOORSMAX_AVG, FLOORSMAX_MODE, FLOORSMAX_MEDI
# 48.7810%: YEARS_BEGINEXPLUATATION_AVG, YEARS_BEGINEXPLUATATION_MODE, YEARS_BEGINEXPLUATATION_MEDI
# 13.5016%: AMT_REQ_CREDIT_BUREAU_HOUR, AMT_REQ_CREDIT_BUREAU_DAY, AMT_REQ_CREDIT_BUREAU_WEEK,
#           AMT_REQ_CREDIT_BUREAU_MON, AMT_REQ_CREDIT_BUREAU_QRT, AMT_REQ_CREDIT_BUREAU_YEAR
# 0.3320%: OBS_30_CNT_SOCIAL_CIRCLE, DEF_30_CNT_SOCIAL_CIRCLE,
#          OBS_60_CNT_SOCIAL_CIRCLE, DEF_60_CNT_SOCIAL_CIRCLE

# 进一步探究缺失模式
for rate, group in missing_summary_nonzero.groupby('missing_rate'):
    if len(group) > 1:
        temp_list = group['Row'].tolist()
        temp_df = application_train[temp_list]
        
        if len(temp_df.isna().T.drop_duplicates()) != 1:
            print(f"{temp_list}的缺失模式不同")

# 没有输出结果

" -> 相同缺失率的字段缺失模式都相同"
" 每个缺失率相同的字段组内部，字段的缺失标记在每条样本上都完全一致 "
" 也就是说，一条样本中组内字段要么全都有值，要么全缺失；不会出现组内某个字段缺失、另一个字段有值的情况 "

" -> 这提示这些字段的可用性可能受同一个数据采集或数据来源条件影响 "
" 对特征工程来说，组内的缺失指示变量会重复表达同一信息，通常一个组保留一个 is_missing 标记就够了；"
" 原字段的值仍可能各自有用，不应仅因缺失模式相同就删除 "

missing_summary_nonzero = missing_summary_nonzero.merge(
    cols_eda[['Row', 'Description']],
    on='Row',
    how='left')

''' -> 缺失率较高的字段大部分是与房屋相关
    -> 说明一部分申请人缺少相关房屋信息，而不是数据随机缺失 '''

# 输出缺失率排名图
# sns.barplot(
#     data=missing_summary_nonzero.sort_values(by='missing_rate', ascending=False).head(30),
#     x='missing_rate',
#     y='Row')

# sns.barplot(
#     data=missing_summary_nonzero.loc[
#         ~missing_summary_nonzero['Description'].str.startswith('Normalized'),
#         ['Row', 'missing_rate']
#         ].sort_values(by='missing_rate', ascending=False),
#     x='missing_rate',
#     y='Row')

cols_eda.to_csv(DATA_PATH / 'interim' / 'cols_eda.csv', index=False)

# %%
"四、双变量分析与风险单调性检验（Bivariate & Risk Discrimination）"
# 先对 TARGET 以外的全部业务特征做统一初筛，再按 IV 降序查看，最后才附加业务类别
# 使用 optbinning 对数值和类别变量批量最优分箱，缺失与常见特殊编码单独处理
# 这是开发样本上的初筛 IV

# %%
"4.1 全量业务特征 IV 初筛与降序排名"
# 1. 区分特征列表与类别型变量
ignore_cols = ['SK_ID_CURR', 'TARGET']
feature_cols = [c for c in application_train.columns if c not in ignore_cols]
cat_cols = application_train[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()

# 2. 配置特殊值编码（Home Credit 中 365243 代表退休/无业特殊编码，XNA 代表未知类别）
# 也可传入字典针对不同特征指定不同特殊值，如：{'DAYS_EMPLOYED': [365243], 'CODE_GENDER': ['XNA']}
special_codes = [365243, 'XNA']

# 3. 配置底层分箱规则（透传给每个特征的 OptimalBinning）
# EDA 阶段底层分箱配置：放开单调性限制，允许发现峰值(peak)/谷值(valley)
eda_fit_params = {
    col: {
        "monotonic_trend": "auto",          # 强制严格单调递增或递减（若允许U型可设为 "auto"）
                                            # EDA 摸底阶段设置为auto，发现真实的业务规律以及快速算出特征的理论最大 IV 上限
        #"min_n_bins":                      # 这里不设置最小分箱数量，防范高同值率特征导致求解器报错或强行切分，
                                            # 并直接利用 n_bins == 1 快速筛出无区分度的废特征
        "cat_cutoff": 0.05                  # 类别型变量中占比低于 5% 的长尾稀有类别自动合并为 "others"
    } 
    for col in feature_cols
}

# 4.实例化 EDA 专用 BinningProcess
binning_process = BinningProcess(
    variable_names=feature_cols,                 # 需要批量分箱的特征列名列表
    categorical_variables=cat_cols,              # 显式指定类别型或离散状态特征
    max_n_bins=8,                                # 最大分箱数（不含缺失箱和特殊值箱），评分卡通常限制在 5~6 箱以防过拟合
                                                 # EDA 阶段切 8~10 箱，观察更细腻的风险趋势
    min_bin_size=0.02,                           # 单个分箱的最小样本占比，一般设置5%，放宽至 2%，捕捉尾部极端风险
    special_codes=special_codes,                 # 特殊编码列表
                                                 # 命中这些值的样本会脱离常规分箱
    binning_fit_params=eda_fit_params,
    selection_criteria=None,                     # EDA 阶段不剔除任何变量，全量输出体检结果
    n_jobs=-1
)

binning_process.fit(
    application_train[feature_cols],
    application_train['TARGET']
)

# summary() 提供各变量的 IV、分箱数及求解状态
iv_summary = binning_process.summary().rename(columns={'name': 'feature'})

iv_summary = iv_summary.sort_values(by='iv', ascending=False).reset_index(drop=True)

# 结果加入cols_eda
cols_eda = cols_eda.join(iv_summary.set_index('feature'), on='Row')

cols_eda.to_csv(DATA_PATH / 'interim' / 'cols_eda.csv', index=False)

# 清理本单元格的临时配置变量；保留 IV 结果及后续 PSI 所需对象。
del ignore_cols, cat_cols, special_codes, eda_fit_params

# %%
"4.2 IV 达标特征清单"
high_iv_features = cols_eda.loc[cols_eda['iv'] >= 0.02, ['Row', 'iv']]
# 按iv值大小降序排列
# 	Row	iv
# 43	EXT_SOURCE_3	0.33448379020482355
# 42	EXT_SOURCE_2	0.31813665994270685
# 41	EXT_SOURCE_1	0.14639290012546213
# 18	DAYS_EMPLOYED	0.11399391282106762
# 10	AMT_GOODS_PRICE	0.09203198201470272
# 17	DAYS_BIRTH	0.08653932081929969
# 28	OCCUPATION_TYPE	0.06104343197256081
# 8	AMT_CREDIT	0.05931096593251868
# 12	NAME_INCOME_TYPE	0.057896106859146104
# 31	REGION_RATING_CLIENT_W_CITY	0.05120149755009072
# 13	NAME_EDUCATION_TYPE	0.049224597800298246
# 30	REGION_RATING_CLIENT	0.04831690185425046
# 95	DAYS_LAST_PHONE_CHANGE	0.04774858490596422
# 40	ORGANIZATION_TYPE	0.04282940064428906
# 20	DAYS_ID_PUBLISH	0.04016654108956729
# 3	CODE_GENDER	0.038597117741972505
# 51	FLOORSMAX_AVG	0.038354500367963174
# 79	FLOORSMAX_MEDI	0.03808228920708525
# 65	FLOORSMAX_MODE	0.03761695808054861
# 88	TOTALAREA_MODE	0.03726883394314958
# 16	REGION_POPULATION_RELATIVE	0.03564908843629902
# 55	LIVINGAREA_AVG	0.03433110147414867
# 83	LIVINGAREA_MEDI	0.03414786950313495
# 49	ELEVATORS_AVG	0.033216838210960886
# 69	LIVINGAREA_MODE	0.03320633184971729
# 23	FLAG_EMP_PHONE	0.032907128220936985
# 77	ELEVATORS_MEDI	0.03289580029009964
# 44	APARTMENTS_AVG	0.03262996981429838
# 72	APARTMENTS_MEDI	0.032328573857287865
# 38	REG_CITY_NOT_WORK_CITY	0.03220724394929676
# 63	ELEVATORS_MODE	0.03200070270485087
# 58	APARTMENTS_MODE	0.03176301782579833
# 9	AMT_ANNUITY	0.030689010660250062
# 74	YEARS_BEGINEXPLUATATION_MEDI	0.029325962819168466
# 46	YEARS_BEGINEXPLUATATION_AVG	0.029115646330580957
# 60	YEARS_BEGINEXPLUATATION_MODE	0.029102029395255802
# 97	FLAG_DOCUMENT_3	0.028579108192551952
# 19	DAYS_REGISTRATION	0.02848639824108758
# 50	ENTRANCES_AVG	0.02770121067094218
# 78	ENTRANCES_MEDI	0.027470830650918418
# 64	ENTRANCES_MODE	0.026449632261004496
# 89	WALLSMATERIAL_MODE	0.024996558933043922
# 90	EMERGENCYSTATE_MODE	0.024043969836771
# 57	NONLIVINGAREA_AVG	0.023089334798925144
# 85	NONLIVINGAREA_MEDI	0.022855664592085406
# 45	BASEMENTAREA_AVG	0.02263941554937595
# 73	BASEMENTAREA_MEDI	0.022603006895192762
# 71	NONLIVINGAREA_MODE	0.022591676410422937
# 87	HOUSETYPE_MODE	0.022402813814709353
# 37	REG_CITY_NOT_LIVE_CITY	0.022298689071670597
# 21	OWN_CAR_AGE	0.02206093456689807
# 59	BASEMENTAREA_MODE	0.02176626152408486
# 14	NAME_FAMILY_STATUS	0.021697828732442588

# 后续进一步做相关性分析

# %%
"五、跨时间稳定性与数据穿越排查（Stability & Leakage Check）"
# 由02的对抗验证结果，AUC=0.89 表明测试集与验证集有强烈的分布偏移
#                        feature     importance
# 6                   AMT_CREDIT  287596.707496
# 7                  AMT_ANNUITY  200538.586412
# 117  AMT_REQ_CREDIT_BUREAU_MON  183063.273897
# 118  AMT_REQ_CREDIT_BUREAU_QRT  157605.925762
# 8              AMT_GOODS_PRICE  114633.155271
# 0           NAME_CONTRACT_TYPE   91235.675193
# 38           ORGANIZATION_TYPE   57943.737944
# 25                  FLAG_EMAIL   57537.064442
# 39                EXT_SOURCE_1   52712.000303
# 93      DAYS_LAST_PHONE_CHANGE   42720.664838

# 补充单变量的PSI：用训练集拟合出的同一套分箱转换训练集和测试集
train_bins = pd.DataFrame(
    binning_process.transform(
        application_train[feature_cols],
        metric="bins"
    ),
    columns=feature_cols,
    index=application_train.index
)
test_bins = pd.DataFrame(
    binning_process.transform(
        application_test[feature_cols],
        metric="bins"
    ),
    columns=feature_cols,
    index=application_test.index
)


# 在相同分箱标签上直接计算 PSI（toad 参数顺序为 test、base）。
psi_summary = toad.metrics.PSI(
    test_bins[feature_cols],
    train_bins[feature_cols]
).sort_values(ascending=False)

# 筛选psi
# 0.10 <= PSI <= 0.25：轻微偏移，有条件保留；结合业务原因排查，必要时合并分箱或加强监控。
# PSI > 0.25：显著偏移（不稳定），原则上删除；若为核心强特征，先进行特征改造，再重新评估。
psi_summary[psi_summary > 0.1]

# AMT_REQ_CREDIT_BUREAU_MON    0.429540
# AMT_REQ_CREDIT_BUREAU_QRT    0.310101
# NAME_CONTRACT_TYPE           0.211138
# FLAG_EMAIL                   0.124205

# 添加结果到cols_eda
cols_eda['psi'] = cols_eda['Row'].map(psi_summary)

cols_eda.to_csv(DATA_PATH / 'interim' / 'cols_eda.csv', index=False)

# 清理 PSI 计算使用的临时分箱数据；保留 psi_summary 和 cols_eda 结果。
del train_bins, test_bins

# %%
# 执行漏斗初筛后进行相关性分析

# 1.数据质量维度（基础过滤）
# -- 缺失率 --
# 逻辑回归评分卡通常剔除缺失率大于 70% 至 80% 的特征；
# XGBoost 或 LightGBM 等树模型因天然支持缺失值分裂，阈值常放宽至 90% 至 95%
# 若“是否缺失”本身代表某种客群特征（例如未查得征信记录），即使高缺失也应先计算 IV 值再决定去留

# -- 同值率 --
# 当某个特征的单一取值（不含缺失值）占比超过 90% 至 95% 时，说明该变量缺乏方差，区分度极低，通常直接剔除

# -- 方差 --
# 连续型变量若方差接近 0，直接过滤

# 2.预测区分能力（核心过滤）
# -- IV 信息值评估标准 --
# IV < 0.02，几乎无预测能力，粗筛直接剔除
# 0.02 <= IV < 0.1，弱预测能力，视特征池丰富程度决定是否保留
# 0.1 <= IV < 0.3，中等预测能力，优先保留
# 0.3 <= IV < 0.5，强预测能力，核心入模候选特征
# IV >= 0.5，区分度过高，需重点排查是否存在数据穿越

# 对于低 IV 特征，优先做业务交叉衍生，完成上述交叉衍生后，原特征本身的任务就完成了
# 进入评分卡的特征初筛环节时，直接从候选特征池中剔除，不对其进行 WOE 转换，也不送入后续的多变量分析

# -- PSI 群体稳定性评估标准 --
# PSI < 0.10：非常稳定，特征分布无明显变化，可直接保留。
# 0.10 <= PSI < 0.25：轻微偏移，有条件保留；结合业务原因排查，必要时合并分箱或加强监控。
# PSI >= 0.25：显著偏移（不稳定），原则上删除；若为核心强特征，先进行特征改造，再重新评估。

# -- Pearson / Spearman相关系数 --
# 当两个特征的相关系数绝对值大于 0.70 或 0.80 时，视为高度共线
# 通常采取“优胜劣汰”原则：对比两者的 IV 值和 PSI，保留 IV 更高或 PSI 更低的那个，剔除另一个

# 第一层：剔除极高同值特征，同值率大于0.95
ignore_cols = ['SK_ID_CURR', 'TARGET']
feature_cols = [c for c in application_train.columns if c not in ignore_cols]

high_primary_value_ratio = cols_eda.loc[cols_eda['primary_value_ratio'] > 0.95, ['Row', 'primary_value_ratio']]

# 第二层：剔除跨期不稳定特征，PSI 大于0.25
high_psi = cols_eda.loc[cols_eda['psi'] >= 0.25, ['Row', 'psi']]

# 第三层：剔除无区分度特征，IV 低于0.02
low_iv = cols_eda.loc[cols_eda['iv'] < 0.02, ['Row', 'iv']]

# 一共剔除的特征
first_delete_features = set(high_primary_value_ratio['Row'].tolist() + high_psi['Row'].tolist() + low_iv['Row'].tolist())

model_features = [col for col in feature_cols
                  if col not in first_delete_features]

model_features_summary = cols_eda[cols_eda['Row'].isin(model_features)].reset_index(drop=True)

# 清理漏斗筛选的中间表；保留最终剔除清单和后续分析所需结果。
del high_primary_value_ratio, high_psi, low_iv

# %%
"六、多变量共线性与相关性分析（Multivariate Analysis）"
# 在model_features_summary中有53个特征，其中有大量的房屋特征，进行去重

# 第一步进行EDA 粗筛，使用原始特征值
house_cols = model_features_summary.loc[
    model_features_summary['Description'].str.startswith('Normalized information'), 'Row'
    ].tolist()
# 一共28个房屋特征

# 筛选数值类型的房屋特征
house_num_features = model_features_summary[
    model_features_summary['Description'].str.startswith('Normalized information') &
    (model_features_summary['dtypes'] == 'float64')
    ]
house_num_cols = house_num_features['Row'].sort_values().tolist()
# 一共25个数值房屋特征

# 计算Pearson相关性矩阵
house_corr = application_train[house_num_cols].corr()

# 结合贪心共线性剔除法（Greedy Collinearity Elimination，即按 IV 从高到低排序，依次剔除与高 IV 特征相关系数超标的低 IV 特征）
# 第一步剔除同源后缀冗余

prefix = house_corr.index.to_series().str.split('_').str[0]
prefix = prefix.drop_duplicates().reset_index(drop=True)
for i in prefix:
    rows = house_corr.index.str.startswith(i)
    cols = house_corr.columns.str.startswith(i)
    print(house_corr.loc[rows, cols])

del i, rows, cols, prefix
#                  APARTMENTS_AVG  APARTMENTS_MEDI  APARTMENTS_MODE
# APARTMENTS_AVG         1.000000         0.995081         0.973259
# APARTMENTS_MEDI        0.995081         1.000000         0.977193
# APARTMENTS_MODE        0.973259         0.977193         1.000000
#                    BASEMENTAREA_AVG  BASEMENTAREA_MEDI  BASEMENTAREA_MODE
# BASEMENTAREA_AVG           1.000000           0.994317           0.973496
# BASEMENTAREA_MEDI          0.994317           1.000000           0.977938
# BASEMENTAREA_MODE          0.973496           0.977938           1.000000
#                 ELEVATORS_AVG  ELEVATORS_MEDI  ELEVATORS_MODE
# ELEVATORS_AVG        1.000000        0.996099        0.978837
# ELEVATORS_MEDI       0.996099        1.000000        0.982828
# ELEVATORS_MODE       0.978837        0.982828        1.000000
#                 ENTRANCES_AVG  ENTRANCES_MEDI  ENTRANCES_MODE
# ENTRANCES_AVG        1.000000        0.996886        0.977743
# ENTRANCES_MEDI       0.996886        1.000000        0.980677
# ENTRANCES_MODE       0.977743        0.980677        1.000000
#                 FLOORSMAX_AVG  FLOORSMAX_MEDI  FLOORSMAX_MODE
# FLOORSMAX_AVG        1.000000        0.997034        0.985689
# FLOORSMAX_MEDI       0.997034        1.000000        0.988237
# FLOORSMAX_MODE       0.985689        0.988237        1.000000
#                  LIVINGAREA_AVG  LIVINGAREA_MEDI  LIVINGAREA_MODE
# LIVINGAREA_AVG         1.000000         0.995596         0.972050
# LIVINGAREA_MEDI        0.995596         1.000000         0.974743
# LIVINGAREA_MODE        0.972050         0.974743         1.000000
#                     NONLIVINGAREA_AVG  NONLIVINGAREA_MEDI  NONLIVINGAREA_MODE
# NONLIVINGAREA_AVG            1.000000            0.990444            0.966087
# NONLIVINGAREA_MEDI           0.990444            1.000000            0.975839
# NONLIVINGAREA_MODE           0.966087            0.975839            1.000000
#                 TOTALAREA_MODE
# TOTALAREA_MODE             1.0
#                               YEARS_BEGINEXPLUATATION_AVG  ...  YEARS_BEGINEXPLUATATION_MODE
# YEARS_BEGINEXPLUATATION_AVG                      1.000000  ...                      0.971893
# YEARS_BEGINEXPLUATATION_MEDI                     0.993825  ...                      0.963539
# YEARS_BEGINEXPLUATATION_MODE                     0.971893  ...                      1.000000

# 观察矩阵可以发现，同一物理属性下的 _AVG（均值）、_MEDI（中位数）和 _MODE（众数）相关系数全部在 0.9635 至 0.9970 之间，属于完全等价的信息重复。
# 在每组中仅保留 IV 最高的一个代表特征，直接剔除其余 16 个后缀特征
house_num_features['prefix'] = house_num_features['Row'].str.split('_').str[0]
house_num_features = house_num_features.sort_values(by=['prefix', 'iv'], ascending=[True, False])
house_num_keep_cols = house_num_features.groupby('prefix')['Row'].head(1).tolist()
# 	Row
# 23	APARTMENTS_AVG
# 24	BASEMENTAREA_AVG
# 26	ELEVATORS_AVG
# 27	ENTRANCES_AVG
# 28	FLOORSMAX_AVG
# 29	LIVINGAREA_AVG
# 30	NONLIVINGAREA_AVG
# 48	TOTALAREA_MODE
# 41	YEARS_BEGINEXPLUATATION_MEDI

# 第二步，剔除跨维度规模共线性
# 进一步对保留的特征做相关性分析
house_corr = application_train[house_num_keep_cols].corr()
mask = house_corr.columns.isin(['APARTMENTS_AVG', 'ELEVATORS_AVG', 'LIVINGAREA_AVG', 'TOTALAREA_MODE'])

house_corr.loc[mask, mask]
#                 APARTMENTS_AVG  ELEVATORS_AVG  LIVINGAREA_AVG  TOTALAREA_MODE
# APARTMENTS_AVG        1.000000       0.836958        0.913619        0.892627
# ELEVATORS_AVG         0.836958       1.000000        0.867776        0.844628
# LIVINGAREA_AVG        0.913619       0.867776        1.000000        0.925039
# TOTALAREA_MODE        0.892627       0.844628        0.925039        1.000000

del house_corr, mask

house_num_features.loc[house_num_features['Row'].isin(['APARTMENTS_AVG', 'ELEVATORS_AVG', 'LIVINGAREA_AVG', 'TOTALAREA_MODE']), 
                   ['Row', 'iv']]
#                Row        iv
# 23  APARTMENTS_AVG  0.032630
# 26   ELEVATORS_AVG  0.033217
# 29  LIVINGAREA_AVG  0.034331
# 48  TOTALAREA_MODE  0.037269

# 这组特征间相关性很强，保留最大iv值的TOTALAREA_MODE
house_num_keep_cols = [col for col in house_num_keep_cols if col not in ['APARTMENTS_AVG', 'ELEVATORS_AVG', 'LIVINGAREA_AVG']]

house_num_delete_cols = [col for col in house_num_cols if col not in house_num_keep_cols]

del house_num_features, house_num_keep_cols

# 去除28个房屋特征后，对剩余数值类型特征进行相关性分析
rest_model_num_features = model_features_summary.loc[
    (~model_features_summary['Row'].isin(house_cols)) & (model_features_summary['dtypes'] != 'object'),
    'Row'].tolist()

del house_cols, house_num_cols
rest_corr = application_train[rest_model_num_features].corr()
# 筛选出相关系数大于0.8的强线性相关变量
mask = np.triu(np.ones(rest_corr.shape), k=1).astype(bool)
result = rest_corr.where(mask).stack().reset_index()
result.columns = ['val1', 'val2', 'corr']

result[np.abs(result['corr']) > 0.8]
#                      val1                         val2      corr
# 1              AMT_CREDIT              AMT_GOODS_PRICE  0.986968
# 83          DAYS_EMPLOYED               FLAG_EMP_PHONE -0.999755
# 135  REGION_RATING_CLIENT  REGION_RATING_CLIENT_W_CITY  0.950842

del rest_corr, mask, result

# 比较各自的iv值
cols_eda.loc[cols_eda['Row'].isin(['AMT_CREDIT', 'AMT_GOODS_PRICE']), ['Row', 'iv']]
#                 Row        iv
# 8        AMT_CREDIT  0.059311
# 10  AMT_GOODS_PRICE  0.092032

# 保留AMT_GOODS_PRICE

cols_eda.loc[cols_eda['Row'].isin(['DAYS_EMPLOYED', 'FLAG_EMP_PHONE']), ['Row', 'iv']]
#                Row        iv
# 18   DAYS_EMPLOYED  0.113994
# 23  FLAG_EMP_PHONE  0.032907

# 保留DAYS_EMPLOYED

cols_eda.loc[cols_eda['Row'].isin(['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']), ['Row', 'iv']]
#                             Row        iv
# 30         REGION_RATING_CLIENT  0.048317
# 31  REGION_RATING_CLIENT_W_CITY  0.051201

# 保留REGION_RATING_CLIENT_W_CITY

# 得到最终相关性筛选后的删除特征名单
secend_delete_features = house_num_delete_cols + ['AMT_CREDIT', 'FLAG_EMP_PHONE', 'REGION_RATING_CLIENT']
del rest_model_num_features, house_num_delete_cols

# 最终入模变量
model_features = [col for col in model_features if col not in secend_delete_features]
model_features_summary = cols_eda[cols_eda['Row'].isin(model_features)]

model_features_summary.to_csv(DATA_PATH / 'interim' / 'model_features_summary.csv', index=False)
