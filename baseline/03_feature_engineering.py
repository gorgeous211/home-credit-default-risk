# %%
# -- 单箱样本约束 --
# 单箱样本占比原则上 >= 5%（全量样本 307511 时，至少约 15376 个样本）。
# 每个分类或分箱中的坏样本数建议不少于 20~30 个；样本过少时应合并分箱，避免 WOE 估计不稳定。
#
# -- WOE 分箱质量 --
# 1. 组间差异：相邻分类/分箱的 |ΔWOE| 建议 >= 0.10，严格标准可取 0.15；差异过小时考虑合并。
# 2. 单箱幅度：|WOE| 通常以 0.10~1.50 为较合理范围。
#    - |WOE| 长期接近 0（如 -0.05~+0.05）：该箱与总体违约率接近，区分信息有限。
#    - |WOE| 过大（如 > 2.0，尤其 > 3.0）：检查箱内样本/坏样本是否过少，并考虑合并分箱。
# 3. 趋势与稳定性：有序变量的 WOE 应单调递增或递减；无序变量应检查各箱排序在训练集、测试集及 OOT 样本间是否一致。
#    不应出现训练集与测试集（或跨时间样本）间 WOE 方向反转。
#
# -- IV 预测能力参考 --
# IV < 0.02：无预测能力，通常剔除；0.02~0.10：弱；0.10~0.30：中等；
# 0.30~0.50：强；IV > 0.50：预测能力极高，需排查数据穿越（Data Leakage）或过拟合。
# 0.10~0.30 通常是较理想、稳健的核心特征区间；IV 应结合业务含义与跨期稳定性判断。

# 对于低 IV 特征，优先做业务交叉衍生，完成上述交叉衍生后，原特征本身的任务就完成了
# 进入评分卡的特征初筛（Feature Selection）环节时，直接从候选特征池中剔除，不对其进行 WOE 转换，也不送入后续的多变量分析

# 逻辑回归评分卡通常只保留 10~20 个核心特征。
# 如果塞入大量弱特征，不仅无法提升模型的 KS 和 AUC，反而会引入噪声、增加共线性风险，
# 导致逻辑回归权重系数（Coefficient）失真甚至符号反转。

# -- PSI 稳定性评估标准 --
# PSI < 0.10：非常稳定，特征分布无明显变化，可直接保留。
# 0.10 <= PSI <= 0.25：轻微偏移，有条件保留；结合业务原因排查，必要时合并分箱或加强监控。
# PSI > 0.25：显著偏移（不稳定），原则上删除；若为核心强特征，先进行特征改造，再重新评估。

# %%
# 导入库
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 文件路径
DATA_PATH = Path().cwd() / 'data'

# 主申请表
application_train = pd.read_csv(DATA_PATH / 'raw' / 'application_train.csv')

# eda部分结果
cols_eda = pd.read_csv(DATA_PATH / 'interim' / 'cols_eda.csv')

# %%
# == 精简变量 ==
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

# -- 剔除零方差与极稀有特征 --
# FLAG_MOBIL
application_train = application_train.drop(columns='FLAG_MOBIL')

# FLAG_DOCUMENT_2, 4, 10, 12, 7, 17, 19, 20, 21 中1出现及低频
application_train = application_train.drop(columns=[
    f"FLAG_DOCUMENT_{i}"
     for i in [2, 4, 10, 12, 7, 17, 19, 20, 21]
])

# -- 剔除高漂移特征且低区分度特征 --
# FLAG_EMAIL 分箱后违约率无显著差异，并且漂移严重
application_train = application_train.drop(columns='FLAG_EMAIL')

# AMT_REQ_CREDIT_BUREAU_HOUR / DAY / WEEK / MON / QRT
# 短期查询非零样本极少或违约率倒挂，且 MON/QRT 在测试集发生严重跨期漂移
# 只保留 YEAR
application_train = application_train.drop(columns=['AMT_REQ_CREDIT_BUREAU_HOUR',
                                                    'AMT_REQ_CREDIT_BUREAU_DAY',
                                                    'AMT_REQ_CREDIT_BUREAU_WEEK',
                                                    'AMT_REQ_CREDIT_BUREAU_MON',
                                                    'AMT_REQ_CREDIT_BUREAU_QRT'])

# -- 精简高度共线的房屋特征群 --
# 47 个房屋相关变量中，_AVG、_MEDI、_MODE 相关性高达 0.90~1.00
# 仅保留_MODE 列
house_cols = cols_eda.loc[
    cols_eda['Description'].str.startswith('Normalized information'),
    'Row'
].tolist()

application_train = application_train.drop(columns=[
    col for col in house_cols
    if not col.endswith('MODE')
])

# AMT_CREDIT 和 AMT_GOODS_PRICE 相关系数0.99，剔除AMT_GOODS_PRICE
application_train = application_train.drop(columns='AMT_GOODS_PRICE')

# %%
# == 构造衍生特征 ==
# -- 还款能力与负债压力 --
# 信贷收入比：总借款是年收入的几倍
application_train['CREDIT_TO_INCOME_RATIO'] = \
    application_train['AMT_CREDIT'] / application_train['AMT_INCOME_TOTAL']

# 负债收入比 DTI：年金占收入比例，风控最核心指标
application_train['ANNUITY_TO_INCOME_RATIO'] = \
    application_train['AMT_ANNUITY'] / application_train['AMT_INCOME_TOTAL']

# 近似贷款还款率，其倒数 AMT_CREDIT / AMT_ANNUITY 即为近似贷款期限
application_train['ANNUITY_TO_CREDIT_RATIO'] = \
    application_train['AMT_ANNUITY'] / application_train['AMT_CREDIT']
    
# 商品价格占贷款额比例：反映是否包含额外保险费或隐性高息，用此比率替代原本共线性高达 0.99 的 AMT_GOODS_PRICE
application_train['GOODS_TO_CREDIT_RATIO'] = \
    application_train['AMT_GOODS_PRICE'] / application_train['AMT_CREDIT']

# 商品价格与贷款额的差额
application_train['DOWN_PAYMENT_PROXY'] = \
    application_train['AMT_GOODS_PRICE'] - application_train['AMT_CREDIT']

# 家庭人均收入
application_train['INCOME_PER_PERSON'] = \
    application_train['AMT_INCOME_TOTAL'] / application_train['CNT_FAM_MEMBERS']
    
# 按子女数调整的收入水平
application_train['INCOME_PER_CHILD'] = \
    application_train['AMT_INCOME_TOTAL'] / (1 + application_train['CMT_CHILDREN'])

# -- 稳定性与年龄交叉比率 --
# 工龄占年龄比：扣除无业异常值后，反映客户步入成年后的工作稳定度
application_train['EMPLOYED_TO_BIRTH_RATIO'] = \
    application_train['DAYS_BIRTH'] / application_train['DAYS_EMPLOYED']

# 车龄占年龄比
application_train['CAR_TO_BIRTH_RATIO'] = \
    application_train['OWN_CAR_AGE'] / (-application_train['DAYS_BIRTH'] / 365)

# 证件更换时间占年龄比
application_train['ID_TO_BIRTH_RATIO'] = \
    application_train['DAYS_ID_PUBLISH'] / application_train['DAYS_BIRTH']

# -- 外部征信评分 --
# EXT_SOURCE_1/2/3 是全表区分度最高的特征，但各自有不同程度的缺失
# 在评分卡中，除了保留这三个原始变量各自去分箱外，可以构造聚合指标
application_train['EXT_SOURCE_MEAN'] = \
    application_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)

# -- 风险信号聚合与同源缺失标志位 --
# 有缺失的变量中，除EXT_SOURCE_2、DAYS_LAST_PHONE_CHANGE、AMT_GOODS_PRICE、CNT_FAM_MEMBERS、
# AMT_ANNUITY外，其余缺失组与正常组的违约率差异均显著，构造是否缺失标志

# 在LightGBM中这么做合理，但是评分卡标准做法是开启缺失值单独成箱
# 故针对成组的同源缺失变量，构造“组级别”的缺失聚合特征

# 房屋特征群，构造房屋信息缺失字段比例
application_train['HOUSE_INFO_MISSING_CNT'] = \
    application_train[[
        col for col in house_cols
        if not col.endswith('MODE')
    ]].isna().mean(axis=1)

# 将主表中REG_..._NOT_LIVE_... 和 REG_..._NOT_WORK_...（户籍地、居住地、工作地不一致标志）相加求均值
# 反映申请人的居住/工作流动异常度
application_train['ADDRESS_MISMATCH_MEAN'] = \
    application_train[['REG_CITY_NOT_LIVE_CITY',
                       'REG_CITY_NOT_WORK_CITY',
                       'REG_REGION_NOT_LIVE_REGION',
                       'REG_REGION_NOT_WORK_REGION']].mean(axis=1)

# 30天社交圈违约占比
application_train['DEF_OBS_RATIO_30'] = \
    application_train['DEF_30_CNT_SOCIAL_CIRCLE'] / (application_train['OBS_30_CNT_SOCIAL_CIRCLE'] + 0.01)

# 60天社交圈违约占比
application_train['DEF_OBS_RATIO_60'] = \
    application_train['DEF_60_CNT_SOCIAL_CIRCLE'] / (application_train['OBS_60_CNT_SOCIAL_CIRCLE'] + 0.01)

# -- 高基数与长尾类别特征归并 --
# 长尾稀有类别合并
# NAME_INCOME_TYPE中，Unemployed、Student、Businessman和Maternity leave种类均不足30人，合并为Other


# %%
# 将主申请表划分为训练集和验证集
train_df, valid_df = train_test_split()
