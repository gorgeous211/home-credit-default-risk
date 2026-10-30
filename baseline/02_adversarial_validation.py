# 训练集与测试集对比
# 使用lightgbm进行对抗性验证

# %%
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# 文件路径
DATA_PATH = Path().cwd() / 'data'

# 主申请表
application_train = pd.read_csv(DATA_PATH / 'raw' / 'application_train.csv').drop(
    columns=['SK_ID_CURR', 'TARGET']
)
application_test = pd.read_csv(DATA_PATH / 'raw' / 'application_test.csv').drop(columns='SK_ID_CURR')

# %%
# 标记训练集和测试集
application_train['is_test'] = 0
application_test['is_test'] = 1

# 合并训练集和测试集
application = pd.concat([application_train, application_test], ignore_index=True)

# 类别特征转换为category类型，使lightgbm可以直接处理
obj_cols = application.select_dtypes(include='object').columns
for col in obj_cols:
    application[col] = application[col].astype('category')

# 切分X和y
X = application.drop(columns='is_test')
y = application['is_test']

# 设置分层五折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
oof_preds = np.zeros(len(X))
feature_importance = np.zeros(X.shape[1])

# learning_rate、n_estimators根据经验确定，min_child_sample根据数据集确定
# max_depth、num_leaves、subsample、colsample_bytree、reg_alpha、reg_lambda通过算法搜索参数

# max_depth: 一般在3，4，5中取一个
# num_leaves: LightGBM中与max_depth配合，要小于2^max_depth-1
# subsample: 一般给出大致的搜索范围如[0.8, 0.9, 1.0]这样几个比较整的数值
# colsample_bytree: 与subsample同理
# reg_alpha与reg_lambda，一般在1-1000的范围调参，比较好的模型上这俩参数值不应该过大，尤其是reg_alpha，过大的话需要查看特征

# 对抗验证不需要网络搜索调参，重点防止过拟合，避免捕捉虚假漂移
params = { 
    # 复杂度控制参数
    # 浅树更倾向于捕捉全局性特征分布差异，避免深树在叶节点上对个别样本强行切分
    'max_depth': 5,             # 默认-1，树的最大深度， 一般在3、4、5这三个数里挑一个
    'num_leaves': 31,           # 默认31，树的最大叶子数，取<=2^max_depth-1=2^5-1=31
    'min_child_samples': 50,     # 默认20，单个叶子节点上所需的最小样本数
                                # 调大该参数，防止叶节点样本数过少导致的局部过拟合
                                
    # 采样与扰动参数
    'subsample': 0.8,           # 默认1.0，行采样比例（样本子集比例）
    'subsample_freq': 1,        # 默认0，行采样频率。0 表示不启用 bagging；若设为k，则每k轮迭代采样一次
    'colsample_bytree': 0.8,    # 默认1.0，列采样比例，每棵树构建时采样的特征比例
                                # 调小该参数，防止少数强漂移特征在每棵树中都占据绝对主导地位
                                # 强制模型去探索其他次要特征的漂移情况
    
    # 正则化与迭代参数
    'reg_alpha': 0.5,           # 默认0.0，L1正则化系数
    'reg_lambda': 1.0,          # 默认0.0，L2正则化系数
    'learning_rate': 0.05,      # 默认0.1，调小该参数提升模型上限与泛化能力
    'n_estimators': 1000,       # 默认100，迭代轮数（基学习器的最大数量），
                                # 由于调小learning_rate并且加入了早停机制，调大该参数
    # 任务与目标参数
    'objective': 'binary',      # 默认None，binary代表二分类任务
    'class_weight': 'balanced', # 默认None，类别权重，用于处理样本不平衡
    'random_state': 123,        # 默认None，随机种子
    'importance_type': 'gain',  # 默认'split'，特征重要性计算方式，'gain'表示按信息增益计算
    'n_jobs': -1,               # 默认None，-1代表使用全部CPU核心
}

callbacks = [lgb.early_stopping(
    stopping_rounds=50,         # 不需要迭代太多轮次，过晚停止往往意味着在拟合噪声
    verbose=True)]

for train_idx, val_idx in skf.split(X, y):
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    
    clf = lgb.LGBMClassifier(**params)
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_names=['tran', 'valid'],
        eval_metric='auc',
        callbacks=callbacks
    )
    
    oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
    feature_importance += clf.feature_importances_ / skf.n_splits

overall_auc = round(roc_auc_score(y, oof_preds), 2)
# 0.89

importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

importance_df.head(10)
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

# 通过AUC的值可以得出，训练集和测试集二者存在可检测的特征分布差异
# AMT_CREDIT、AMT_ANNUITY 等排名靠前，可作为进一步检查分布差异的重点变量
# 该结果反映当前模型的区分能力；要判断具体漂移特征及其对违约预测的影响，还需比较这些变量的训练集与测试集分布