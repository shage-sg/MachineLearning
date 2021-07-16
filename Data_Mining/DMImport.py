# encoding=utf-8

# Time : 2021/5/11 16:39 

# Author : 啵啵

# File : DMImport.py

# Software: PyCharm

from sklearn.model_selection import (
    # 训练集测试集划分train_test_split
    train_test_split,
    # 交叉验证
    cross_val_score,
    # 网格搜索
    GridSearchCV,
    # 随机搜索
    RandomizedSearchCV
)


""" 
线性回归的评估指标：
explained_variance_score：解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量的方差变化，值越小则说明效果越差。
mean_absolute_error：平均绝对误差（Mean Absolute Error，MAE），用于评估预测结果和真实数据集的接近程度的程度，其其值越小说明拟合效果越好。
mean_squared_error：均方差（Mean squared error，MSE），该指标计算的是拟合数据和原始数据对应样本点的误差的平方和的均值，其值越小说明拟合效果越好。
r2_score：判定系数，其含义是也是解释回归模型的方差得分，其值取值范围是[0,1]，越接近于1说明自变量越能解释因变量的方差变化，值越小则说明效果越差。
"""
# 评分
from sklearn.metrics import (
    # 准确率
    accuracy_score,
    # f1_score
    f1_score,
    # 精确率
    precision_score,
    # 召回率
    recall_score,
    # 混淆矩阵
    confusion_matrix,
    # 分类报告
    classification_report,
    # 方差
    explained_variance_score,
    # roc/auc曲线
    roc_auc_score
)

# 线性回归
from sklearn.linear_model import (
    # 逻辑回归
    LogisticRegression,
    # 线性回归
    LinearRegression,
    )

# 支持向量机
from sklearn.svm import (
    # 支持向量机分类
    SVC,
    # 支持向量机回归
    SVR)

# K近邻算法
from sklearn.neighbors import (
    # KNC 分类
    KNeighborsClassifier,
    # KNR 回归
    KNeighborsRegressor)

# 决策树
from sklearn.tree import (
    # 分类
    DecisionTreeClassifier,
    # 回归
    DecisionTreeRegressor)

# 随机森林
from sklearn.ensemble import (
    # 随机森林分类
    RandomForestClassifier,
    # 随机森林回归
    RandomForestRegressor,
    AdaBoostClassifier
)

# 朴素贝叶斯
from sklearn.naive_bayes import GaussianNB

# 数据集
from sklearn.datasets import load_iris

# 合成少数类过采样技术
from imblearn.over_sampling import SMOTE

# 离散值连续化
from sklearn.preprocessing import (
    # 标签专用，将分类标签转换为数值标签
    LabelEncoder,
    # 特征专用，将分类特征转换为数值特征
    # 有序变量
    OrdinalEncoder,
    # 哑变量，独热编码 有你没我
    # 名义变量
    OneHotEncoder,
    StandardScaler,
    MinMaxScaler)

# 缺失值填充
# numpy fillna dropna
from sklearn.impute import SimpleImputer

from sklearn.feature_selection import (
    # 特征选择
    SelectKBest,
    # 卡方 k,p<=0.05 or 0.01 相关
    # 卡方检验，专用于分类算法，捕捉相关性，追求P值小于显著性水平的特征
    chi2,
    # 只要求数据服从正态分布 
    f_classif,
    # 互信息法
    # 互信息法不返回p值或F值类似的统计量，它返回“每个特征与目标之间的互信息量的估计”，这个估计量在[0,1]之间取值，为0则表示两个变量独立，为1则表示两个变量完全相关。
    mutual_info_classif,
)

# 朴素贝叶斯
from sklearn.naive_bayes import GaussianNB
import torch
import tensorflow as tf