# encoding=utf-8

# Time : 2021/4/29 21:04 

# Author : 啵啵

# File : RCLF_CLF.py 

# Software: PyCharm

# 交叉验证导入的是未经训练的模型


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import logging

# 记录器
logger = logging.getLogger()

# 设置级别
logger.setLevel(logging.INFO)
# 没有给handler指定日志级别 默认使用logger的级别

# 写入文件
fileHandler = logging.FileHandler(filename="RandomForestClassifierInfo.log", mode='a')
# formatter格式
formatter = logging.Formatter(fmt="%(asctime)s-%(levelname)s-%(filename)s-[line:%(levelno)s]-%(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
# 给处理器设置格式
fileHandler.setFormatter(formatter)
# 记录器设置处理器
logger.addHandler(fileHandler)

HealthCareData = pd.read_csv('HealthCareData.csv')
# normalize参数可以返回占比   也可以自己算
Before_OverSampling_stroke_data = HealthCareData['stroke'].value_counts()

Before_OverSampling_stroke_data_sum = Before_OverSampling_stroke_data.values.sum()

print("Before OverSampling")
print(Before_OverSampling_stroke_data.apply(lambda divide: divide / Before_OverSampling_stroke_data_sum))

print("==" * 30)
"""
0    0.951272
1    0.048728
数据是不均衡的
"""

# 特征值选择 stroke为Classifier
X = HealthCareData.drop('stroke', axis=1)
Y = HealthCareData['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=2)

# 合成少数类过采样技术
sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

After_OverSampling_stroke_data = y_train_res.value_counts()
After_OverSampling_stroke_data_sum = After_OverSampling_stroke_data.values.sum()

print("After OverSampling")
print(After_OverSampling_stroke_data.apply(lambda divide: divide / After_OverSampling_stroke_data_sum))

print("==" * 30)
# 实例化模型

dtc = DecisionTreeClassifier(random_state=2)
rfc = RandomForestClassifier(random_state=2)

# 训练模型
dtc = dtc.fit(X_train_res, y_train_res)
rfc = rfc.fit(X_train_res, y_train_res)

# 计算模型得分

score_dtc = dtc.score(X_test, y_test)
score_rfc = rfc.score(X_test, y_test)

print(f"Decision Tree:{score_dtc}")
print(f"Random Forest:{score_rfc}")

"""
Decision Tree:0.9151989562948467
Random Forest:0.9510763209393346
我们可以选用Random Forest模型
"""

"""
参数影响排序，以便我们分配每个超参数的权重

n_estimators:提升至平稳，n_estimators↑，不影响单个模型的复杂度 ⭐⭐⭐⭐

max_depth:有增有减，默认最大深度，即最高复杂度，向复杂度降低的方向调参
max_depth↓，模型更简单，且向图像的左边移动 ⭐⭐⭐

min_samples_leaf:有增有减，默认最小限制1，即最高复杂度，向复杂度降低的方向调参
min_samples_leaf↑，模型更简单，且向图像的左边移动 ⭐⭐

min_samples_split:有增有减，默认最小限制2，即最高复杂度，向复杂度降低的方向调参
min_samples_split↑，模型更简单，且向图像的左边移动 ⭐⭐

max_features:有增有减，默认auto，是特征总数的开平方，位于中间复杂度，既可以向复杂度升高的方向，也可以向复杂度降低的方向调参
max_features↓，模型更简单，图像左移
max_features↑，模型更复杂，图像右移
max_features是唯一的，既能够让模型更简单，也能够让模型更复杂的参
数，所以在调整这个参数的时候，需要考虑我们调参的方向⭐

"""

"""
我们只使用它来缩小每个超参数的值范围，以便可以为GridSearchCV提供更好的参数网格
随机森林的准确度已经很高了，我们可以注重基评估器的数量
"""
"""
交叉验证是网格搜索法的思想的一部分
交叉验证：数据量较少时使用
数据集划分为n份，依次取每一份做测试集，其他的n-1份做训练集
用来观测模型的稳定性
确定一个比较好的n_estimators：森林中树木的数量,基评估器的数目
"""
superpa = []
for n_e in range(100):
    # n_jobs设定工作的core的数量等于-1时，表示cpu里的所有core进行工作
    rfc = RandomForestClassifier(n_estimators=n_e+1,n_jobs=-1)
    rfc_super = cross_val_score(rfc,X_train,y_train,cv=10).mean()
    superpa.append(rfc_super)
#打印出：最高精确度取值，因为索引下标从零开始的
# max(superpa))+1指的是森林数目的数量n_estimators
print("Accuracy:",max(superpa))
print("n_estimators:",superpa.index(max(superpa))+1)
print("=="*30)
plt.figure(figsize=[20,5])
plt.plot(range(1,101),superpa)
plt.savefig(fname='n_estimators.png',bbox_inches='tight',dpi=200)
plt.show()


param_dict = {
    'bootstrap': [True, False],
    'max_depth': np.arange(1, 200),
    'max_features': np.arange(0.1, 1, 0.1),
    'min_samples_leaf': np.arange(0.1, 0.5, 0.1),
    'min_samples_split': np.arange(0.1, 1, 0.1),
    'n_estimators': [71],
}

random_search = RandomizedSearchCV(estimator=rfc,
                                   param_distributions=param_dict,
                                   n_iter=100,
                                   cv=10,
                                   # 使用所有的CPU进行训练，默认为1，使用1个CPU
                                   n_jobs=-1,
                                   # 日志打印
                                   verbose=1)

random_search.fit(X_train, y_train)

# 搜索训练后的副产品
superpa = []
for params, score in zip(random_search.cv_results_['params'], random_search.cv_results_['mean_test_score']):
    superpa.append(score)
    logger.info('\t'.join([str(params), str(score)]))

plt.figure(figsize=[20, 5])
plt.plot(range(len(superpa)), superpa)
plt.savefig('random_search.png',dpi=200,bbox_inches='tight')
plt.show()



print("==" * 30)
print(f"random_search模型的最优参数：{random_search.best_params_}")
print(f"random_search最优模型分数：{random_search.best_score_}")
print(f"random_search最优模型对象：")
pprint(random_search.best_estimator_)


param_grid = [
    {'bootstrap': [False],
     'max_depth': [9,10,11],
     'max_features': [0.5,0.6,0.7,0.8],
     'min_samples_leaf': [0.2,0.3,0.4],
     'min_samples_split': [0.7,0.8,0.9],
     'n_estimators': [71],
     }
]

grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid,
                          cv = 10, n_jobs = -1, verbose = 1)

grid_search.fit(X_train,y_train)

# 搜索训练后的副产品
superpa = []
for params, score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score']):
    superpa.append(score)
    logger.info('\t'.join([str(params), str(score)]))

plt.figure(figsize=[20, 5])
plt.plot(range(len(superpa)), superpa)
plt.savefig('grid_search.png',dpi=80,bbox_inches='tight')
plt.show()


print("==" * 30)
print(f"grid_search模型的最优参数：{grid_search.best_params_}")
print(f"grid_search最优模型分数：{grid_search.best_score_}")
print(f"grid_search最优模型对象：")
pprint(grid_search.best_estimator_)

