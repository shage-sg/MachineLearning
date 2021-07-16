# encoding=utf-8

# Time : 2021/4/29 15:29 

# Author : 啵啵

# File : Clean_Data.py

# Software: PyCharm

import pandas as pd
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
# 读取数据
HealthCareData = pd.read_csv(r'./healthcare-dataset-stroke-data.csv')

"""
在pandas中object类型数据我们可以理解为String类型的数据
"""

# 提取标签值为object类型的特征值
SolutionData = list(HealthCareData.dtypes[HealthCareData.dtypes == 'object'].index)

"""
离散型的数据转换成0到n−1之间的数，这里n是一个列表的不同取值的个数，可以认为是某个特征的所有不同取值的个数。
"""

# 实例化模型
LE = OrdinalEncoder()

# 字典生成式
dict = {data: LE.fit_transform(HealthCareData[data]) for data in SolutionData}

for key in dict.keys():
    HealthCareData.loc[:, key] = dict.get(key)
from pprint import pprint
pprint(HealthCareData.isna().sum())
exit(0)
are_na = list(HealthCareData.isna().sum()[HealthCareData.isna().sum()!=0].index)
for is_na in are_na:
    HealthCareData[is_na] = HealthCareData[is_na].fillna(HealthCareData[is_na].mean())

HealthCareData.to_csv(r"./Project_Middle_Exam/HealthCareData.csv", index=None,float_format='%.1f')
