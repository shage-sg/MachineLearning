# encoding=utf-8

# Time : 2021/3/30 21:03 

# Author : 啵啵

# File : demo_knn.py 

# Software: PyCharm

# 构建已经分好类的原始数据集
import pandas as pd
from DaPy.datasets import iris

rowdata = {"电影名称": ['无问西东', '后来的我们', '前任3', '红海行动', '唐人街探案', '战狼2'],
           "打斗镜头": [1, 5, 12, 108, 112, 115],
           "接吻镜头": [101, 89, 97, 5, 9, 8],
           "电影类型": ['爱情片', '爱情片', '爱情片', '动作片', '动作片', '动作片'],
           }

movie_data = pd.DataFrame(rowdata)
print(movie_data)

# 计算已知类别数据集中的点与当前点之间的距离(欧氏距离)
new_data = [24, 67]
# 广播 按列求和
dist = list((((movie_data.iloc[:6, 1:3] - new_data) ** 2).sum(1)) ** 0.5)
print(dist)

# 将距离升序排列，然后选取距离最小的k个点
dist_1 = pd.DataFrame(
    {'dist': dist,
     'labels': (movie_data.iloc[:6, 3])}
)
dr = dist_1.sort_values(by='dist')[:4]
print(dr)

# 确定前k个点所在类别出现的频率
re = dr.loc[:, 'labels'].value_counts()
print(re)

# 选择频率最高的类别作为当前点的预测类别
result = []
result.append(re.index[0])
print(result[0])


def classify(inx, dataSet, k):
    result = []
    dist = list((((dataSet.iloc[:6, 1:3] - new_data) ** 2).sum(1)) ** 0.5)
    dist_1 = pd.DataFrame(
        {'dist': dist,
         'labels': (movie_data.iloc[:6, 3])}
    )
    dr = dist_1.sort_values(by='dist')[:k]
    re = dr.loc[:, 'labels'].value_counts()
    result.append(re.index[0])
    return result
