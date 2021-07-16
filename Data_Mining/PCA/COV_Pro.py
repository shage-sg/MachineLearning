# encoding=utf-8

# Time : 2021/5/22 16:36 

# Author : 啵啵

# File : COV_Pro.py 

# Software: PyCharm

from sklearn.datasets import load_iris,fetch_lfw_people
import numpy as np
import pandas as pd

def read_iris():
    faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)
    data = faces.data
    print(data.shape)


def COV_func(*, data, **kwargs):
    shapes = kwargs['shapes']
    means = data.sum(axis=0) / shapes[0]
    data_cen = data - means
    matrix_index = [(x, y) for x in range(shapes[1]) for y in range(shapes[1])]
    cov_matrix = np.array(list(
        # 矩阵的点乘则要求矩阵必须维数相等，即MxN维矩阵乘以MxN维矩阵,各元素逐一相乘
        round((np.dot(data_cen[:, index[0]], data_cen[:, index[1]])).sum() / shapes[1], 4) for index in
        matrix_index)).reshape(shapes[1], shapes[1])
    print(cov_matrix)


read_iris()
