# encoding=utf-8

# Time : 2021/5/22 14:13 

# Author : 啵啵

# File : cal.py 

# Software: PyCharm

import numpy as np
import pandas as pd
def cosine_distance(a, b):
    if a.shape != b.shape:
        raise RuntimeError("array {} shape not match {}".format(a.shape, b.shape))
    if a.ndim==1:
        a_norm = np.linalg.norm(a)
        b_norm = np.linalg.norm(b)
    elif a.ndim==2:
        a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    else:
        raise RuntimeError("array dimensions {} not right".format(a.ndim))
    similiarity = np.dot(a, b.T)/(a_norm * b_norm)
    dist = 1. - similiarity
    return similiarity
d1 = np.array([2,2,1,0,2,1])
d2 = np.array([1,3,0,2,7,0])
d3 = np.array([0,2,3,3,0,2])
d4 = np.array([2,1,0,3,8,4])
d5 = np.array([6,5,0,1,4,0])
d6 = np.array([2,1,6,8,0,4])
d7 = np.array([5,3,1,0,2,0])

c1 = np.array([1.5,2,0,2.5,7.5,2])
c2 = np.array([1,1.5,4.5,5.5,0,3])
c3 = np.array([13/3,10/3,2/3,1/3,8/3,1/3])

print((d7.T * d7).sum())
print((c1.T * c1).sum())
print(cosine_distance(d7,c1))

# from sklearn.datasets import load_iris
#
# iris = load_iris()
# data = pd.DataFrame(iris.data[:5,:],index=[1,2,3,4,5],columns=[1,2,3,4])
# print(data.cov())
# # print(data.iloc[:,0].cov(data.iloc[:,0]))
# means = iris.data[:5,:].sum(axis=0)/5
# print(iris.data[:5,:][:,0] - means[0])
# print(((iris.data[:5,:][:,1] - means[1])*(iris.data[:5,:][:,2] - means[2])).sum()/4)
# print(((iris.data[:5,:][:,1] - means[1])*(iris.data[:5,:][:,2] - means[2])).sum())