# encoding=utf-8

# Time : 2021/5/18 15:22 

# Author : 啵啵

# File : Clustering_Pro.py 

# Software: PyCharm


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
# TypeVar参数化来约束一个类型集合
from typing import TypeVar
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import MinMaxScaler


class Clustering(object):
    """
    arg:
        X: 用于模型训练的数据
        predict_Y1: KMeans模型训练后的数据标签
        predict_Y1: agglomerativeClustering模型训练后的数据标签

    function_models:
        read_data: 读取数据
        process_data: 预处理数据
        kmeans_model: KMeans模型实例化并进行训练
        feature_select: 特征选择
        percentage_score: KMeans模型打分
        min_max_scale: 数据归一化(无量纲化)
        agglomerativeClustering_model: agglomerativeClustering_model模型实例化并进行训练
        draw_dendrogram: 画树图

    returns:
        self: 实例化类本身
        predict_Y: KMeans模型预测的标签
        scaler_X: 归一化后的训练数据

    note:
        MinMaxScaler: (x - min(x)) / (max(x) - min(x))
        survived (1)
        not survived (0)
        OrdinalEncoder: 无法处理含有缺失值的特征列，需要填充缺失值
     """

    def __init__(self):
        self.path = '../data/Titanic.csv'

    def __read_data(self) -> None:
        self.titanic_before = pd.read_csv(self.path, sep=",")
        return self

    def __process_data(self) -> None:
        # Drop the columns that have no significant impact on the training of the K-Means models.
        self.titanic_after = self.titanic_before.drop(["Name", "Cabin", "Embarked", "Ticket"], axis=1)
        return self

    def __feature_select(self) -> None:
        # drop the Survival coltitanic_afterumn from the data.
        # 名义变量，需要将其变成哑变量
        # self.original_X = OneHotEncoder(categories='auto').fit_transform(self.titanic_after.drop(["Survived","PassengerId"], axis=1)).toarray()
        self.titanic_after[self.titanic_after.isna().sum()[self.titanic_after.isna().sum()!=0].index.values[0]] \
        = SimpleImputer(missing_values=np.nan, strategy="median").fit_transform(
                self.titanic_after[self.titanic_after.isna().sum()[self.titanic_after.isna().sum()!=0].index.values[0]].values.reshape(-1,1)
        )

        self.original_X = OrdinalEncoder().fit_transform(
                self.titanic_after.drop(["Survived", "PassengerId"], axis=1).values)

        self.original_Y = np.array(self.titanic_after["Survived"])
        return self

    def __kmeans_model(self, X: TypeVar(int)) -> TypeVar(int):
        # Run K-means clustering on the data of titanic.
        KMeansModel = KMeans(
            n_clusters=2,
        ).fit(
            X=X
        )
        predict_Y1 = KMeansModel.labels_
        return predict_Y1

    def __percentage_score(self, obj1: TypeVar(int), obj2: TypeVar(int)) -> None:
        # the percentage of passenger records that were clustered correctly
        # boolean index
        score = obj1[obj2 == obj1].shape[0] / obj1.shape[0]
        if score > 0.5:
            pass
        else:
            score = 1 - score
        percentage_score = f"{float('{:.5f}'.format(score)) * 100}%"
        print("the percentage of passenger records that were clustered correctly:" + percentage_score)
        return self

    def __min_max_scale(self) -> TypeVar(int):

        scaler_X = MinMaxScaler(
            feature_range=[0, 1]
        ).fit_transform(
            self.original_X
        )
        return scaler_X

    def __agglomerativeClustering_model(self, X: TypeVar(float)) -> TypeVar(int):

        agglomerativeClustering = AgglomerativeClustering(
            n_clusters=2
        ).fit(
            X
        )
        predict_Y2 = agglomerativeClustering.labels_
        return predict_Y2

    def __draw_dendrogram(self) -> None:

        plt.figure(figsize=(12, 8))

        sch.dendrogram(sch.linkage(self.original_X, method='ward'))
        # remove xtick
        plt.xticks([])
        plt.yticks([])
        # draw the split line
        plt.plot(list(range(10000)), [2200] * 10000, ls='--', lw=3, c='red')
        # set title
        plt.title("Dendrogram",size=15)
        # set tick style
        plt.tick_params(labelsize=15)
        # savefig
        plt.savefig("dendrogram.jpg", bbox_inches="tight", dpi=300)
        # show picture
        plt.show()

    def __call__(self):

        self.__read_data()
        self.__process_data()
        self.__feature_select()

        print("KMeans Before the MinMaxScaler:")
        self.__percentage_score(self.original_Y, self.__kmeans_model(self.original_X))

        print("KMeans After the MinMaxScaler:")
        self.__percentage_score(self.original_Y, self.__kmeans_model(self.__min_max_scale()))

        print("AgglomerativeClustering:")
        self.__percentage_score(self.original_Y, self.__agglomerativeClustering_model(self.__min_max_scale()))

        self.__draw_dendrogram()


if __name__ == '__main__':
    # 查看项目文档
    print(Clustering().__doc__)
    Clustering()()
