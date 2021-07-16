# encoding=utf-8

# Time : 2021/5/26 19:26 

# Author : 啵啵

# File : Label_images.py 

# Software: PyCharm

from function_models.Load_images import Load_images
from pprint import pprint
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from function_models.Log_inf import Log_inf


class Label_images(object):

    def __init__(self):
        Log_inf().log("Use Label_images Function!")
        self.__new_X = []

    def label_images(self):
        pre_X = []
        load_images = Load_images()
        images = load_images.data()
        # print(load_images)
        # pprint(images)
        for m in range(0, len(list(images.values()))):
            for n in list(images.values())[m]:
                pre_X.append(n)
        return pre_X

    def process_images(self):
        X = pd.DataFrame(self.label_images(), columns=['path', 'labels'])
        oneHot = OneHotEncoder(categories="auto", handle_unknown="ignore", sparse=False)
        ohLabel = pd.DataFrame(oneHot.fit_transform(X[["labels"]]),
                               dtype='float16',
                               # 一共五种作物  名义变量 ==> 哑变量
                               columns=['label0', 'label1', 'label2', 'label3', 'label4'])
        # label_X = X.copy()
        X = pd.concat([X, ohLabel], axis=1)
        self.__new_X = X.drop(['labels'], axis=1)


    def data(self):
        self.process_images()
        # pprint(self.__new_X)
        return self.__new_X



if __name__ == '__main__':
    Label_images().data()
