# encoding=utf-8

# Time : 2021/5/26 21:23 

# Author : 啵啵

# File : Split_train_test.py

# Software: PyCharm

from sklearn.model_selection import train_test_split
from function_models.Label_images import Label_images
from pprint import pprint

from function_models.Log_inf import Log_inf


class Split_train_test(object):

    def __init__(self):
        Log_inf().log("Use Split_train_test Function !")

    def __split_train_test(self):
        data = Label_images().data()
        train, test = train_test_split(data, test_size=0.2, random_state=66, shuffle=True)
        X_train = train['path'].values
        y_train = train.drop(['path'], axis=1).values
        X_test = test['path'].values
        y_test = test.drop(['path'], axis=1).values
        return X_train, y_train, X_test, y_test

    def data(self):
        return self.__split_train_test()

    def __str__(self):
        return self.__split_train_test()


if __name__ == '__main__':
    Split_train_test().data()
