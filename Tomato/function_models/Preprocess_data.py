# encoding=utf-8

# Author : 啵啵

# File : Preprocess_data.py 

# Software: PyCharm
from Tomato.function_models.Log_inf import Log_inf
from Tomato.function_models.Split_train_test import Split_train_test
from Tomato.function_models.Deep_pipline import Deep_pipline
from pprint import pprint
import numpy as np

class Preprocess_data(object):

    def __init__(self):
        Log_inf().log("Use Preprocess_data Function!")
        self.__X_train = None
        self.__X_test = None


    def __preproess_data(self):

        X_train, y_train, X_test, y_test = Split_train_test().data()
        deep_pipline = Deep_pipline()
        self.__X_train = deep_pipline.process(data=X_train)
        self.__X_test = deep_pipline.process(data=X_test)
        return self.__X_train, y_train, self.__X_test, y_test,X_test

    def data(self):
        return self.__preproess_data()


if __name__ == '__main__':
    Preprocess_data().data()
