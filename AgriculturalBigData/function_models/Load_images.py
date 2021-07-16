# encoding=utf-8

# Time : 2021/5/26 17:55 

# Author : 啵啵

# File : Load_images.py

# Software: PyCharm

import os
from pprint import pprint

from function_models.Log_inf import Log_inf


class Load_images(object):
    # 类变量
    index = 0

    def __init__(self):
        Log_inf().log("Use Load_Images Function!")
        self.path = r'../data/kag2'
        self.__images_dict = {}
        self.__count= 0

    def __load_images(self,index,*args,**kwargs):

        for root,dirs,files in os.walk(self.path):
            paths = []
            for file in files:
                paths.append([os.path.join(root,file),index])


            if paths != []:
                self.__images_dict[root.split("\\")[-1]] = paths
                index += 1
                self.__count += len(paths)
        print(f"Loading images SUCCESS! Count:{self.__count}")
        # pprint(self.__images_dict)

    def data(self):
        self.__load_images(Load_images.index)
        return self.__images_dict

    def __str__(self):
        return f"Loading images SUCCESS! Count:{self.__count}"

if __name__ == '__main__':
    Load_images().data()









