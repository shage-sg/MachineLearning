# encoding=utf-8

# Author : 啵啵

# File : Deep_pipline.py 

# Software: PyCharm
import matplotlib.pyplot as plt
import numpy as np
from typing import TypeVar

from Tomato.function_models.Log_inf import Log_inf


class Deep_pipline(object):
    def __init__(self):
       Log_inf().log("Use Deep_pipline function!")

    def __deep_pipline(self, data):
        flat = []
        for i in data:
            # 直接返回numpy.ndarray格式通道顺序是RGB，通道值默认范围0-255
            # (224, 224, 3)
            img = plt.imread(i)
            # 像素值缩放处理
            img = img / 255.

            flat.append(img)

        flat = np.array(flat,dtype=np.float16)
        flat = flat.reshape(-1, flat[0].shape[0], flat[0].shape[1], flat[0].shape[2])
        return flat

    def process(self, data):
        return self.__deep_pipline(data)

