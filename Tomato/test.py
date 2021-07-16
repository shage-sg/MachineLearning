# encoding=utf-8

# Author : 啵啵

# File : jianmo.py

# Software: PyCharm

import torch

import tensorflow as tf

# print(tf.__version__)
#
# print(torch.__path__)
print(tf.__version__) #
# CUDA是否可用
print(tf.test.is_built_with_cuda())
# GPU是否可用
print(tf.test.is_gpu_available())

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))

tf.compat.v1.ConfigProto() #这是tensorflow2.0+版本的写法，这个方法的作用就是设置运行tensorflow代码的时候的一些配置，例如如何分配显存，是否打印日志等;所以它的参数都是　配置名称＝True/False(默认为False) 这种形式
gpu_options=tf.compat.v1.GPUOptions(allow_growth=True) #限制GPU资源的使用，此处allow_growth=True是动态分配显存，需要多少，申请多少，不是一成不变、而是一直变化
sess = tf.compat.v1.Session(config=config) #让这些配置生效

tf.config.experimental.list_physical_devices('GPU')
# print(tf.test.is_gpu_available())


# print(torch.backends.cudnn.version())