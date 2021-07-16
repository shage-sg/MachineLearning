# encoding=utf-8

# Time : 2021/5/26 21:53 

# Author : 啵啵

# File : VGG19_model.py

# Software: PyCharm

import warnings

warnings.filterwarnings('ignore')

import torch
import tensorflow
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from function_models.Preprocess_data import Preprocess_data
from function_models.Plot_accuracy_loss import Plot_accuracy_loss
from function_models.Plot_confusion_matrix import Plot_confusion_matrix
from function_models.Deep_model_pipline import Deep_model_pipline
from function_models.Log_inf import Log_inf


class VGG19_model(object):

    def __init__(self) -> None:
        Log_inf().log("Use VGG19_model Function!")
        self.hist = None
        self.vggmodel = None
        self.vggmodel_X_train = None
        self.vggmodel_y_train = None
        self.vggmodel_X_test = None
        self.vggmodel_y_test = None
        self.true = None
        self.predict = None

    def __model_init(self) -> None:
        Log_inf().log("model_init  Function")
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
        tensorflow.keras.backend.clear_session()
        tensorflow.config.experimental.list_physical_devices('GPU')

        config = tensorflow.compat.v1.ConfigProto(gpu_options=tensorflow.compat.v1.GPUOptions(allow_growth=True))

        tensorflow.compat.v1.ConfigProto()  # 这是tensorflow2.0+版本的写法，这个方法的作用就是设置运行tensorflow代码的时候的一些配置，例如如何分配显存，是否打印日志等;所以它的参数都是　配置名称＝True/False(默认为False) 这种形式
        tensorflow.compat.v1.GPUOptions(
            allow_growth=True)  # 限制GPU资源的使用，此处allow_growth=True是动态分配显存，需要多少，申请多少，不是一成不变、而是一直变化
        tensorflow.compat.v1.Session(config=config)  # 让这些配置生效

        vgg = tensorflow.keras.applications.VGG19(input_shape=(224, 224, 3),
                                                  # 不添加全连接层
                                                  include_top=False,
                                                  weights='imagenet',
                                                  pooling='avg')
        vgg.trainable = False
        self.vggmodel = Sequential([vgg,
                                    Dense(1000, activation='tanh'),
                                    Dense(1000, activation='tanh'),
                                    Dense(1000, activation='tanh'),
                                    Dense(5, activation='softmax')])
        self.vggmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # 模型摘要
        self.vggmodel.summary()
        # 绘制的图形每一层的输入和输出形状
        tensorflow.keras.utils.plot_model(self.vggmodel, "./vgg19_model.png", show_shapes=True)

    def __train_test(self) -> None:
        Log_inf().log("train_test  Function")
        self.vggmodel_X_train, self.vggmodel_y_train, \
        self.vggmodel_X_test, self.vggmodel_y_test, \
        self.paths = Preprocess_data().data()

    def __model_train(self):
        Log_inf().log("model_train  Function")
        self.hist = self.vggmodel.fit(self.vggmodel_X_train, self.vggmodel_y_train, epochs=50, validation_split=0.3,
                                      batch_size=16, verbose=True)

    def __plot_accuracy_loss(self):
        Log_inf().log("plot_accuracy_loss  Function")
        Plot_accuracy_loss(self.hist)()

    def __model_accuracy(self):
        Log_inf().log("model_accuracy  Function")
        accuracy = self.vggmodel.evaluate(self.vggmodel_X_test, self.vggmodel_y_test)
        with open("../accuracy/accuracy.data") as write:
            write.write(f"accuracy:{accuracy[1]}\n")

    def __model_predict(self):
        Log_inf().log("model_predict  Function")
        self.pre = self.vggmodel.predict(self.vggmodel_X_test)
        self.predict = np.argmax(self.pre, axis=1)
        self.true = np.argmax(self.vggmodel_y_test, axis=1)
        best_prob = [self.pre[num, :][i] for num, i in enumerate(self.predict)]

    def __plot_confusion_matrix(self):
        Log_inf().log("plot_confusion_matrix")
        Plot_confusion_matrix(true=self.true, predict=self.predict)()

    def __plot_prediction_samples_images(self):
        Log_inf().log("plot_prediction_samples_images")
        jute_path = '../data/kag2/jute/jute001a.jpeg'
        maize_path = '../data/kag2/maize/maize001a.jpeg'
        rice_path = '../data/kag2/rice/rice001a.jpeg'
        sugarcane_path = '../data/kag2/sugarcane/sugarcane0001a.jpeg'
        wheat_path = '../data/kag2/wheat/wheat0001a.jpeg'
        for path in [jute_path, maize_path, rice_path, sugarcane_path, wheat_path]:
            Deep_model_pipline(image_path=[path], model=self.vggmodel, ckey=[-1])()

    def __plot_prediction_images(self):
        Log_inf().log("plot_prediction_images  Function")
        ceil: int = int((len(self.paths) - len(self.paths) % 20) / 20)
        for num in range(ceil):
            Deep_model_pipline(key=num,
                               image_path=self.paths[num * 20:(num + 1) * 20],
                               model=self.vggmodel,
                               ckey=[-1],
                               figsize=(25, 25),
                               y_test=self.vggmodel_y_test
                               )()

    def __save_model(self):
        Log_inf().log("save_model Function")
        self.vggmodel.save_weights("../models/vggmodelweight.h5")
        print("Save model SUCCESS!")

    def __load_model(self):
        Log_inf().log("load_model  Function")
        self.vggmodel.load_weights(r"../models/vggmodelweight.h5")

    def __call__(self):
        self.__model_init()
        self.__train_test()
        self.__model_train()
        self.__plot_accuracy_loss()
        self.__model_accuracy()
        self.__model_predict()
        self.__plot_confusion_matrix()
        self.__plot_prediction_samples_images()
        self.__plot_prediction_images()
        self.__save_model()

    def VGG199(self):
        return self.__load_model()


if __name__ == '__main__':
    VGG19_model()()
