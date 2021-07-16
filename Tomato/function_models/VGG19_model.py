# encoding=utf-8

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

from Tomato.function_models.Preprocess_data import Preprocess_data
from Tomato.function_models.Plot_accuracy_loss import Plot_accuracy_loss
from Tomato.function_models.Plot_confusion_matrix import Plot_confusion_matrix
from Tomato.function_models.Deep_model_pipline import Deep_model_pipline
from Tomato.function_models.Log_inf import Log_inf


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
        # 加载模型之前清理模型所占用的内存
        tensorflow.keras.backend.clear_session()
        tensorflow.config.experimental.list_physical_devices('GPU')

        config = tensorflow.compat.v1.ConfigProto(gpu_options=tensorflow.compat.v1.GPUOptions(allow_growth=True))

        tensorflow.compat.v1.ConfigProto()  # 这是tensorflow2.0+版本的写法，这个方法的作用就是设置运行tensorflow代码的时候的一些配置，例如如何分配显存，是否打印日志等;所以它的参数都是　配置名称＝True/False(默认为False) 这种形式
        tensorflow.compat.v1.GPUOptions(
            allow_growth=True)  # 限制GPU资源的使用，此处allow_growth=True是动态分配显存，需要多少，申请多少，不是一成不变、而是一直变化
        tensorflow.compat.v1.Session(config=config)  # 让这些配置生效

        vgg = tensorflow.keras.applications.VGG19(input_shape=(256, 256, 3),
                                                  # 不添加全连接层
                                                  include_top=False,
                                                  weights='imagenet',
                                                  pooling='avg')
        vgg.trainable = False
        self.vggmodel = Sequential([vgg,
                                    Dense(1000, activation='tanh'),
                                    Dense(1000, activation='tanh'),
                                    Dense(5, activation='softmax')])
        self.vggmodel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.vggmodel.summary()

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

        with open("../accuracy/Accuracy.data", 'w') as writer:
            writer.write(f"Accuracy:{accuracy[1]}\n")

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
        Bacterial_spot_path = '../data/Tomato___Bacterial_spot/00a7c269-3476-4d25-b744-44d6353cd921___GCREC_Bact.Sp 5807.JPG'
        Tomato_Early_blight_path = '../data/Tomato___Early_blight/00c5c908-fc25-4710-a109-db143da23112___RS_Erly.B 7778.JPG'
        Tomato_healthy_path = '../data/Tomato___healthy/000bf685-b305-408b-91f4-37030f8e62db___GH_HL Leaf 308.1.JPG'
        Tomato_Late_blight_path = '../data/Tomato___Late_blight/00ce4c63-9913-4b16-898c-29f99acf0dc3___RS_Late.B 4982.JPG'
        Tomato_Leaf_Mold_path = '../data/Tomato___Leaf_Mold/0a9b3ff4-5343-4814-ac2c-fdb3613d4e4d___Crnl_L.Mold 6559.JPG'
        for path in [Bacterial_spot_path, Tomato_Early_blight_path, Tomato_healthy_path, Tomato_Late_blight_path,
                     Tomato_Leaf_Mold_path]:
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
        self.__model_init()
        self.vggmodel.load_weights(r"../models/vggmodelweight.h5")
        return self.vggmodel

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

    def VGG19(self):
        return self.__load_model()


if __name__ == '__main__':
    VGG19_model()()
