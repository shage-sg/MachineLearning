# encoding=utf-8

# Time : 2021/5/27 10:33 

# Author : 啵啵

# File : Deep_model_pipline.py 

# Software: PyCharm

from function_models.Deep_pipline import Deep_pipline
import numpy as np
import matplotlib.pyplot as plt

from function_models.Log_inf import Log_inf


class Deep_model_pipline(object):

    def __init__(self, image_path, model, ckey, y_test=None, key=None, figsize=None):
        Log_inf().log("Use Deep_model_pipline Function!")
        self.image_path = image_path
        self.model = model
        self.ckey = ckey
        self.figsize = figsize
        self.name = None
        self.key = key
        self.y_test = y_test

    def __deep_model_pipline(self):

        pre_dict = {
            0: "jute",
            1: "maize",
            2: "rice",
            3: "sugarcane",
            4: "wheat"
        }

        plt.figure(figsize=self.figsize) if self.figsize is not None else plt.figure()

        if len(self.image_path) == 1:
            bool = False
            self.name = self.image_path[0].split("/")[-2]
        else:
            bool = True
            self.name = f"test_group{self.key + 1}"

        for num, im_path in enumerate(self.image_path):
            predict_X = Deep_pipline().process([im_path])
            pre = self.model.predict(predict_X)
            predict = np.argmax(pre[0])
            if bool:
                self.ckey = self.y_test[20 * self.key + num]
                plt.subplot(4, 5, num + 1)
            if (self.ckey[0] != -1):
                plt.title(
                    "prediction:{0} \n Accuracy:{1:.2f}% \n True:{2}"
                        .format(
                        pre_dict.get(predict),
                        pre[0,predict] * 100,
                        pre_dict.get(np.argmax(self.ckey))
                    ))
            else:
                plt.title(
                    "prediction:{0} Accuracy:{1:.2f}%"
                        .format(
                        pre_dict.get(predict),
                        pre[0, predict] * 100,
                    ))

            plt.imshow(plt.imread(im_path))

        plt.savefig(f'../pictures/prediction_images/{self.name}.png', bbox_inches="tight", dpi=400)
        # plt.show()

    def __call__(self, *args, **kwargs):
        self.__deep_model_pipline()
