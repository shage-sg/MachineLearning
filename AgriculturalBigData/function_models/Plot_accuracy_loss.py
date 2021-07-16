# encoding=utf-8

# Time : 2021/5/27 9:29 

# Author : 啵啵

# File : Plot_accuracy_loss.py

# Software: PyCharm

import matplotlib.pyplot as plt

from function_models.Log_inf import Log_inf


class Plot_accuracy_loss(object):

    def __init__(self, hist):
        Log_inf().log("Use Plot_accuracy_loss Function!")
        self.hist = hist

    def __plot_accuracy_loss(self):
        plt.figure(figsize=(10, 7))
        plt.subplot(1, 2, 1)
        plt.plot(self.hist.history['accuracy'], label='accuracy')
        plt.plot(self.hist.history['loss'], label='loss')
        plt.legend()
        plt.title("training set")
        plt.grid()
        plt.subplot(1, 2, 2)
        plt.plot(self.hist.history['val_accuracy'], label='val_accuracy')
        plt.plot(self.hist.history['val_loss'], label='val_loss')
        plt.legend()
        plt.title("validation set")
        plt.grid()
        plt.ylim((0, 4))
        plt.savefig('../pictures/plot_acc_loss.png', bbox_inches="tight", dpi=500)
        # plt.show()

    def __call__(self, *args, **kwargs):
        self.__plot_accuracy_loss()
