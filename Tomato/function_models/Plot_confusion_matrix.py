# encoding=utf-8

# Author : 啵啵

# File : Plot_confusion_matrix.py 

# Software: PyCharm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from Tomato.function_models.Log_inf import Log_inf


class Plot_confusion_matrix(object):
    def __init__(self, true, predict):
        Log_inf().log("Use Plot_confusion_matrix Function!")
        self.true = true
        self.predict = predict

    def __plot_confusion_matrix(self):
        plt.figure(figsize=(9, 8))

        class_label = ["Bacterial_spot",
                       "Early_blight",
                       "healthy",
                       "Late_blight",
                       "Leaf_Mold"]
        fig = sns.heatmap(confusion_matrix(self.true, self.predict), cmap="coolwarm", annot=True, vmin=0, cbar=False,
                          center=True, xticklabels=class_label, yticklabels=class_label)
        fig.set_xlabel("Prediction", fontsize=30)
        fig.xaxis.set_label_position('top')
        fig.set_ylabel("True", fontsize=30)
        fig.xaxis.tick_top()
        plt.savefig('../pictures/plot_confusion_matrix.png', bbox_inches="tight", dpi=500)
        # plt.show()

    def __call__(self, *args, **kwargs):
        self.__plot_confusion_matrix()
