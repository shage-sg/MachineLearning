# encoding=utf-8

# Time : 2021/5/26 18:10 

# Author : 啵啵

# File : Plot_images.py

# Software: PyCharm

import matplotlib.pyplot as plt
from function_models.Load_images import Load_images
from function_models.Log_inf import Log_inf


class Plot_images(object):

    def __init__(self, ):
        Log_inf().log("Use Plot_images Function!")

    def __plot_images(self):
        load_images = Load_images()
        images = load_images.data()
        print(load_images)

        plt.figure(figsize=(20, 3))
        for crop in images.keys():
            plt_crop = plt.imread(images.get(crop)[0][0])
            plt.subplot(1, 5, list(images.keys()).index(crop) + 1)
            plt.title(crop)
            plt.imshow(plt_crop)
        plt.title("$images\ show$")
        plt.savefig('../pictures/plot_images.png', bbox_inches="tight", dpi=500)
        plt.show()

    def __call__(self, *args, **kwargs):
        self.__plot_images()


if __name__ == '__main__':
    Plot_images()()
