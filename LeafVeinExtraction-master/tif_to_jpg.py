# encoding=utf-8

import os
import cv2
import math

def tif_to_png():
    # file_path = 'E:\dataming\leaves\leaves'
    file_path = r'./leaves_test/'
    if r'/' != file_path[-1]:
        file_path += r'/'

    path_dir = os.listdir(file_path)
    for i in path_dir[:10]:
        img = cv2.imread(os.path.join(file_path, i))
        img_re = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)  # 按比例缩放
        i = "./split_before/" + os.path.splitext(i)[0] + ".JPG"
        cv2.imwrite(i, img_re)

if __name__ == '__main__':
    tif_to_png()

