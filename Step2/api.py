import cv2
import numpy as np
from collections import Counter
from matplotlib.pyplot import imread  # , imresize, imsave
# from scipy.misc import imread, imresize, imsave
from skimage import io, measure
from time import *
import warnings
import os
from .func import *
import matplotlib.pyplot as plt

from typing import Any, List
import numpy as np

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def api(image1: np.ndarray, image2: np.ndarray) -> List[np.ndarray]:
    # l = max(image1.shape)
    image2_ori = image2

    #
    # upperbound = 1000
    # if l > upperbound:
    #     multiple = l / upperbound
    #     new_size = np.array(image1.shape) / multiple
    # else:
    #     multiple = 1
    #     new_size = np.array(image1.shape)
    #
    # # new_size = new_size / 5
    # new_size = new_size.astype(int)  # * 5
    # old_size = (np.array(new_size)*(multiple)).astype(int)
    #
    # w, h, _ = new_size
    # new_size = (w, h)
    #
    # w, h, _ = old_size
    # old_size = (w, h)
    #
    #
    # #image2_ori = cv2.resize(image2,(old_size))
    # image1 = cv2.resize(image1, (new_size))
    # image2 = cv2.resize(image2, (new_size))  # .astype(np.int16)
    # #image2_ori = image2


    image1 = cv2.medianBlur(image1, 7)  # 使用7个卷积核进行中值化
    image2 = cv2.medianBlur(image2, 7)

    cv2.normalize(image1, image1, alpha=0, beta=128, norm_type=cv2.NORM_MINMAX)
    cv2.normalize(image2, image2, alpha=0, beta=128, norm_type=cv2.NORM_MINMAX)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    image1 = cv2.medianBlur(image1, 5).astype(np.int16)
    image2 = cv2.medianBlur(image2, 5).astype(np.int16)

    diff_image = abs(image1 - image2)
    # 判断是否需要利用hpf进行滤波
    fig_size = float(image1.shape[0] * image1.shape[1])


    diff_image = (diff_image > 10) * 255

    change_map = diff_image.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    cleanChangeMap = cv2.erode(change_map, kernel)  # 腐蚀操作
    cleanChangeMap = cv2.dilate(cleanChangeMap, kernel)  # 膨胀操作

    area_threshold = 0.001 * fig_size  # 1/1000就可以达成异常
    area_threshold2 = 0.1 * fig_size  # 认为不出现10%面积的异常，判定为光照因素
    image_list = remove_small_points(cleanChangeMap, area_threshold, area_threshold2, image2_ori, 4)#, multiple)
    lens = len(image_list)


    for i in range(lens):
        plt.subplot(1, lens, i + 1)
        plt.imshow(image_list[i])

    plt.savefig('test.jpg')
    #plt.show()
    return image_list
