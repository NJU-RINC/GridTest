from __future__ import annotations
import cv2
import numpy as np
from collections import Counter
from time import *
import warnings
import os
from skimage import measure

class Box:
    def __init__(self, xmin: int, xmax: int, ymin: int, ymax: int):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

class Rect:
    def __init__(self, left, top, right, bottom) -> None:
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def __mul__(self, other):
        return self.left < other.right and other.left < self.right and self.top < other.bottom and other.top < self.bottom

    def union(self, left, top, right, bottom):
        if ((left < right) and (top < bottom)):
            if ((self.left < self.right) and (self.top < self.bottom)):
                if self.left > left: self.left = left
                if self.top > top: self.top = top
                if self.right < right: self.right = right
                if self.bottom < bottom: self.bottom = bottom
            else:
                self.left = left
                self.top = top
                self.right = right
                self.bottom = bottom
    
    def __iadd__(self, other: Rect):
        self.union(other.left, other.top, other.right, other.bottom)
        return self

    def __repr__(self) -> str:
        return f'|left: {self.left}, top: {self.top}, right: {self.right}, bottom: {self.bottom}|'

def merge(image_list):
    rects = np.array(image_list)
    check = [False] * len(rects)
    check = np.array(check).astype('bool')

    for i in range(len(rects)-1):
        for j in range(i+1, len(rects)):
            if check[i]:
                break # try next i
            if rects[i]*rects[j]:
                rects[i] += rects[j]
                check[j] = True

    return rects[~check]


def get_box(img, threshold_point,threshold_point2,image2,ab_num,expand_p):
    ################删去大小符合阈值的连通区域
    img_label, num = measure.label(img, return_num=True,connectivity=2)#输出二值图像中所有的连通域
    props = measure.regionprops(img_label)#输出连通域的属性，包括面积等
    numm=0

    res_area = []
    for i in range(1, len(props)):
        if props[i].area > threshold_point and props[i].area <threshold_point2:
            numm+=1
            res_area.append(props[i])


    res_area.sort(key=lambda t: t.area,reverse=True)
    image_list=[]

    for i in range(min(numm,ab_num)):
        image_list.append(Rect(res_area[i].bbox[1],res_area[i].bbox[0],res_area[i].bbox[3],res_area[i].bbox[2]))

    return merge(image_list)

    # while(1):
    #     image_list1=[]
    #     flag = 0
    #     temp = []
    #     for i in range(len(image_list)):
    #         flag1 = 0
    #         if(i in temp):
    #             continue
    #         for j in range(i+1,len(image_list)):
    #             if(check_inter(image_list[i][0],image_list[i][1],image_list[i][2],image_list[i][3],
    #             image_list[j][0],image_list[j][1],image_list[j][2],image_list[j][3])):
    #                 flag = 1
    #                 flag1 = 1
    #                 image_list1.append([min(image_list[i][0],image_list[j][0]),min(image_list[i][1],image_list[j][1]),max(image_list[i][2],image_list[j][2]),max(image_list[i][3],image_list[j][3])])
    #                 temp.append(j)
    #         if(flag1 == 0):
    #             image_list1.append([image_list[i][0],image_list[i][1],image_list[i][2],image_list[i][3]])

    #     image_list = image_list1

    #     if(flag==0):
    #         image_list = []
    #         for i in range(len(image_list1)):
    #             width_ = int((image_list1[i][3] - image_list1[i][1])*expand_p) #对背景进行额外的扩展
    #             height_ = int((image_list1[i][2] - image_list1[i][0])*expand_p)

    #             image_list.append(
    #                 Box(image_list1[i][1] - width_, image_list1[i][3] + width_, image_list1[i][0] - height_, height_+
    #             image_list1[i][2]))

    #             #image_list.append(image2[image_list1[i][0]-height_:image_list1[i][2]+height_,
    #              #                 image_list1[i][1]-width_:image_list1[i][3]+width_])

    #         break

    # return image_list


def check_inter(ax,ay,px,py,x1,y1,x2,y2):#检查两个矩阵是否相交

    newLeft = max(ay, y1)
    newRight = min(py, y2)

    newTop = max(ax, x1)
    newBottom = min(px, x2)


    return not(newLeft > newRight or newBottom < newTop)