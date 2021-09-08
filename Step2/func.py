import cv2
import numpy as np
from collections import Counter
from skimage import io, measure
from time import *
import warnings
import os
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def clust_cv(FVS, components, new):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 0.1)
    FVS = np.float32(FVS)

    ret, label, center = cv2.kmeans(FVS, components, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

    change_map = np.reshape(label, (new[0], new[1]))

    count=Counter(label.ravel())
    least_index = min(count, key=count.get)

    change_map[change_map == least_index] = 255
    change_map[change_map != 255] = 0

    return change_map

def remove_small_points(img, threshold_point,threshold_point2,image2,ab_num):
    ################删去大小符合阈值的连通区域
    img_label, num = measure.label(img, return_num=True,connectivity=2)#输出二值图像中所有的连通域
    props = measure.regionprops(img_label)#输出连通域的属性，包括面积等
    numm=0

    res_area = []
    for i in range(1, len(props)):
        if props[i].area > threshold_point and props[i].area <threshold_point2:
            #tmp = (img_label == i + 1).astype(np.uint8)
            #resMatrix += tmp #组合所有符合条件的连通域
            numm+=1
            res_area.append(props[i])
            #print(f"{props[i].bbox[1]},{props[i].bbox[3]}:{props[i].bbox[0]},{props[i].bbox[2]}")
            #crop = image2[props[i].bbox[0]:props[i].bbox[2], props[i].bbox[1]:props[i].bbox[3]]

    res_area.sort(key=lambda t: t.area,reverse=True)
    image_list=[]

    for i in range(min(numm,ab_num)):
        image_list.append([res_area[i].bbox[0],res_area[i].bbox[1],res_area[i].bbox[2],res_area[i].bbox[3]])
        #print(f"{res_area[i].bbox[1]},{res_area[i].bbox[3]},{res_area[i].bbox[0]},{res_area[i].bbox[2]}")
        #cv2.rectangle(image2, (res_area[i].bbox[1], res_area[i].bbox[0]), (res_area[i].bbox[3], res_area[i].bbox[2]), (0, 0, 255),
         #             2)
    while(1):
        image_list1=[]
        flag = 0
        temp = []
        for i in range(len(image_list)):
            flag1 = 0
            if(i in temp):
                continue
            for j in range(i+1,len(image_list)):
                if(check_inter(image_list[i][0],image_list[i][1],image_list[i][2],image_list[i][3],
                image_list[j][0],image_list[j][1],image_list[j][2],image_list[j][3])):
                    # print(f"{image_list[i][0]},{image_list[i][1]},{image_list[i][2]},{image_list[i][3]}")
                    # print(f"{image_list[j][0]},{image_list[j][1]},{image_list[j][2]},{image_list[j][3]}")
                    # exit(0)
                    flag = 1
                    flag1 = 1
                    image_list1.append([min(image_list[i][0],image_list[j][0]),min(image_list[i][1],image_list[j][1]),max(image_list[i][2],image_list[j][2]),max(image_list[i][3],image_list[j][3])])
                    temp.append(j)
            if(flag1 == 0):
                image_list1.append([image_list[i][0],image_list[i][1],image_list[i][2],image_list[i][3]])

       # print(len(image_list1))

        image_list = image_list1
        #print(len(image_list1))
        if(flag==0):
            image_list = []
            for i in range(len(image_list1)):
                image_list.append(image2[image_list1[i][0]:image_list1[i][2],
                                  image_list1[i][1] :image_list1[i][3]])

            break

    # for i in range(len(image_list)):
    #     cv2.rectangle(image2, (image_list[i][1], image_list[i][0]), (image_list[i][3], image_list[i][2]), (0, 0, 255),
    #                  2)
    #print(f"联通域的数目{numm}")
    return image_list


def check_inter(ax,ay,px,py,x1,y1,x2,y2):#检查两个矩阵是否相交

    newLeft = max(ay, y1)
    newRight = min(py, y2)

    newTop = max(ax, x1)
    newBottom = min(px, x2)


    return not(newLeft > newRight or newBottom < newTop)

def HPF_cv(img):#这其实对边缘进行检测，可否将两幅图的亮度调整到同一水平？
    laplace = cv2.Laplacian(img, cv2.CV_16S, ksize=5)
    laplacian = cv2.convertScaleAbs(laplace)

    return laplacian

def HPF(img, D0, N=2, type='hp', filter='Gaussian'):
    '''
    频域滤波器
    Args:
        img: 灰度图片
        D0: 截止频率
        N: butterworth的阶数(默认使用二阶)
        type: lp-低通 hp-高通
        filter:butterworth、ideal、Gaussian即巴特沃斯、理想、高斯滤波器
    Returns:
        imgback：滤波后的图像
    '''
    # 离散傅里叶变换
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 中心化
    dtf_shift = np.fft.fftshift(dft) #将图像中的低频部分移动到图像的中心

    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 计算频谱中心
    mask = np.zeros((rows, cols, 2))  # 生成rows行cols列的二维矩阵

    #优化循环的赋值过程

    x_l=list(range(rows))
    xx = [val for val in x_l for i in range(2)]
    xx = [val for val in xx for i in range(cols)]
    xx = np.array(xx)
    xx = xx.reshape(rows,cols,2)

    y_l=list(range(cols))
    yy = [val for val in y_l for i in range(2)]
    yy=[yy]
    yy =[val for val in yy for i in range(rows)]
    yy = np.array(yy)
    yy = yy.reshape(rows, cols, 2)

    D = np.sqrt((xx-crow)**2+(yy-ccol)**2)

    if (filter.lower() == 'butterworth'):  # 巴特沃斯滤波器
        if (type == 'lp'):
            mask = 1 / (1 + (D / D0) ** (2 * N))
        elif (type == 'hp'):
            mask = 1 / (1 + (D0 / D) ** (2 * N))
        else:
            assert ('type error')
    elif (filter.lower() == 'ideal'):  # 理想滤波器
        if (type == 'lp'):
            mask = (D <= D0)
        elif (type == 'hp'):
            mask = (D > D0)
        else:
            assert ('type error')

    elif (filter.lower() == 'gaussian'):  # 高斯滤波器
        if (type == 'lp'):
            mask = np.exp(-(D * D) / (2 * D0 * D0))
        elif (type == 'hp'):
            oness=np.ones((rows, cols, 2))
           # print(oness)
            mask = (1 - np.exp(-(D * D) / (2 * D0 * D0)))
        else:
            assert ('type error')

    fshift = dtf_shift * mask

    f_ishift = np.fft.ifftshift(fshift) #将图像的低频和高频部分移动到图像原来的位置
    img_back = cv2.idft(f_ishift) #进行傅里叶的逆变化
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])  # 计算像素梯度的绝对值
    img_back = np.abs(img_back)
    # img_back = (img_back - np.amin(img_back)) / (np.amax(img_back) - np.amin(img_back))
    return img_back

