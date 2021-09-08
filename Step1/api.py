from typing import Any, Tuple
import numpy as np

import cv2
import os
import numpy as np
from typing import List

def api(base: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # do some manipulation
    im1Gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    MAX_FEATURES = 5000
    # 确定要查找的最大关键点数
    
    orb = cv2.ORB_create(MAX_FEATURES)
    
    # detect特征点、计算描述子
    kp1 = orb.detect(im1Gray)
    kp1, des1 = orb.compute(im1Gray, kp1)
    kp2 = orb.detect(im2Gray)
    kp2, des2 = orb.compute(im2Gray, kp2)
    # 目前是暴力匹配
    # 交叉匹配则在后加True参数
    matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING) # 返回两个测试对象之间的汉明距离
    matches = matcher.match(des1, des2)
    # 按相近程度排序
    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    # 单应性矩阵计算
    M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 0.5)

    imOnes = np.ones_like(target, dtype=np.uint8)

    h, w = im2Gray.shape
    # print(im2Gray.shape)
    #im1Reg = cv2.warpPerspective(im2Gray, M, (w, h))

    baseReg = cv2.warpPerspective(base, M, (w, h), flags=cv2.INTER_LINEAR)

    imMask = cv2.warpPerspective(imOnes, M, (w, h), flags=cv2.INTER_LINEAR)
    #im2_mask = im1Gray * imMask

    targetMask = target * imMask
    
    
    return baseReg, targetMask


