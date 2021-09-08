from typing import Any, Tuple
import numpy as np

import cv2
import os
import numpy as np
from typing import List

np.array()

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
    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 0.5)
    im_mask = np.ones_like(base, dtype=np.uint8)
    h, w = im2Gray.shape
    # print(im2Gray.shape)
    #im1Reg = cv2.warpPerspective(im2Gray, M, (w, h))

    targetReg = cv2.warpPerspective(target, M, (w, h))

    imMask = cv2.warpPerspective(im_mask, M, (w, h))
    #im2_mask = im1Gray * imMask

    baseMask = base * imMask
    
    # cv2.namedWindow('reg',0)
    # cv2.imshow('reg',im1Reg)
    # cv2.imshow('org+mark',im2_mark)

    # origin, origin_mask, results = get_path(path, name)
    # cv2.imwrite(results, im1Reg)
    # cv2.imwrite(origin, im1)
    # cv2.imwrite(origin_mask, im2_mask)
    
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    # Match_percentage = 100 * len(matches) / MAX_FEATURES
    
    # print("描述子匹配点数为:", len(matches))
    # # print(matches[2])
    # print("描述子匹配率为: {:.0f}%".format(Match_percentage))
    
    
    # Max_dist = 0
    # Min_dist = 100

    # for i in range(0, len(matches)):
    #     dist = matches[i].distance
    #     if dist < Min_dist:
    #         Min_dist = dist
    #     if dist > Max_dist:
    #         Max_dist = dist
    # print("最大汉明距离:", Max_dist)
    # print("最小汉明距离:", Min_dist)

    
    # count = 0
    # for n in range(0, len(matches)):
    #     good_dist = 0.8 * Max_dist
    #     distance = matches[n].distance
    #     if distance <= good_dist:
    #         count += 1
    # print("配准成功点数共", count,"个")
    # success_percentage = 100 * count/len(matches)
    # print("配准成功率为: {:.1f}%".format(success_percentage))
    
    
    return baseMask, targetReg


