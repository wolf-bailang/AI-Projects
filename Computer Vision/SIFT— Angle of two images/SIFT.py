# ！/usr/bin/python3.7
# encoding: utf-8
"""
@author: Junbin Zhang
@email: p78083025@gs.ncku.edu.tw
@time: 2020/10/25
"""

from __future__ import print_function

import sys
import os
import cv2
import numpy as np
#import skimage.io as io
#import time
from numpy import *
import math
#import random

keypoints = []
descriptors = []


def Keypoints(Template_image, Target_image):
    # 用SIFT提取特徵點
    sift = cv2.xfeatures2d.SIFT_create()
    # 模板圖特徵點
    kp1 = sift.detect(Template_image, None)
    # 測試圖特徵點
    kp2 = sift.detect(Target_image, None)
    # 排序，取前20個
    tmp_kp_sort1 = sorted(kp1, key=lambda x: x.size, reverse=True)[:20]
    tmp_kp_sort2 = sorted(kp2, key=lambda x: x.size, reverse=True)[:20]
    # 存儲
    keypoints_Template_image, descriptor_Template_image = sift.compute(Template_image, tmp_kp_sort1)
    keypoints.append(keypoints_Template_image)
    descriptors.append(descriptor_Template_image)
    keypoints_Target_image, descriptor_Target_image = sift.compute(Target_image, tmp_kp_sort2)
    keypoints.append(keypoints_Target_image)
    descriptors.append(descriptor_Target_image)
    # img_Template_image = cv2.drawKeypoints(image=Template_image, outImage=Template_image, keypoints=keypoints_Template_image, color=(0, 255, 0))
    # img_Target_image = cv2.drawKeypoints(image=Target_image, outImage=Target_image, keypoints=keypoints_Target_image, color=(0, 0, 255))
    # cv2.imwrite('/home/wolf/桌面/11111111/Feature_Template_image.jpg', img_Template_image)
    # cv2.imwrite('/home/wolf/桌面/11111111/Feature_Target_image.jpg', img_Target_image)
    # cv2.imshow('Feature_Template_image', img_Template_image)
    # cv2.imshow('Feature_Target_image', img_Target_image)
    # cv2.waitKey(0)


def angle(v1, v2):
    # 計算四個點，也就是兩條直線的夾角
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = float(angle1 * 180.0 / math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = float(angle2 * 180.0 / math.pi)
    if angle1 * angle2 >= 0.0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180.0:
            included_angle = 360.0 - included_angle
    return included_angle


# 根據第一列值從小到大排序
def takeSecond(elem):
    return elem[0]


def Matched_Keypoints(Template_image, Target_image):
    src_pts = []
    dst_pts = []
    text = " "
    MIN_MATCH_COUNT = 1
    # 特征点匹配用的是BFMatcher，brute force暴力匹配，就是选取几个最近的
    flann = cv2.BFMatcher(cv2.NORM_L2)
    # 使用KNN算法匹配
    matches = flann.knnMatch(descriptors[0], descriptors[1], k=2)
    # 去除错误匹配
    good = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good.append(m)
    # 单应性
    if len(good) > MIN_MATCH_COUNT:
        # 改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置
        src_pts = np.float32([keypoints[0][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[1][m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # findHomography 函数是计算变换矩阵
        # 参数cv2.RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H，即返回值M
        # 返回值：M 为变换矩阵，mask是掩模
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        # ravel方法将数据降维处理，最后并转换成列表格式
        matchesMask = mask.ravel().tolist()
    else:
        print("Not enough matches are found - %d/%d") % (len(good), MIN_MATCH_COUNT)
        matchesMask = None
    # 显示匹配结果
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    # 畫匹配的特徵點
    image_Matches = cv2.drawMatches(Template_image, keypoints[0], Target_image, keypoints[1], good, None, **draw_params)
    cv2.imwrite('/home/wolf/桌面/11111111/MatchedKeypoints.png', image_Matches)
    cv2.imshow('MatchedKeypoints.png', image_Matches)
    # cv2.waitKey(1)
    pop = []
    # 將模板圖和測試圖匹配的特徵點提取出來
    for i in range(len(matchesMask)):
        if matchesMask[i] == 1:
            pop.append([src_pts[i][0][0], src_pts[i][0][1], dst_pts[i][0][0], dst_pts[i][0][1]])
        i += 1
    # 匹配的特徵點排序，以x從小到大
    # pop.sort(key=takeSecond)
    i = 0
    j = 0
    k = 0
    # 指示是否有找到滿足的４個點
    FLAGE = 0
    t = True
    while t:
        for i in range(0, len(pop)):
            for j in range(0, len(pop)):
                if i == j:
                    continue
                # 根據兩點座標計算直線距離
                # 來源於模板圖的兩個點
                AB_distance = np.sqrt(np.sum((np.array([pop[i][0], pop[i][1]]) - np.array([pop[j][0], pop[j][1]])) ** 2))
                # 來源於測試圖的兩個點
                CD_distance = np.sqrt(np.sum((np.array([pop[i][2], pop[i][3]]) - np.array([pop[j][2], pop[j][3]])) ** 2))
                # 距離在閾值內，判斷點相似
                if abs(AB_distance - CD_distance) <= 15:
                    k = 1
                    break
                else:
                    k = 0
            if k == 1:
                t = False
                break
            if i == len(pop) - 1:
                t = False
                # 沒有有找到滿足的４個點
                FLAGE = 1
                break
    # 直線的點座標矩陣
    AB = [pop[i][0], pop[i][1], pop[j][0], pop[j][1]]
    CD = [pop[i][2], pop[i][3], pop[j][2], pop[j][3]]
    # 直線的方向始終由右往左
    if pop[i][0] < pop[j][0]:
        temp = i
        i = j
        j = temp
    # 兩直線的夾角
    angle_ABCD = angle(AB, CD)
    # 畫直線
    angle_Template_image = cv2.arrowedLine(Template_image, tuple(pop[i][0:2]), tuple(pop[j][0:2]), (0, 255, 0), 2)
    angle_Target_image = cv2.arrowedLine(Target_image, tuple(pop[i][2:4]), tuple(pop[j][2:4]), (0, 0, 255), 2)
    # 模板圖和測試圖合併爲一張
    image = np.concatenate((angle_Template_image, angle_Target_image), axis=1)
    # 夾角在閾值內，且找到了４個點．說明方向相同
    if angle_ABCD < 30 and FLAGE == 0:
        text = 'OK'
    else:
        text = 'No'
    text = "{:.3f}   :   {}".format(angle_ABCD, text)
    # 圖片上顯示結果
    cv2.putText(image, text, (5, 10), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.3, color=(0, 255, 255), thickness=1)
    cv2.imshow(text, image)
    cv2.imwrite('/home/wolf/桌面/11111111/image.png', image)
    cv2.waitKey(0)

def Read_Image():
    # 讀取一張圖作爲匹配模板
    Template_image = cv2.imread('./Template_image/frame00294_1.jpg')
    # cv2.imshow('Template_image', Template_image)
    # 讀取一張測試圖
    Target_image = cv2.imread('./Target_image/frame00015_1.jpg')
    width = Template_image.shape[1]
    height = Template_image.shape[0]
    # 將測試圖的大小resize成與模板圖一樣
    Target_image = cv2.resize(Target_image, (width, height))
    # cv2.imshow('Target_image', Target_image)
    return Template_image, Target_image


if __name__ == "__main__":
    Template_image, Target_image = Read_Image()
    # 提取關鍵特徵點
    Keypoints(Template_image, Target_image)
    # 特徵點匹配，計算角度
    Matched_Keypoints(Template_image, Target_image)

