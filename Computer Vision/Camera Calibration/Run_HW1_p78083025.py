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
import skimage.io as io
import time
from UI import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QPixmap

obj_p = np.zeros((8 * 11, 3), np.float32)
obj_p[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
obj_points = []  # 3d points in real world space 存储3D点
img_points = []  # 2d points in image plane.存储2D点
# 获取标定板角点的位置
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 设置世界坐标的坐标
axis = np.float32([[1, 1, 0], [5, 1, 0], [3, 5, 0], [3, 3, -3]]).reshape(-1, 3)
keypoints = []
descriptors = []


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        # 将三个面板，加入stackedWidget
        self.stackedWidget.addWidget(self.page)
        self.stackedWidget.addWidget(self.page_1)
        self.stackedWidget.addWidget(self.page_2)
        self.stackedWidget.addWidget(self.page_3)
        self.stackedWidget.addWidget(self.page_4)
        self.stackedWidget.addWidget(self.page_5)
        self.pushButton_image.clicked.connect(self.HOME_page)
        self.pushButton_image_10.clicked.connect(self.HOME_page)
        self.pushButton_image_13.clicked.connect(self.HOME_page)
        self.pushButton_image_16.clicked.connect(self.HOME_page)
        self.pushButton_image_1.clicked.connect(self.Camera_Calibration_page)
        self.pushButton_image_2.clicked.connect(self.Augmented_Reality_page)
        self.pushButton_image_3.clicked.connect(self.Stereo_Disparity_Map_page)
        self.pushButton_image_4.clicked.connect(self.SIFT_page)
        self.pushButton_image_6.clicked.connect(self.Find_Corners_click)  # 1.1 Find Corners
        self.pushButton_image_7.clicked.connect(self.Find_Intrinsic_click)  # 1.2 Find Intrinsic
        self.pushButton_image_8.clicked.connect(self.Find_Extrinsic_click)  # 1.3 Find Extrinsic
        self.comboBox.currentIndexChanged.connect(self.get_comboBox)
        self.pushButton_image_9.clicked.connect(self.Find_Distortion_click)  # 1.4 Find Distortion
        self.pushButton_image_11.clicked.connect(self.Augmented_Reality_click)
        self.pushButton_image_12.clicked.connect(self.Stereo_Disparity_Map_click)
        self.pushButton_image_14.clicked.connect(self.Keypoints_click)
        self.pushButton_image_15.clicked.connect(self.Matched_Keypoints_click)
        self.flag = 1
        self.current_path = ' '
        self.image_path = ' '
        self.comboBox.addItems([' ', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'])
        self.distortion_coff = []

    def HOME_page(self):
        self.stackedWidget.setCurrentIndex(0)

    def Camera_Calibration_page(self):
        self.stackedWidget.setCurrentIndex(1)

    def Augmented_Reality_page(self):
        self.stackedWidget.setCurrentIndex(2)

    def Stereo_Disparity_Map_page(self):
        self.stackedWidget.setCurrentIndex(3)

    def SIFT_page(self):
        self.stackedWidget.setCurrentIndex(4)

    def get_comboBox(self):
        self.comboBox.setCurrentText(self.comboBox.currentText())
        self.image_path = self.comboBox.currentText()

    def get_current_path(self):
        paths = sys.path
        current_file = os.path.basename(__file__)
        for path in paths:
            try:
                if current_file in os.listdir(path):
                    self.current_path = path
                    break
            except (FileExistsError, FileNotFoundError) as e:
                print(e)

    def Find_Corners_click(self):
        self.get_current_path()
        for self.flag in range(15):
            img = cv2.imread(self.current_path + '/Q1_Image/' + str(self.flag + 1) + '.bmp')
            self.textBrowser_image.append(self.current_path + '/Q1_Image/' + str(self.flag + 1) + '.bmp')
            self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
            img2 = cv2.resize(img, (400, 300))
            io.imsave(self.current_path + "/Q1_Image/inputPixmap.png", img2)
            pix = QPixmap(self.current_path + "/Q1_Image/inputPixmap.png")
            self.label_inputimage.setPixmap(pix)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 提取角点
            # 第一个参数为图片，第二个为图片横纵角点的个数。
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret:
                obj_points.append(obj_p)
                # 在原角点的基础上寻找亚像素角点
                # winsize为搜索窗口边长的一半。
                # zeroZone：搜索区域中间的dead region边长的一半，有时用于避免自相关矩阵的奇异性。如果值设为(-1,-1)则表示没有这个区域。
                # criteria：角点精准化迭代过程的终止条件。也就是当迭代次数超过criteria.maxCount，或者角点位置变化小于criteria.epsilon时，停止迭代过程。
                corners2 = cv2.cornerSubPix(gray, corners, (12, 9), (-1, -1), criteria)
                if corners2.any():
                    img_points.append(corners2)
                else:
                    img_points.append(corners)
                #  Draw and display the corners把角点画出来
                cv2.drawChessboardCorners(img, (11, 8), corners2, ret)
                img = cv2.resize(img, (800, 700))
                cv2.imwrite(self.current_path + "/Q1_Image/resultPixmap.png", img)
                pix = QPixmap(self.current_path + "/Q1_Image/resultPixmap.png")
                self.label_inputimage_2.setPixmap(pix)
                QApplication.processEvents()
                time.sleep(0.5)
        self.flag = 1

    def Find_Intrinsic_click(self):
        img = cv2.imread(self.current_path + '/Q1_Image/1.bmp')
        size = (img.shape[1], img.shape[0])
        # 标定
        ret, mtx, dist, rve_cs, tve_cs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
        self.textBrowser_image_1.append('Intrinsic Matrix: ')
        self.textBrowser_image_1.append(str(mtx))
        self.textBrowser_image_1.append(' ')
        self.textBrowser_image_1.moveCursor(self.textBrowser_image_1.textCursor().End)  # 文本框顯示到底部
        self.distortion_coff = dist

    def Find_Extrinsic_click(self):
        self.get_current_path()
        img = cv2.imread(self.current_path + '/Q1_Image/' + self.image_path + '.bmp')
        self.textBrowser_image.append(self.current_path + '/Q1_Image/' + self.image_path + '.bmp')
        self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
        size = (img.shape[1], img.shape[0])
        # 标定
        _, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)  #
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 提取角点
        # 第一个参数为图片，第二个为图片横纵角点的个数。
        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        if ret:
            # 在原角点的基础上寻找亚像素角点
            # winsize为搜索窗口边长的一半。
            # zeroZone：搜索区域中间的dead region边长的一半，有时用于避免自相关矩阵的奇异性。如果值设为(-1,-1)则表示没有这个区域。
            # criteria：角点精准化迭代过程的终止条件。也就是当迭代次数超过criteria.maxCount，或者角点位置变化小于criteria.epsilon时，停止迭代过程。
            corners2 = cv2.cornerSubPix(gray, corners, (12, 9), (-1, -1), criteria)
            _, r_vec, t_vec, inl_ = cv2.solvePnPRansac(obj_p, corners2, mtx, dist)
            r_vec, _ = cv2.Rodrigues(r_vec, jacobian=None)
            etx = np.append(r_vec, t_vec, axis=1)
            self.textBrowser_image_1.append('Extrinsic Matrix: ')
            self.textBrowser_image_1.append(str(etx))
            self.textBrowser_image_1.append(' ')
            self.textBrowser_image_1.moveCursor(self.textBrowser_image_1.textCursor().End)  # 文本框顯示到底部

    def Find_Distortion_click(self):
        self.textBrowser_image_1.append('Distortion Matrix: ')
        self.textBrowser_image_1.append(str(self.distortion_coff[0]))
        self.textBrowser_image_1.append(' ')
        self.textBrowser_image_1.moveCursor(self.textBrowser_image_1.textCursor().End)  # 文本框顯示到底部

    def Augmented_Reality_draw(self, img, corners, img_pts):
        corner = tuple(corners[12].ravel())
        img = cv2.line(img, corner, tuple(img_pts[1].ravel()), (0, 0, 255), 5)
        img = cv2.line(img, corner, tuple(img_pts[2].ravel()), (0, 0, 255), 5)
        img = cv2.line(img, corner, tuple(img_pts[3].ravel()), (0, 0, 255), 5)
        img = cv2.line(img, tuple(img_pts[1].ravel()), tuple(img_pts[2].ravel()), (0, 0, 255), 5)
        img = cv2.line(img, tuple(img_pts[1].ravel()), tuple(img_pts[3].ravel()), (0, 0, 255), 5)
        img = cv2.line(img, tuple(img_pts[2].ravel()), tuple(img_pts[3].ravel()), (0, 0, 255), 5)
        return img

    def Augmented_Reality_click(self):
        self.get_current_path()
        while self.flag <= 5:
            img = cv2.imread(self.current_path + '/Q2_Image/' + str(self.flag) + '.bmp')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 提取角点
            # 第一个参数为图片，第二个为图片横纵角点的个数。
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret:
                obj_points.append(obj_p)
                # 在原角点的基础上寻找亚像素角点
                # winsize为搜索窗口边长的一半。
                # zeroZone：搜索区域中间的dead region边长的一半，有时用于避免自相关矩阵的奇异性。如果值设为(-1,-1)则表示没有这个区域。
                # criteria：角点精准化迭代过程的终止条件。也就是当迭代次数超过criteria.maxCount，或者角点位置变化小于criteria.epsilon时，停止迭代过程。
                corners2 = cv2.cornerSubPix(gray, corners, (12, 9), (-1, -1), criteria)
                if corners2.any():
                    img_points.append(corners2)
                else:
                    img_points.append(corners)
                self.flag += 1
            else:
                self.flag = 1
        img = cv2.imread(self.current_path + '/Q2_Image/1.bmp')
        size = (img.shape[1], img.shape[0])
        for i in range(5):
            # print('2')
            img = cv2.imread(self.current_path + '/Q2_Image/' + str(i + 1) + '.bmp')
            # 标定
            _, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # 提取角点
            # 第一个参数为图片，第二个为图片横纵角点的个数。
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)
            if ret:
                # 在原角点的基础上寻找亚像素角点
                # winsize为搜索窗口边长的一半。
                # zeroZone：搜索区域中间的dead region边长的一半，有时用于避免自相关矩阵的奇异性。如果值设为(-1,-1)则表示没有这个区域。
                # criteria：角点精准化迭代过程的终止条件。也就是当迭代次数超过criteria.maxCount，或者角点位置变化小于criteria.epsilon时，停止迭代过程。
                corners2 = cv2.cornerSubPix(gray, corners, (12, 9), (-1, -1), criteria)
                _, r_vec, t_vec, inl_ = cv2.solvePnPRansac(obj_p, corners2, mtx, dist)
                img_pts, jac = cv2.projectPoints(axis, r_vec, t_vec, mtx, dist)
                # 可视化角点
                img = self.Augmented_Reality_draw(img, corners2, img_pts)
                img = cv2.resize(img, (800, 700))
                cv2.imwrite(self.current_path + "/Q2_Image/output/resultPixmap_" + str(i + 1) + ".png", img)
            # Show the image 0.5 seconds and close it
            pix = QPixmap(self.current_path + "/Q2_Image/output/resultPixmap_" + str(i + 1) + ".png")
            self.label_inputimage_3.setPixmap(pix)
            QApplication.processEvents()
            time.sleep(0.5)

    def Stereo_Disparity_Map_click(self):
        self.get_current_path()
        im_l = cv2.imread(self.current_path + '/Q3_Image/imL.png', 0)
        im_r = cv2.imread(self.current_path + '/Q3_Image/imR.png', 0)
        pix = QPixmap(self.current_path + "/Q3_Image/imL.png")
        self.label_inputimage_8.setPixmap(pix)
        pix = QPixmap(self.current_path + "/Q3_Image/imR.png")
        self.label_inputimage_9.setPixmap(pix)
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(im_l, im_r)
        img = cv2.resize(disparity, (800, 700))
        cv2.imwrite(self.current_path + "/Q3_Image/resultPixmap.png", img)
        pix = QPixmap(self.current_path + "/Q3_Image/resultPixmap.png")
        self.label_inputimage_4.setPixmap(pix)

    def Keypoints_click(self):
        self.get_current_path()
        Aerial1 = cv2.imread(self.current_path + '/Q4_Image/Aerial1.jpg', 0)
        Aerial2 = cv2.imread(self.current_path + '/Q4_Image/Aerial2.jpg', 0)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1 = sift.detect(Aerial1)
        kp2 = sift.detect(Aerial2)
        tmp_kp_sort1 = sorted(kp1, key=lambda x: x.size, reverse=True)[0:7]
        tmp_kp_sort2 = sorted(kp2, key=lambda x: x.size, reverse=True)[0:7]
        keypoints_Aerial1, descriptor_Aerial1 = sift.compute(Aerial1, tmp_kp_sort1)
        keypoints.append(keypoints_Aerial1)
        descriptors.append(descriptor_Aerial1)
        keypoints_Aerial2, descriptor_Aerial2 = sift.compute(Aerial2, tmp_kp_sort2)
        keypoints.append(keypoints_Aerial2)
        descriptors.append(descriptor_Aerial2)
        img_Aerial1 = cv2.drawKeypoints(image=Aerial1, outImage=Aerial1, keypoints=keypoints_Aerial1, color=(0, 255, 0))
        img_Aerial2 = cv2.drawKeypoints(image=Aerial2, outImage=Aerial2, keypoints=keypoints_Aerial2, color=(0, 0, 255))
        cv2.imwrite(self.current_path + "/Q4_Image/FeatureAerial1.jpg", img_Aerial1)
        cv2.imwrite(self.current_path + "/Q4_Image/FeatureAerial2.jpg", img_Aerial2)
        img_Aerial1 = cv2.resize(img_Aerial1, (300, 400))
        cv2.imwrite(self.current_path + "/Q4_Image/Keypoints1Pixmap.png", img_Aerial1)
        pix = QPixmap(self.current_path + "/Q4_Image/Keypoints1Pixmap.png")
        self.label_inputimage_5.setPixmap(pix)
        img_Aerial2 = cv2.resize(img_Aerial2, (300, 400))
        cv2.imwrite(self.current_path + "/Q4_Image/Keypoints2Pixmap.png", img_Aerial2)
        pix = QPixmap(self.current_path + "/Q4_Image/Keypoints2Pixmap.png")
        self.label_inputimage_6.setPixmap(pix)

    def Matched_Keypoints_click(self):
        MIN_MATCH_COUNT = 1
        self.get_current_path()
        img1 = cv2.imread(self.current_path + '/Q4_Image/FeatureAerial1.jpg')
        img2 = cv2.imread(self.current_path + '/Q4_Image/FeatureAerial2.jpg')
        flann = cv2.BFMatcher(cv2.NORM_L2)  # 特征点匹配用的是BFMatcher，brute force暴力匹配，就是选取几个最近的
        # 使用KNN算法匹配
        matches = flann.knnMatch(descriptors[0], descriptors[1], k=2)
        # 去除错误匹配
        good = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
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
        img3 = cv2.drawMatches(img1, keypoints[0], img2, keypoints[1], good, None, **draw_params)
        img3 = cv2.resize(img3, (900, 700))
        cv2.imwrite(self.current_path + "/Q4_Image/MatchedKeypointsPixmap.png", img3)
        pix = QPixmap(self.current_path + "/Q4_Image/MatchedKeypointsPixmap.png")
        self.label_inputimage_7.setPixmap(pix)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
