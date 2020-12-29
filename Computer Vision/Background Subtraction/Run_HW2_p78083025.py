# ！/usr/bin/python3.6
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
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPalette, QBrush, QPixmap
from tqdm import tqdm
from PIL import Image
import time
import matplotlib.pyplot as plt
from cv2 import Stitcher
from sklearn.decomposition import PCA
import tkinter as tk
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import img_to_array

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
        self.pushButton_image_28.clicked.connect(self.HOME_page)
        self.pushButton_image_1.clicked.connect(self.Background_Subtraction_page)
        self.pushButton_image_2.clicked.connect(self.Optical_Flow_page)
        self.pushButton_image_3.clicked.connect(self.Perspective_Transform_page)
        self.pushButton_image_4.clicked.connect(self.PCA_page)
        self.pushButton_image_5.clicked.connect(self.ResNet50_Classifier_page)
        self.pushButton_image_6.clicked.connect(self.Preprocessing_click)
        self.pushButton_image_7.clicked.connect(self.Video_tracking_click)
        self.pushButton_image_8.clicked.connect(self.Play_video_click)
        self.pushButton_image_11.clicked.connect(self.Background_Subtraction_click)
        self.pushButton_image_12.clicked.connect(self.Perspective_Transform_click)
        self.pushButton_image_14.clicked.connect(self.Image_Reconstruction_click)
        self.pushButton_image_15.clicked.connect(self.Reconstruction_Error_click)
        self.pushButton_image_17.clicked.connect(self.Play_video_click)
        self.pushButton_image_18.clicked.connect(self.Play_video_click)
        self.pushButton_image_25.clicked.connect(self.Training_Images_click)
        self.pushButton_image_26.clicked.connect(self.TensorBoard_click)
        # self.pushButton_image_27.clicked.connect(self.Test_click)
        self.pushButton_image_27.clicked.connect(self.Test_openFile_image_click)
        self.pushButton_image_29.clicked.connect(self.Model_ResNet50_click)
        self.current_path = ' '
        self.image_path = ' '
        self.Play_video = 0
        self.TEST_SIZE = 0.5
        self.RANDOM_STATE = 2018
        self.BATCH_SIZE = 64
        self.NO_EPOCHS = 20
        self.NUM_CLASSES = 2
        self.SAMPLE_SIZE = 20000
        self.PATH = self.current_path + '/Q5/'
        self.TRAIN_FOLDER = self.current_path + '/Q5/train/'
        self.TEST_FOLDER = self.current_path + '/Q5/test/'
        self.IMG_SIZE = 224
        self.RESNET_WEIGHTS_PATH = self.current_path + '/Q5/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

    def HOME_page(self):
        self.stackedWidget.setCurrentIndex(0)

    def Optical_Flow_page(self):
        self.stackedWidget.setCurrentIndex(1)

    def Background_Subtraction_page(self):
        self.stackedWidget.setCurrentIndex(2)

    def Perspective_Transform_page(self):
        self.stackedWidget.setCurrentIndex(3)

    def PCA_page(self):
        self.stackedWidget.setCurrentIndex(4)

    def ResNet50_Classifier_page(self):
        self.stackedWidget.setCurrentIndex(5)

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

    def Background_Subtraction_click(self):
        Simple_Gaussian_mean = np.zeros((176, 320), dtype=np.float)
        Simple_Gaussian_standard_deviation = np.zeros((176, 320), dtype=np.float)
        self.get_current_path()
        self.Play_video = 1
        self.textBrowser_image_2.append('获取一个视频')
        self.textBrowser_image_2.moveCursor(self.textBrowser_image_2.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        # 获取一个视频并打开
        cap = cv2.VideoCapture(self.current_path + '/Q1_Image/bgSub.mp4')
        if cap.isOpened():  # VideoCaputre对象是否成功打开
            self.textBrowser_image_2.append('已经打开视频文件')
            self.textBrowser_image_2.moveCursor(self.textBrowser_image_2.textCursor().End)  # 文本框顯示到底部
            QApplication.processEvents()
            fps = cap.get(cv2.CAP_PROP_FPS)  # 返回视频的fps--帧率
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 返回视频的宽
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 返回视频的高
            # print('fps:', fps, 'width:', width, 'height:', height)
            self.textBrowser_image_2.append('fps=' + str(fps) + ' width=' + str(width) + ' height=' + str(height))
            self.textBrowser_image_2.moveCursor(self.textBrowser_image_2.textCursor().End)  # 文本框顯示到底部
            QApplication.processEvents()
            i = 0
            while 1:
                if i == 50:
                    self.textBrowser_image_2.append('截取视频的前50帧图像，保存结束')
                    self.textBrowser_image_2.moveCursor(self.textBrowser_image_2.textCursor().End)  # 文本框顯示到底部
                    QApplication.processEvents()
                    break
                else:
                    i = i + 1
                    ret, frame = cap.read()  # 读取一帧视频
                    # ret 读取了数据就返回True,没有读取数据(已到尾部)就返回False
                    # frame 返回读取的视频数据--一帧数据
                    file_name = self.current_path + '/Q1_Image/frame/' + str(i) + '.jpg'
                    cv2.imwrite(file_name, frame)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    for row in range(height):  # 遍历高
                        for col in range(width):  # 遍历宽
                            Simple_Gaussian_mean[row, col] += gray[row, col]
        else:
            self.textBrowser_image_2.append('视频文件打开失败')
            self.textBrowser_image_2.moveCursor(self.textBrowser_image_2.textCursor().End)  # 文本框顯示到底部
            QApplication.processEvents()
        self.textBrowser_image_2.append('正在建立高斯模型.....')
        self.textBrowser_image_2.moveCursor(self.textBrowser_image_2.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        for row in range(176):  # 遍历高
            for col in range(320):  # 遍历宽
                Simple_Gaussian_mean[row, col] = Simple_Gaussian_mean[row, col] / 50.0
        i = 1
        while i <= 50:
            img = cv2.imread(self.current_path + '/Q1_Image/frame/' + str(i) + '.jpg')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for row in range(176):  # 遍历高
                for col in range(320):  # 遍历宽
                    Simple_Gaussian_standard_deviation[row, col] += (gray[row, col] - Simple_Gaussian_mean[
                        row, col]) ** 2
            i = i + 1
        for row in range(176):  # 遍历高
            for col in range(320):  # 遍历宽
                if np.sqrt(Simple_Gaussian_standard_deviation[row, col] / 50.0) < 5.0:
                    Simple_Gaussian_standard_deviation[row, col] = 5.0
                else:
                    Simple_Gaussian_standard_deviation[row, col] = np.sqrt(
                        Simple_Gaussian_standard_deviation[row, col] / 50.0)
        # 获取一个视频并打开
        cap = cv2.VideoCapture(self.current_path + '/Q1_Image/bgSub.mp4')
        if cap.isOpened():  # VideoCaputre对象是否成功打开
            # print('已经打开了视频文件')
            fps = cap.get(cv2.CAP_PROP_FPS)  # 返回视频的fps--帧率
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 返回视频的宽
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 返回视频的高
            # print('fps:', fps, 'width:', width, 'height:', height)
            i = 0
            ret, frame = cap.read()  # 读取一帧视频
            while ret:
                i = i + 1
                ret, frame = cap.read()  # 读取一帧视频
                if ret:
                    k = 0
                else:
                    break
                if i < 50:
                    # print('保存了视频的前15帧图像，保存结束')
                    continue
                else:
                    # ret 读取了数据就返回True,没有读取数据(已到尾部)就返回False
                    # frame 返回读取的视频数据--一帧数据
                    file_name = self.current_path + '/Q1_Image/frame/' + str(i) + '.jpg'
                    cv2.imwrite(file_name, frame)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    for row in range(height):  # 遍历高
                        for col in range(width):  # 遍历宽
                            if (gray[row, col] - Simple_Gaussian_mean[row, col]) > 4 * \
                                    Simple_Gaussian_standard_deviation[
                                        row, col]:
                                gray[row, col] = 255
                            else:
                                gray[row, col] = 0
                    # 横向连接
                    imgorg_temp = cv2.imread(self.current_path + '/Q1_Image/frame/' + str(i) + '.jpg')
                    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    image = np.concatenate((imgorg_temp, gray_rgb), axis=1)
                    image = cv2.resize(image, (1000, 400))
                    file_name = self.current_path + '/Q1_Image/result/' + str(i) + '.jpg'
                    cv2.imwrite(file_name, image)
        self.textBrowser_image_2.append('生成视频文件......')
        self.textBrowser_image_2.moveCursor(self.textBrowser_image_2.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        fsp = 30
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # 对应格式 mp4
        n = 0
        video_path = self.current_path + '/Q1_Image/result.mp4'.format(n)
        img_path = self.current_path + '/Q1_Image/result/'.format(n)
        list_image = os.listdir(self.current_path + '/Q1_Image/result/'.format(n))
        list_image.sort()
        list_image = [os.path.join(img_path, x) for x in list_image]
        width = cv2.imread(list_image[0]).shape[1]
        height = cv2.imread(list_image[0]).shape[0]
        video_out = cv2.VideoWriter(video_path, fourcc, fsp, (width, height))
        count = 1
        for i in range(len(list_image)):
            frame = cv2.imread(self.current_path + '/Q1_Image/result/' + str(i) + '.jpg')
            video_out.write(frame)
            count += 1
        video_out.release()
        self.textBrowser_image_2.append('end')
        self.textBrowser_image_4.append(' ')
        self.textBrowser_image_2.moveCursor(self.textBrowser_image_2.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        for i in range(len(list_image)):
            pix = QPixmap(self.current_path + "/Q1_Image/result/" + str(i) + ".jpg")
            self.label_inputimage_3.setPixmap(pix)
            QApplication.processEvents()
            time.sleep(0.05)

    def Play_video_click(self):
        self.get_current_path()
        img_path = self.current_path + '/Q' + str(self.Play_video) + '_Image/result/'.format()
        list_image = os.listdir(self.current_path + '/Q1_Image/result/'.format(0))
        list_image.sort()
        list_image = [os.path.join(img_path, x) for x in list_image]
        for i in range(len(list_image)):
            pix = QPixmap(self.current_path + '/Q' + str(self.Play_video) + '_Image/result/' + str(i) + ".jpg")
            if self.Play_video == 1:
                self.label_inputimage_3.setPixmap(pix)
            if self.Play_video == 2:
                self.label_inputimage_2.setPixmap(pix)
            if self.Play_video == 3:
                self.label_inputimage_4.setPixmap(pix)
            QApplication.processEvents()
            time.sleep(0.05)

    def Video_tracking_click(self):
        temp = []
        p0 = np.zeros((7, 1, 2), dtype=np.float32)
        self.get_current_path()
        self.Play_video = 2
        self.textBrowser_image.append(' ')
        self.textBrowser_image.append('Video_tracking')
        self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        # 读取视频
        self.textBrowser_image.append('读取视频')
        self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        cap = cv2.VideoCapture(self.current_path + '/Q2_Image/opticalFlow.mp4')
        ret, old_frame = cap.read()
        # ret 读取了数据就返回True,没有读取数据(已到尾部)就返回False
        # frame 返回读取的视频数据--一帧数据
        self.textBrowser_image.append('Setup SimpleBlobDetector parameters')
        self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        # params.filterByColor = True    # 斑点颜色的限制变量
        # params.blobColor = 190   # 表示只提取黑色斑点；如果该变量为255，表示只提取白色斑点
        # Change thresholds
        params.minThreshold = 80
        params.maxThreshold = 180
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 20
        params.maxArea = 80
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.8  # 圆的类圆性
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.87
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.53
        # Create a detector with the parameters
        detector = cv2.SimpleBlobDetector_create(params)
        # Detect blobs.
        keypoints = detector.detect(old_frame)
        tmp_kp_sort1 = sorted(keypoints, key=lambda x: x.size, reverse=True)[0:7]
        k = 0
        for f in tmp_kp_sort1:
            a = []
            a.append([int(f.pt[0]), int(f.pt[1])])
            temp.append(np.array(a, dtype=np.float32))
            p0[k] = np.array(a, dtype=np.float32)
            k += 1
        # params for ShiTomasi corner detection构建角点检测所需参数
        feature_params = dict(maxCorners=7, qualityLevel=0.3, minDistance=7, blockSize=5)
        # Parameters for lucas kanade optical flow lucas kanade参数
        lk_params = dict(winSize=(11, 11), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        color = (0, 0, 255)
        # Take first frame and find corners in it拿到第一帧图像并灰度化作为前一帧图片
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        # 返回所有检测特征点，需要输入图片，角点的最大数量，品质因子，minDistance=7如果这个角点里有比这个强的就不要这个弱的
        # Create a mask image for drawing purposes创建一个mask, 用于进行横线的绘制
        mask = np.zeros_like(old_frame)
        self.textBrowser_image.append('optical flow ......')
        self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        j = 0
        while True:
            # 读取图片灰度化作为后一张图片的输入
            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # calculate optical flow进行光流检测需要输入前一帧和当前图像及前一帧检测到的角点tmp_kp_sort1
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
            # Select good points读取运动了的角点st == 1表示检测到的运动物体，即v和u表示为0
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            # draw the tracks绘制轨迹
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color, 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color, -1)
            # 将两个图片进行结合，并进行图片展示
            result_image = cv2.add(frame, mask)
            image = cv2.resize(result_image, (770, 560))
            file_name = self.current_path + '/Q2_Image/result/' + str(j) + '.jpg'
            cv2.imwrite(file_name, image)
            # Now update the previous frame and previous points更新前一帧图片和角点的位置
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
            j += 1
        self.textBrowser_image.append('正在生成视频......')
        self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        fsp = 30
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # 对应格式 mp4
        n = 0
        video_path = self.current_path + '/Q2_Image/result.mp4'.format(n)
        img_path = self.current_path + '/Q2_Image/result/'.format(n)
        list_image = os.listdir(self.current_path + '/Q2_Image/result/'.format(n))
        list_image.sort()
        list_image = [os.path.join(img_path, x) for x in list_image]
        width = cv2.imread(list_image[0]).shape[1]
        height = cv2.imread(list_image[0]).shape[0]
        video_out = cv2.VideoWriter(video_path, fourcc, fsp, (width, height))
        count = 1
        for i in range(len(list_image)):
            frame = cv2.imread(self.current_path + '/Q2_Image/result/' + str(i) + '.jpg')
            video_out.write(frame)
            count += 1
        video_out.release()
        self.textBrowser_image.append('end')
        self.textBrowser_image_4.append(' ')
        self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        for i in range(len(list_image)):
            pix = QPixmap(self.current_path + "/Q2_Image/result/" + str(i) + ".jpg")
            self.label_inputimage_2.setPixmap(pix)
            QApplication.processEvents()
            time.sleep(0.05)

    def Preprocessing_click(self):
        temp = []
        self.get_current_path()
        self.Play_video = 2
        self.textBrowser_image.append('Preprocessing')
        self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        # 获取一个视频并打开
        self.textBrowser_image.append('获取一个视频并打开')
        self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        cap = cv2.VideoCapture(self.current_path + '/Q2_Image/opticalFlow.mp4')
        if cap.isOpened():  # VideoCaputre对象是否成功打开
            # print('已经打开了视频文件')
            self.textBrowser_image.append('已经打开了视频文件')
            self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
            QApplication.processEvents()
            fps = cap.get(cv2.CAP_PROP_FPS)  # 返回视频的fps--帧率
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 返回视频的宽
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 返回视频的高
            # print('fps:', fps, 'width:', width, 'height:', height)
            self.textBrowser_image.append('fps:' + str(fps) + ' width:' + str(width) + ' height:' + str(height))
            self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
            QApplication.processEvents()
            i = 0
            ret, frame = cap.read()  # 读取一帧视频
            self.textBrowser_image.append('Draw detected blobs as red circles')
            self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
            QApplication.processEvents()
            while ret:
                # ret 读取了数据就返回True,没有读取数据(已到尾部)就返回False
                # frame 返回读取的视频数据--一帧数据
                # Setup SimpleBlobDetector parameters.
                params = cv2.SimpleBlobDetector_Params()
                # params.filterByColor = True    # 斑点颜色的限制变量
                # params.blobColor = 190   # 表示只提取黑色斑点；如果该变量为255，表示只提取白色斑点
                # Change thresholds
                params.minThreshold = 80
                params.maxThreshold = 180
                # Filter by Area.
                params.filterByArea = True
                params.minArea = 20
                params.maxArea = 80
                # Filter by Circularity
                params.filterByCircularity = True
                params.minCircularity = 0.8  # 圆的类圆性
                # Filter by Convexity
                params.filterByConvexity = True
                params.minConvexity = 0.87
                # Filter by Inertia
                params.filterByInertia = True
                params.minInertiaRatio = 0.53
                # Create a detector with the parameters
                detector = cv2.SimpleBlobDetector_create(params)
                # Detect blobs.
                keypoints = detector.detect(frame)
                tmp_kp_sort1 = sorted(keypoints, key=lambda x: x.size, reverse=True)[0:7]
                k = 0
                for f in tmp_kp_sort1:
                    temp.append(int(f.pt[0]))
                    temp.append(int(f.pt[1]))
                    k += 1
                # Draw detected blobs as red circles.
                # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
                # the size of the circle corresponds to the size of blob np.array([])
                im_with_keypoints = cv2.line(frame, (temp[0] - 5, temp[1]), (temp[0] + 5, temp[1]), (0, 0, 255), 1)
                im_with_keypoints = cv2.line(im_with_keypoints, (temp[0], temp[1] - 5), (temp[0], temp[1] + 5),
                                             (0, 0, 255), 1)
                im_with_keypoints = cv2.rectangle(im_with_keypoints, (temp[0] - 5, temp[1] - 5),
                                                  (temp[0] + 5, temp[1] + 5), (0, 0, 255), 1)
                for j in range(1, k):
                    im_with_keypoints = cv2.line(im_with_keypoints, (temp[j * 2] - 5, temp[j * 2 + 1]),
                                                 (temp[j * 2] + 5, temp[j * 2 + 1]), (0, 0, 255), 1)
                    im_with_keypoints = cv2.line(im_with_keypoints, (temp[j * 2], temp[j * 2 + 1] - 5),
                                                 (temp[j * 2], temp[j * 2 + 1] + 5), (0, 0, 255), 1)
                    im_with_keypoints = cv2.rectangle(im_with_keypoints, (temp[j * 2] - 5, temp[j * 2 + 1] - 5),
                                                      (temp[j * 2] + 5, temp[j * 2 + 1] + 5), (0, 0, 255), 1)
                temp = []
                file_name = self.current_path + '/Q2_Image/result.jpg'
                image = cv2.resize(im_with_keypoints, (450, 360))
                cv2.imwrite(file_name, image)
                ret = False
        else:
            # print('视频文件打开失败')
            self.textBrowser_image.append('视频文件打开失败')
            self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
            QApplication.processEvents()
        self.textBrowser_image.append('end')
        self.textBrowser_image_4.append(' ')
        self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        pix = QPixmap(self.current_path + '/Q2_Image/result.jpg')
        self.label_inputimage.setPixmap(pix)
        QApplication.processEvents()

    def Perspective_Transform_click(self):
        global width, height, frame, i
        self.get_current_path()
        self.Play_video = 3
        img_src = cv2.imread(self.current_path + '/Q3_Image/rl.jpg')
        image = cv2.resize(img_src, (430, 290))
        cv2.imwrite(self.current_path + "/Q3_Image/img_src.png", image)
        pix = QPixmap(self.current_path + "/Q3_Image/img_src.png")
        self.label_inputimage_6.setPixmap(pix)
        QApplication.processEvents()
        # 获取一个视频并打开
        self.textBrowser_image_4.append('获取一个视频并打开')
        self.textBrowser_image_4.moveCursor(self.textBrowser_image_4.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        cap = cv2.VideoCapture(self.current_path + '/Q3_Image/test4perspective.mp4')
        if cap.isOpened():  # VideoCaputre对象是否成功打开
            # print('已经打开了视频文件')
            self.textBrowser_image_4.append('已经打开视频文件')
            self.textBrowser_image_4.moveCursor(self.textBrowser_image_4.textCursor().End)  # 文本框顯示到底部
            QApplication.processEvents()
            fps = cap.get(cv2.CAP_PROP_FPS)  # 返回视频的fps--帧率
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 返回视频的宽
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 返回视频的高
            # print('fps:', fps, 'width:', width, 'height:', height)
            self.textBrowser_image_4.append('fps:' + str(fps) + '  width:' + str(width) + '  height:' + str(height))
            self.textBrowser_image_4.moveCursor(self.textBrowser_image_4.textCursor().End)  # 文本框顯示到底部
            QApplication.processEvents()
            i = 0
            ret, frame = cap.read()  # 读取一帧视频
            while ret:
                file_name = self.current_path + '/Q3_Image/result1/' + str(i) + '.jpg'
                cv2.imwrite(file_name, frame)
                i += 1
                ret, frame = cap.read()
        else:
            self.textBrowser_image_4.append('视频文件打开失败')
            self.textBrowser_image_4.moveCursor(self.textBrowser_image_4.textCursor().End)  # 文本框顯示到底部
            QApplication.processEvents()
        k = 0
        self.textBrowser_image_4.append('透视变换......')
        self.textBrowser_image_4.moveCursor(self.textBrowser_image_4.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        for j in range(0, i):
            time1 = time.time()
            imgorg = cv2.imread(self.current_path + '/Q3_Image/result1/' + str(j) + '.jpg')
            img = cv2.cvtColor(imgorg, cv2.COLOR_BGR2GRAY)
            dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
            parameters = cv2.aruco.DetectorParameters_create()
            markerCorners, markerIds, rejectCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)
            if markerIds.size != 4:
                j += 1
                continue
            index = np.squeeze(np.where(markerIds == 25))
            refPt1 = np.squeeze(np.array(markerCorners[index[0]], dtype=int))[1]
            index = np.squeeze(np.where(markerIds == 33))
            refPt2 = np.squeeze(np.array(markerCorners[index[0]], dtype=int))[2]
            distance = np.linalg.norm(refPt1 - refPt2)
            scalingFac = 0.02
            pts_dst = [[refPt1[0] - round(scalingFac * distance), refPt1[1] - round(scalingFac * distance)]]
            pts_dst = pts_dst + [[refPt2[0] + round(scalingFac * distance), refPt2[1] - round(scalingFac * distance)]]
            index = np.squeeze(np.where(markerIds == 30))
            refPt3 = np.squeeze(np.array(markerCorners[index[0]], dtype=int))[0]
            pts_dst = pts_dst + [[refPt3[0] + round(scalingFac * distance), refPt3[1] + round(scalingFac * distance)]]
            index = np.squeeze(np.where(markerIds == 23))
            refPt4 = np.squeeze(np.array(markerCorners[index[0]], dtype=int))[0]
            pts_dst = pts_dst + [[refPt4[0] - round(scalingFac * distance), refPt4[1] + round(scalingFac * distance)]]
            pts_src = [[0, 0], [img_src.shape[1], 0], [img_src.shape[1], img_src.shape[0]], [0, img_src.shape[0]]]
            retval, mask = cv2.findHomography(np.array(pts_src, dtype=np.float32), np.array(pts_dst, dtype=np.float32),
                                              cv2.RANSAC, 5.0)
            # 通过下图中的透视变换函数，输入两种数组，并返回M矩阵——扭转矩阵
            # 将扭转矩阵M输入，进行的原来图片的变换，其中img代表的是原图像，M代表的是扭转矩阵，
            # img_size代表的是转变之后的尺寸（可以设置为相同尺寸），inter_linear代表的是线性内插
            dst = cv2.warpPerspective(src=img_src, M=retval, dsize=(width, height), dst=imgorg,
                                      flags=cv2.WARP_FILL_OUTLIERS, borderMode=cv2.BORDER_TRANSPARENT,
                                      borderValue=None)
            # 纵向连接 image = np.vstack((gray1, gray2))
            # 横向连接
            imgorg_temp = cv2.imread(self.current_path + '/Q3_Image/result1/' + str(j) + '.jpg')
            image = np.concatenate((imgorg_temp, dst), axis=1)
            image = cv2.resize(image, (960, 450))
            file_name = self.current_path + '/Q3_Image/result/' + str(k) + '.jpg'
            cv2.imwrite(file_name, image)
            j += 1
            k += 1
        self.textBrowser_image_4.append('正在生成视频......')
        self.textBrowser_image_4.moveCursor(self.textBrowser_image_4.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        fsp = 30
        fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')  # 对应格式 mp4
        n = 0
        video_path = self.current_path + '/Q3_Image/result.mp4'.format(n)
        img_path = self.current_path + '/Q3_Image/result/'.format(n)
        list_image = os.listdir(self.current_path + '/Q3_Image/result/'.format(n))
        list_image.sort()
        list_image = [os.path.join(img_path, x) for x in list_image]
        width = cv2.imread(list_image[0]).shape[1]
        height = cv2.imread(list_image[0]).shape[0]
        video_out = cv2.VideoWriter(video_path, fourcc, fsp, (width, height))
        count = 1
        for i in range(len(list_image)):
            imgs = cv2.imread(self.current_path + '/Q3_Image/result/' + str(i) + '.jpg')
            video_out.write(imgs)
            count += 1
        video_out.release()
        self.textBrowser_image_4.append('end')
        self.textBrowser_image_4.append(' ')
        self.textBrowser_image_4.moveCursor(self.textBrowser_image_4.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        for i in range(len(list_image)):
            pix = QPixmap(self.current_path + "/Q3_Image/result/" + str(i) + ".jpg")
            self.label_inputimage_4.setPixmap(pix)
            QApplication.processEvents()
            time.sleep(0.05)

    def Image_Reconstruction_click(self):
        self.get_current_path()
        self.textBrowser_image_3.append('Image Reconstruction')
        self.textBrowser_image_3.moveCursor(self.textBrowser_image_3.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        for i in range(1, 35):
            imgorig = cv2.imread(self.current_path + '/Q4_Image/' + str(i) + '.jpg')
            img = imgorig
            # Load the image
            blue, green, red = cv2.split(imgorig)
            for j in range(3):
                if j == 0:
                    img = blue
                if j == 1:
                    img = green
                if j == 2:
                    img = red
                # Calculating the mean columnwise
                M = np.mean(img.T, axis=1)
                # Sustracting the mean columnwise
                C = img - M
                # Calculating the covariance matrix
                V = np.cov(C.T)
                # Computing the eigenvalues and eigenvectors of covarince matrix
                values, vectors = np.linalg.eig(V)
                p = np.size(vectors, axis=1)
                # Sorting the eigen values in ascending order
                idx = np.argsort(values)
                idx = idx[::-1]
                # Sorting eigen vectors
                vectors = vectors[:, idx]
                values = values[idx]
                # PCs used for reconstruction (can be varied)
                num_PC = 80
                # Cutting the PCs
                if num_PC < p or num_PC > 0:
                    vectors = vectors[:, range(num_PC)]
                # Reconstructing the image with PCs
                score = np.dot(vectors.T, C)
                Compressed_img = np.dot(vectors, score) + M
                constructed_img = np.uint8(np.absolute(Compressed_img))
                if j == 0:
                    blue = constructed_img
                if j == 1:
                    green = constructed_img
                if j == 2:
                    red = constructed_img
                j += 1
            # Show reconstructed image
            # show the result
            constructed_img = (np.dstack((red, green, blue))).astype(np.uint8)
            file_name = self.current_path + '/Q4_Image/result/' + str(i) + '.jpg'
            cv2.imwrite(file_name, constructed_img)
            i += 1
        self.textBrowser_image_3.append('Show Image Reconstruction')
        self.textBrowser_image_3.moveCursor(self.textBrowser_image_3.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        _, axs = plt.subplots(nrows=4, ncols=17, figsize=(500, 500))
        fig = plt.gcf()
        fig.set_size_inches(12, 8)
        fig.subplots_adjust(wspace=0.1, hspace=0.1)
        for i in range(1, 3):
            for j in range(0, 17):
                if i == 1:
                    img1 = plt.imread(self.current_path + '/Q4_Image/' + str(j + 1) + '.jpg')
                    img2 = plt.imread(self.current_path + '/Q4_Image/result/' + str(j + 1) + '.jpg')
                    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
                    axs[i - 1, j].imshow(img1)  # plot the data.numpy()float()
                    axs[i - 1][j].axes.get_xaxis().set_visible(False)
                    axs[i - 1][j].axes.get_yaxis().set_visible(False)
                    axs[i, j].imshow(img2)  # plot the data.numpy()float()
                    axs[i][j].axes.get_xaxis().set_visible(False)
                    axs[i][j].axes.get_yaxis().set_visible(False)
                if i == 2:
                    img1 = plt.imread(self.current_path + '/Q4_Image/' + str(j + 18) + '.jpg')
                    img2 = plt.imread(self.current_path + '/Q4_Image/result/' + str(j + 18) + '.jpg')
                    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
                    axs[i, j].imshow(img1)  # plot the data.numpy()float()
                    axs[i][j].axes.get_xaxis().set_visible(False)
                    axs[i][j].axes.get_yaxis().set_visible(False)
                    axs[i + 1, j].imshow(img2)  # plot the data.numpy()float()
                    axs[i + 1][j].axes.get_xaxis().set_visible(False)
                    axs[i + 1][j].axes.get_yaxis().set_visible(False)
        axs[0][0].set_yticks([])
        axs[0][0].axes.get_yaxis().set_visible(True)
        axs[0, 0].set(title=" ", xlabel=" ", ylabel="Original")  # 设置标题
        axs[1][0].set_yticks([])
        axs[1][0].axes.get_yaxis().set_visible(True)
        axs[1, 0].set(title=" ", xlabel=" ", ylabel="Reconstruction")  # 设置标题
        axs[2][0].set_yticks([])
        axs[2][0].axes.get_yaxis().set_visible(True)
        axs[2, 0].set(title=" ", xlabel=" ", ylabel="Original")  # 设置标题
        axs[3][0].set_yticks([])
        axs[3][0].axes.get_yaxis().set_visible(True)
        axs[3, 0].set(title=" ", xlabel=" ", ylabel="Reconstruction")  # 设置标题
        plt.savefig(self.current_path + "/Q4_Image/train.png")
        imgorig = cv2.imread(self.current_path + '/Q4_Image/train.png')
        print(imgorig.shape)
        cropped = imgorig[120:700, 120:1100]  # 裁剪坐标为[y0:y1, x0:x1]
        image = cv2.resize(cropped, (900, 600))
        cv2.imwrite(self.current_path + '/Q4_Image/train.png', image)
        pix = QPixmap(self.current_path + "/Q4_Image/train.png")
        self.label_inputimage_5.setPixmap(pix)
        QApplication.processEvents()
        self.textBrowser_image_3.append('end')
        self.textBrowser_image_3.append(' ')
        self.textBrowser_image_3.moveCursor(self.textBrowser_image_3.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()

    def Reconstruction_Error_click(self):
        err = []
        self.get_current_path()
        for i in range(1, 35):
            imgorig = plt.imread(self.current_path + '/Q4_Image/' + str(i) + '.jpg')
            reconsturct_image = plt.imread(self.current_path + '/Q4_Image/result/' + str(i) + '.jpg')
            reconsturct_image = cv2.cvtColor(reconsturct_image, cv2.COLOR_RGB2BGR)
            err.append(np.sum(np.abs(np.subtract(reconsturct_image, imgorig))))
        self.textBrowser_image_3.append('err: ')
        self.textBrowser_image_3.append(str(err))
        self.textBrowser_image_3.moveCursor(self.textBrowser_image_3.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()

    def Training_Images_click(self):
        self.get_current_path()
        img = cv2.imread(self.current_path + '/Q5/train.png')
        image = cv2.resize(img, (820, 270))
        cv2.imwrite(self.current_path + '/Q5/train.png', image)
        pix = QPixmap(self.current_path + "/Q5/train.png")
        self.label_inputimage_9.setPixmap(pix)
        QApplication.processEvents()

    def TensorBoard_click(self):
        self.get_current_path()
        img = cv2.imread(self.current_path + '/Q5/TensorBoard.png')
        image = cv2.resize(img, (820, 430))
        cv2.imwrite(self.current_path + '/Q5/TensorBoard.png', image)
        pix = QPixmap(self.current_path + "/Q5/TensorBoard.png")
        self.label_inputimage_19.setPixmap(pix)
        QApplication.processEvents()

    def Test_openFile_image_click(self):
        self.get_current_path()
        get_filename_path, ok = QFileDialog.getOpenFileName(self, "选取单个文件", self.current_path + "/",
                                                            "All Files (*);;Text Files (*.png)")  # ,
        i = 5
        while str(get_filename_path[-i]) != '/':
            i += 1
        self.image_path = get_filename_path[-(i-1):]
        img = cv2.imread(self.current_path + '/Q5/test/' + self.image_path)
        model = Sequential()
        model.add(ResNet50(include_top=False, pooling='max', weights='imagenet'))
        model.add(Dense(1, activation='sigmoid'))
        model.layers[0].trainable = False
        # load the best weights saved using checkpointer
        model.load_weights('/home/wolf/zjb/Homework/Hw2/Q5/resnet50_best.h5')
        x = cv2.resize(img, (224, 224))
        img_array = img_to_array(x)
        img_array = np.expand_dims(img_array, axis=0)
        out = model.predict(img_array)
        out = 'Dog' if float(out) > 0.5 else 'Cat'
        plt.figure()  # 图像窗口名称
        plt.imshow(img)
        plt.axis('on')  # 关掉坐标轴为 off
        plt.title(out)  # 图像题目
        plt.savefig(self.current_path + "/Q5/Test.png")
        plt.show()
        img = cv2.imread(self.current_path + '/Q5/Test.png')
        image = cv2.resize(img, (420, 300))
        cv2.imwrite(self.current_path + '/Q5/Test.png', image)
        pix = QPixmap(self.current_path + "/Q5/Test.png")
        self.label_inputimage_18.setPixmap(pix)
        QApplication.processEvents()

    # 显示高度
    def autolabel(self, rects):
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width() / 2. - 0.2, 1.03 * height, '%s' % height)

    def Model_ResNet50_click(self):
        self.get_current_path()
        name_list = ['Before Random-Erasing', 'After Random-Erasing']
        num_list = [98.92,  99.57]
        self.autolabel(plt.bar(range(len(num_list)), num_list, color='rgb', tick_label=name_list))
        plt.savefig(self.current_path + "/Q5/Random_Erasing.png")
        plt.show()
        img = cv2.imread(self.current_path + '/Q5/Random_Erasing.png')
        image = cv2.resize(img, (420, 330))
        cv2.imwrite(self.current_path + '/Q5/Random_Erasing.png', image)
        pix = QPixmap(self.current_path + "/Q5/Random_Erasing.png")
        self.label_inputimage_20.setPixmap(pix)
        QApplication.processEvents()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
