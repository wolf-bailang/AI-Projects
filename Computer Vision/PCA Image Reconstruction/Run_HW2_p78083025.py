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
from tqdm import tqdm
from PIL import Image
import time
import matplotlib.pyplot as plt
from cv2 import Stitcher
from sklearn.decomposition import PCA

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
        self.pushButton_image_1.clicked.connect(self.Background_Subtraction_page)
        self.pushButton_image_2.clicked.connect(self.Optical_Flow_page)
        self.pushButton_image_3.clicked.connect(self.Perspective_Transform_page)
        self.pushButton_image_4.clicked.connect(self.PCA_page)
        self.pushButton_image_6.clicked.connect(self.Preprocessing_click)  # 1.1 Find Corners
        self.pushButton_image_7.clicked.connect(self.Video_tracking_click)  # 1.2 Find Intrinsic
        self.pushButton_image_8.clicked.connect(self.Play_video_click)  # 1.3 Find Extrinsic
        # self.comboBox.currentIndexChanged.connect(self.get_comboBox) Perspective_Transform
        # self.pushButton_image_9.clicked.connect(self.Stereo_Disparity_Map_click)  # 1.4 Find Distortion
        self.pushButton_image_11.clicked.connect(self.Background_Subtraction_click)
        self.pushButton_image_12.clicked.connect(self.Perspective_Transform_click)
        self.pushButton_image_14.clicked.connect(self.Image_Reconstruction_click)
        self.pushButton_image_15.clicked.connect(self.Reconstruction_Error_click)
        self.pushButton_image_17.clicked.connect(self.Play_video_click)
        self.pushButton_image_18.clicked.connect(self.Play_video_click)
        self.flag = 1
        self.current_path = ' '
        self.image_path = ' '
        # self.comboBox.addItems([' ', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'])
        self.Play_video = 0

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

    '''
    def get_comboBox(self):
        self.comboBox.setCurrentText(self.comboBox.currentText())
        self.image_path = self.comboBox.currentText()
    '''

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
        Simple_Gaussian_mean = np.zeros((176, 320), dtype=np.float)  #
        Simple_Gaussian_standard_deviation = np.zeros((176, 320), dtype=np.float)
        self.get_current_path()
        self.Play_video = 1
        self.textBrowser_image_2.append('获取一个视频')
        # self.textBrowser_image_2.append(' ')
        self.textBrowser_image_2.moveCursor(self.textBrowser_image_2.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        # 获取一个视频并打开
        cap = cv2.VideoCapture(self.current_path + '/Q1_Image/bgSub.mp4')
        if cap.isOpened():  # VideoCaputre对象是否成功打开
            self.textBrowser_image_2.append('已经打开视频文件')
            # self.textBrowser_image_2.append(' ')
            self.textBrowser_image_2.moveCursor(self.textBrowser_image_2.textCursor().End)  # 文本框顯示到底部
            QApplication.processEvents()
            fps = cap.get(cv2.CAP_PROP_FPS)  # 返回视频的fps--帧率
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 返回视频的宽
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 返回视频的高
            # print('fps:', fps, 'width:', width, 'height:', height)
            self.textBrowser_image_2.append('fps=' + str(fps) + ' width=' + str(width) + ' height=' + str(height))
            # self.textBrowser_image_2.append('　')
            self.textBrowser_image_2.moveCursor(self.textBrowser_image_2.textCursor().End)  # 文本框顯示到底部
            QApplication.processEvents()
            i = 0
            while 1:
                if i == 50:
                    self.textBrowser_image_2.append('截取视频的前50帧图像，保存结束')
                    # self.textBrowser_image_2.append('　')
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
            # self.textBrowser_image_2.append('　')
            self.textBrowser_image_2.moveCursor(self.textBrowser_image_2.textCursor().End)  # 文本框顯示到底部
            QApplication.processEvents()
        self.textBrowser_image_2.append('正在建立高斯模型.....')
        # self.textBrowser_image_2.append('　')
        self.textBrowser_image_2.moveCursor(self.textBrowser_image_2.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        # print('Simple_Gaussian_sum: ' + str(Simple_Gaussian_mean))
        for row in range(176):  # 遍历高
            for col in range(320):  # 遍历宽
                Simple_Gaussian_mean[row, col] = Simple_Gaussian_mean[row, col] / 50.0
        # print('Simple_Gaussian_mean: ' + str(Simple_Gaussian_mean))
        i = 1
        while i <= 50:
            img = cv2.imread(self.current_path + '/Q1_Image/frame/' + str(i) + '.jpg')
            # print('gray: ' + str(gray))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for row in range(176):  # 遍历高
                for col in range(320):  # 遍历宽
                    # print('gray ' + str(gray[row, col]))
                    # print('Simple_Gaussian_mean ' + str(Simple_Gaussian_mean[row, col]))
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
        """"""
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
            # file_name = self.current_path + '/Q1_Image/frame/' + str(i) + '.jpg'
            # cv2.imwrite(file_name, frame)
            # file_name = self.current_path + '/Q1_Image/result/' + str(i) + '.jpg'
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # cv2.imwrite(file_name, gray)
            # print(str(frame.shape))
            while ret:
                i = i + 1
                ret, frame = cap.read()  # 读取一帧视频
                if ret:
                    k = 0
                else:
                    break
                # file_name = self.current_path + '/Q1_Image/frame/' + str(i) + '.jpg'
                # cv2.imwrite(file_name, frame)
                if i < 50:
                    # print('保存了视频的前15帧图像，保存结束')
                    # file_name = self.current_path + '/Q1_Image/result/' + str(i) + '.jpg'
                    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # cv2.imwrite(file_name, gray)
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
                    # img3 = np.hstack([frame, gray])
                    # file_name = self.current_path + '/Q1_Image/result/c_' + str(i) + '.jpg'
                    # cv2.imwrite(file_name, img3)
                    # print(str(gray.shape))
        self.textBrowser_image_2.append('生成视频文件......')
        # self.textBrowser_image_2.append('　')
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
        # print(list_image[0])
        width = cv2.imread(list_image[0]).shape[1]
        height = cv2.imread(list_image[0]).shape[0]
        video_out = cv2.VideoWriter(video_path, fourcc, fsp, (width, height))
        # print(len(list_image))
        count = 1
        for i in range(len(list_image)):
            frame = cv2.imread(self.current_path + '/Q1_Image/result/' + str(i) + '.jpg')
            video_out.write(frame)
            count += 1
        # print('cout = ', count)
        # self.textBrowser_image_2.append('總幀數: ' + str(count))
        # self.textBrowser_image_2.append('　')
        # self.textBrowser_image_2.moveCursor(self.textBrowser_image_2.textCursor().End)  # 文本框顯示到底部
        # QApplication.processEvents()
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
        # file_name = self.current_path + '/Q1_Image/frame/' + str(i) + '.jpg'
        # cv2.imwrite(file_name, frame)
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.textBrowser_image.append('Setup SimpleBlobDetector parameters')
        self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
        # params.filterByColor = True    # 斑点颜色的限制变量
        # params.blobColor = 190   # 表示只提取黑色斑点；如果该变量为255，表示只提取白色斑点
        # Change thresholds
        params.minThreshold = 80  # 80
        params.maxThreshold = 180  # 180
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 20
        params.maxArea = 80
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.8  # 0.8 圆的类圆性
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.87
        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.53  # 0.53
        # Create a detector with the parameters
        # ver = (cv2.__version__).split('.')
        # if int(ver[0]) < 3:
        # detector = cv2.SimpleBlobDetector(params)
        # else:
        detector = cv2.SimpleBlobDetector_create(params)
        # Detect blobs.
        keypoints = detector.detect(old_frame)
        tmp_kp_sort1 = sorted(keypoints, key=lambda x: x.size, reverse=True)[0:7]
        k = 0
        for f in tmp_kp_sort1:
            a = []
            a.append([int(f.pt[0]), int(f.pt[1])])
            # print('a= ' + str(a))
            # temp.append(int(f.pt[0]))
            # temp.append(int(f.pt[1]))
            temp.append(np.array(a, dtype=np.float32))
            # print(str(f.pt[0]) + '  ' + str(f.pt[1]))
            # print(str(temp[0]) + '  ' + str(temp[1]))
            p0[k] = np.array(a, dtype=np.float32)
            k += 1
        # print('tmp_kp_sort1= ' + str(tmp_kp_sort1))
        # print('temp= ' + str(temp))
        # p0 = np.array(temp, dtype=np.float)
        '''
        p0= [[[467.  26.]] [[375. 112.]] [[536.  83.]] [[318.  15.]] [[509.  83.]] [[351. 140.]] [[433.  36.]]]
        p0= [[[117.48451996  72.01285553]] [[118.72344208 169.06144714]] [[321.96606445  20.77165031]] 
             [[135.06681824 240.90565491]] [[111.36567688  95.52036285]] [[129.55197144 259.88110352]]
             [[241.8878479   89.54856873]]]
        '''
        # p0 = [[[117., 72.]], [[118., 169.]], [[321., 20.]], [[135., 240.]], [[111., 95.]], [[129., 259.]], [[241., 89.]]]
        # p0 = np.array(p0, dtype=np.float)
        # params for ShiTomasi corner detection构建角点检测所需参数
        feature_params = dict(maxCorners=7, qualityLevel=0.3, minDistance=7, blockSize=5)
        # Parameters for lucas kanade optical flow lucas kanade参数
        lk_params = dict(winSize=(11, 11), maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        color = (0, 0, 255)  # np.random.randint(0, 255, (100, 3))
        # Take first frame and find corners in it拿到第一帧图像并灰度化作为前一帧图片
        # ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        # 返回所有检测特征点，需要输入图片，角点的最大数量，品质因子，minDistance=7如果这个角点里有比这个强的就不要这个弱的
        # p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        # print('p0= ' + str(p0))
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
            # p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, tmp_kp_sort1, None, **lk_params)
            # Select good points读取运动了的角点st == 1表示检测到的运动物体，即v和u表示为0
            good_new = p1[st == 1]
            # good_old = tmp_kp_sort1[st == 1]
            # print('tmp_kp_sort1[st == 1]= ' + str(tmp_kp_sort1[st == 1]))
            good_old = p0[st == 1]
            # print('p0[st == 1]= ' + str(p0[st == 1]))
            # draw the tracks绘制轨迹
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color, 2)  # [i].tolist()
                frame = cv2.circle(frame, (int(a), int(b)), 5, color, -1)  # [i].tolist()
            # 将两个图片进行结合，并进行图片展示
            result_image = cv2.add(frame, mask)
            image = cv2.resize(result_image, (770, 560))
            file_name = self.current_path + '/Q2_Image/result/' + str(j) + '.jpg'
            cv2.imwrite(file_name, image)
            # cv2.imshow('Lucas-Kanade Optical Flow', result_image)
            # keyboard = cv2.waitKey(30) & 0xff
            # if keyboard == 'q' or keyboard == 27:
            #    break
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
        # print(list_image[0])
        width = cv2.imread(list_image[0]).shape[1]
        height = cv2.imread(list_image[0]).shape[0]
        video_out = cv2.VideoWriter(video_path, fourcc, fsp, (width, height))
        # print(len(list_image))
        count = 1
        for i in range(len(list_image)):
            frame = cv2.imread(self.current_path + '/Q2_Image/result/' + str(i) + '.jpg')
            video_out.write(frame)
            count += 1
        # print('cout', count)
        video_out.release()
        # print('end')
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
            # im_with_keypoints = frame
            self.textBrowser_image.append('Draw detected blobs as red circles')
            self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
            QApplication.processEvents()
            while ret:
                # ret 读取了数据就返回True,没有读取数据(已到尾部)就返回False
                # frame 返回读取的视频数据--一帧数据
                # file_name = self.current_path + '/Q1_Image/frame/' + str(i) + '.jpg'
                # cv2.imwrite(file_name, frame)
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Setup SimpleBlobDetector parameters.
                params = cv2.SimpleBlobDetector_Params()
                # params.filterByColor = True    # 斑点颜色的限制变量
                # params.blobColor = 190   # 表示只提取黑色斑点；如果该变量为255，表示只提取白色斑点
                # Change thresholds
                params.minThreshold = 80  # 80
                params.maxThreshold = 180  # 180
                # Filter by Area.
                params.filterByArea = True
                params.minArea = 20  # 20
                params.maxArea = 80  # 80
                # Filter by Circularity
                params.filterByCircularity = True
                params.minCircularity = 0.8  # 0.8 圆的类圆性
                # Filter by Convexity
                params.filterByConvexity = True
                params.minConvexity = 0.87  # 0.87
                # Filter by Inertia
                params.filterByInertia = True
                params.minInertiaRatio = 0.53  # 0.53
                # Create a detector with the parameters
                # ver = (cv2.__version__).split('.')
                # if int(ver[0]) < 3:
                # detector = cv2.SimpleBlobDetector(params)
                # else:
                detector = cv2.SimpleBlobDetector_create(params)
                # Detect blobs.
                keypoints = detector.detect(frame)
                tmp_kp_sort1 = sorted(keypoints, key=lambda x: x.size, reverse=True)[0:7]
                k = 0
                for f in tmp_kp_sort1:
                    temp.append(int(f.pt[0]))
                    temp.append(int(f.pt[1]))
                    k += 1
                    # print(str(f.pt[0]) + '  ' + str(f.pt[1]))
                    # print(str(temp[0]) + '  ' + str(temp[1]))
                # print('k= ' + str(k))
                # Draw detected blobs as red circles.
                # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
                # the size of the circle corresponds to the size of blob np.array([])
                im_with_keypoints = cv2.line(frame, (temp[0] - 5, temp[1]), (temp[0] + 5, temp[1]), (0, 0, 255), 1)
                im_with_keypoints = cv2.line(im_with_keypoints, (temp[0], temp[1] - 5), (temp[0], temp[1] + 5),
                                             (0, 0, 255), 1)
                im_with_keypoints = cv2.rectangle(im_with_keypoints, (temp[0] - 5, temp[1] - 5),
                                                  (temp[0] + 5, temp[1] + 5), (0, 0, 255), 1)
                for j in range(1, k):
                    # print('i= ' + str(i))
                    im_with_keypoints = cv2.line(im_with_keypoints, (temp[j * 2] - 5, temp[j * 2 + 1]),
                                                 (temp[j * 2] + 5, temp[j * 2 + 1]), (0, 0, 255), 1)
                    im_with_keypoints = cv2.line(im_with_keypoints, (temp[j * 2], temp[j * 2 + 1] - 5),
                                                 (temp[j * 2], temp[j * 2 + 1] + 5), (0, 0, 255), 1)
                    im_with_keypoints = cv2.rectangle(im_with_keypoints, (temp[j * 2] - 5, temp[j * 2 + 1] - 5),
                                                      (temp[j * 2] + 5, temp[j * 2 + 1] + 5), (0, 0, 255), 1)
                '''
                im_with_keypoints = cv2.rectangle(im_with_keypoints, (temp[4]-5, temp[5]-5), (temp[4]+5, temp[5]+5), (0, 0, 255), -1)
                im_with_keypoints = cv2.rectangle(im_with_keypoints, (temp[6]-5, temp[7]-5), (temp[6]+5, temp[7]+5), (0, 0, 255), -1)
                im_with_keypoints = cv2.rectangle(im_with_keypoints, (temp[8]-5, temp[9]-5), (temp[8]+5, temp[9]+5), (0, 0, 255), -1)
                im_with_keypoints = cv2.rectangle(im_with_keypoints, (temp[10]-5, temp[11]-5), (temp[10]+5, temp[11]+5), (0, 0, 255), -1)
                im_with_keypoints = cv2.rectangle(im_with_keypoints, (temp[12]-5, temp[13]-5), (temp[12]+5, temp[13]+5), (0, 0, 255), -1)
                # print(tmp_kp_sort1.)
                # Show blobs
                im_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, (0, 0, 255),
                                                      cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                '''
                temp = []
                # im_with_keypoints = cv2.cvtColor(im_with_keypoints, cv2.COLOR_GRAY2RGB)
                file_name = self.current_path + '/Q2_Image/result.jpg'
                image = cv2.resize(im_with_keypoints, (450, 360))
                cv2.imwrite(file_name, image)
                # cv2.imshow("Keypoints", image)
                # cv2.waitKey(0)
                ret = False
                # ret, frame = cap.read()  # 读取一帧视频
                # i = i + 1
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
        # print('end')

    def Augmented_Reality_draw(self, img, corners, img_pts):
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
                # ret = False
                i += 1
                # print('i= ' + str(i))
                ret, frame = cap.read()
        else:
            # print('')
            self.textBrowser_image_4.append('视频文件打开失败')
            self.textBrowser_image_4.moveCursor(self.textBrowser_image_4.textCursor().End)  # 文本框顯示到底部
            QApplication.processEvents()
        # print(str(i))
        k = 0
        self.textBrowser_image_4.append('透视变换......')
        self.textBrowser_image_4.moveCursor(self.textBrowser_image_4.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        for j in range(0, i):
            time1 = time.time()
            imgorg = cv2.imread(self.current_path + '/Q3_Image/result1/' + str(j) + '.jpg')
            img = cv2.cvtColor(imgorg, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("frame", img)
            # print('time1= ' + str(time.time() - time1))
            # print('img' + str(j) + ' = ' + str(img))
            dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
            parameters = cv2.aruco.DetectorParameters_create()
            markerCorners, markerIds, rejectCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)
            # print('markerCorners:', markerCorners)
            # print('j = ' + str(j))
            # print('markerIds:', markerIds)
            if markerIds.size != 4:
                '''
                image = np.concatenate([imgorg, imgorg], axis=1)
                file_name = self.current_path + '/Q3_Image/result/' + str(j) + '.jpg'
                cv2.imwrite(file_name, image)
                '''
                j += 1
                continue
            # print('rejectCandidates:', rejectCandidates)
            index = np.squeeze(np.where(markerIds == 25))
            # print('index:', index)
            # print('index:', index[0])
            refPt1 = np.squeeze(np.array(markerCorners[index[0]], dtype=int))[1]
            index = np.squeeze(np.where(markerIds == 33))
            # print('index:', index)
            # print('index:', index[0])
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
            # print('pts_dst:', pts_dst)
            # print('pts_src:', pts_src)
            retval, mask = cv2.findHomography(np.array(pts_src, dtype=np.float32), np.array(pts_dst, dtype=np.float32),
                                              cv2.RANSAC, 5.0)
            # 通过下图中的透视变换函数，输入两种数组，并返回M矩阵——扭转矩阵
            # target = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            # 将扭转矩阵M输入，进行的原来图片的变换，其中img代表的是原图像，M代表的是扭转矩阵，
            # img_size代表的是转变之后的尺寸（可以设置为相同尺寸），inter_linear代表的是线性内插
            dst = cv2.warpPerspective(src=img_src, M=retval, dsize=(width, height), dst=imgorg,
                                      flags=cv2.WARP_FILL_OUTLIERS, borderMode=cv2.BORDER_TRANSPARENT,
                                      borderValue=None)  # , frameflags=
            # cv2.imshow("dst",  frame)
            # cv2.waitKey(0)

            # stitcher = cv2.createStitcher(False)
            # stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA), 根据不同的OpenCV版本来调用
            # (_result, pano) = stitcher.stitch((imgorg, dst))
            # cv2.imshow('pano', pano)
            # cv2.waitKey(0)
            # gray1 = cv2.cvtColor(imgorg, cv2.COLOR_BGR2GRAY)
            # gray2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            # image = np.concatenate((imgorg_temp, dst))
            # 纵向连接 image = np.vstack((gray1, gray2))
            # 横向连接
            imgorg_temp = cv2.imread(self.current_path + '/Q3_Image/result1/' + str(j) + '.jpg')
            image = np.concatenate((imgorg_temp, dst), axis=1)
            # image = np.array(df) # dataframe to ndarray
            image = cv2.resize(image, (960, 450))
            file_name = self.current_path + '/Q3_Image/result/' + str(k) + '.jpg'
            cv2.imwrite(file_name, image)
            # ret = False
            j += 1
            k += 1
            # print('i= ' + str(i))
            # ret, frame = cap.read()
            # print('time2= ' + str(time.time() - time1))
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
        # print(list_image[0])
        width = cv2.imread(list_image[0]).shape[1]
        height = cv2.imread(list_image[0]).shape[0]
        video_out = cv2.VideoWriter(video_path, fourcc, fsp, (width, height))
        # print(len(list_image))
        count = 1
        for i in range(len(list_image)):
            imgs = cv2.imread(self.current_path + '/Q3_Image/result/' + str(i) + '.jpg')
            video_out.write(imgs)
            count += 1
        # print('cout', count)
        video_out.release()
        # print('end')
        self.textBrowser_image_4.append('end')
        self.textBrowser_image_4.append(' ')
        self.textBrowser_image_4.moveCursor(self.textBrowser_image_4.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        for i in range(len(list_image)):
            pix = QPixmap(self.current_path + "/Q3_Image/result/" + str(i) + ".jpg")
            self.label_inputimage_4.setPixmap(pix)
            QApplication.processEvents()
            time.sleep(0.05)

    # 数据中心化
    def Z_centered(self, dataMat, rows):
        # rows, cols = dataMat.shape
        meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
        meanVal = np.tile(meanVal, (rows, 1))
        newdata = dataMat - meanVal
        return newdata, meanVal

    # 协方差矩阵
    def Cov(self, dataMat, data, rows):
        meanVal = np.mean(data, 0)  # 压缩行，返回1*cols矩阵，对各列求均值
        meanVal = np.tile(meanVal, (rows, 1))  # 返回rows行的均值矩阵
        Z = dataMat - meanVal
        Zcov = (1 / (rows - 1)) * Z.T * Z
        return Zcov

    # 最小化降维造成的损失，确定k
    def Percentage2n(self, eigVals, percentage):
        sortArray = np.sort(eigVals)  # 升序
        sortArray = sortArray[-1::-1]  # 逆转，即降序
        arraySum = sum(sortArray)
        tmpSum = 0
        num = 0
        for i in sortArray:
            tmpSum += i
            num += 1
            if tmpSum >= arraySum * percentage:
                return num

    # 得到最大的k个特征值和特征向量
    def EigDV(self, covMat, p):
        D, V = np.linalg.eig(covMat)  # 得到特征值和特征向量
        k = self.Percentage2n(D, p)  # 确定k值
        print("保留99%信息，降维后的特征个数：" + str(k) + "\n")
        eigenvalue = np.argsort(D)
        K_eigenValue = eigenvalue[-1:-(k + 1):-1]
        K_eigenVector = V[:, K_eigenValue]
        return K_eigenValue, K_eigenVector

    # 得到降维后的数据
    def getlowDataMat(self, DataMat, K_eigenVector):
        return DataMat * K_eigenVector

    # 重构数据
    def Reconstruction(self, lowDataMat, K_eigenVector, meanVal):
        reconDataMat = lowDataMat * K_eigenVector.T + meanVal
        return reconDataMat

    # PCA算法
    def PCA(self, data, p, rows):
        dataMat = np.float32(np.mat(data))
        # 数据中心化
        dataMat, meanVal = self.Z_centered(dataMat, rows)
        # 计算协方差矩阵
        # covMat = Cov(dataMat)
        covMat = np.cov(dataMat)  # rows data,, rowvar=0
        # 得到最大的k个特征值和特征向量
        D, V = self.EigDV(covMat, p)
        # 得到降维后的数据
        lowDataMat = self.getlowDataMat(dataMat, V)
        # 重构数据
        reconDataMat = self.Reconstruction(lowDataMat, V, meanVal)
        return lowDataMat, reconDataMat

    def Image_Reconstruction_click(self):
        self.get_current_path()
        self.textBrowser_image_3.append('Image Reconstruction')
        self.textBrowser_image_3.moveCursor(self.textBrowser_image_3.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()
        for i in range(1, 35):
            imgorig = cv2.imread(self.current_path + '/Q4_Image/' + str(i) + '.jpg')
            img = imgorig
            # Load the image
            # = cv2.imread(imgpath, 0)
            # cv2.imshow("img", img)
            blue, green, red = cv2.split(imgorig)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
                # cv2.imshow("Compressed_img", Compressed_img)
                constructed_img = np.uint8(np.absolute(Compressed_img))
                if j == 0:
                    blue = constructed_img
                if j == 1:
                    green = constructed_img
                if j == 2:
                    red = constructed_img
                j += 1
            # Show reconstructed image
            # constructed_img = cv2.cvtColor(constructed_img, cv2.COLOR_GRAY2RGB)
            # constructed_img = cv2.cvtColor(constructed_img, cv2.COLOR_GRAY2BGR)
            # constructed_img = cv2.cvtColor(constructed_img, cv2.COLOR_BGR2RGB)
            # constructed_img = cv2.applyColorMap(constructed_img, cv2.COLORMAP_JET)
            # show the result
            # plt.figure()
            # plt.imshow(constructed_img)
            # plt.show()
            # cv2.imshow("Reconstructed Image", constructed_img)+34
            constructed_img = (np.dstack((red, green, blue))).astype(np.uint8)
            file_name = self.current_path + '/Q4_Image/result/' + str(i) + '.jpg'
            cv2.imwrite(file_name, constructed_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
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
        # plt.show()
        '''
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rows, cols = image.shape
        print("降维前的特征个数：" + str(cols) + "\n")
        print(image)
        print('----------------------------------------')
        cv2.imshow('image', image)
        Image, reconImage = self.PCA(image, 0.90, rows)
        Image = cv2.cvtColor(Image, cv2.COLOR_GRAY2BGR)
        cv2.imshow('Image', Image)
        reconImage = reconImage.astype(np.uint8)
        print(reconImage)
        reconImage = cv2.cvtColor(reconImage, cv2.COLOR_GRAY2BGR)
        cv2.imshow('reconImage', reconImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        imgorig = cv2.imread(self.current_path + '/Q4_Image/train.png')
        print(imgorig.shape)
        cropped = imgorig[120:700, 120:1100]  # 裁剪坐标为[y0:y1, x0:x1]
        image = cv2.resize(cropped, (900, 600))
        cv2.imwrite(self.current_path + '/Q4_Image/train.png', image)
        pix = QPixmap(self.current_path + "/Q4_Image/train.png")
        self.label_inputimage_5.setPixmap(pix)
        QApplication.processEvents()
        # print('end')
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
            # reconsturct_image = cv2.cvtColor(reconsturct_image, cv2.COLOR_BGR2GRAY)
            # reconsturct_image = cv2.cvtColor(reconsturct_image, cv2.COLOR_BGR2GRAY)
            err.append(np.sum(np.abs(np.subtract(reconsturct_image, imgorig))))
        # print(str(err))
        self.textBrowser_image_3.append('err: ')
        self.textBrowser_image_3.append(str(err))
        self.textBrowser_image_3.moveCursor(self.textBrowser_image_3.textCursor().End)  # 文本框顯示到底部
        QApplication.processEvents()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
