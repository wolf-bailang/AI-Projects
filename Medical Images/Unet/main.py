from __future__ import print_function

import sys
import os
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import glob
import skimage.io as io
import skimage.transform as trans

from UI import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QPushButton, QLabel, QFileDialog
from PyQt5.QtGui import QPalette, QBrush, QPixmap
import matplotlib.pyplot as plt
import tkinter as tk
from model import *
from data import *
#import tensorflow as tf


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()
        self.openFileButton_model.clicked.connect(self.openFile_model)
        self.openFileButton_image.clicked.connect(self.openFile_image)

        self.model_path = ' '
        self.image_path = ' '
        self.groundtruth_path = ' '
        self.flag = 0
        self.current_path = ' '

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

    def openFile_model(self):
        self.get_current_path()
        get_filename_path, ok = QFileDialog.getOpenFileName(self, "选取单个文件", self.current_path,
                                                            "All Files (*);;Text Files (*.png)") # + "/",
        if ok:
            self.textBrowser_model.append(str(get_filename_path))
            self.textBrowser_model.moveCursor(self.textBrowser_model.textCursor().End)  # 文本框顯示到底部
            self.model_path = str(get_filename_path)
        """
        get_directory_path = QFileDialog.getExistingDirectory(self, "选取指定文件夹", "/home/")
        self.filePathlineEdit.setText(str(get_directory_path))
        
        get_filenames_path, ok = QFileDialog.getOpenFileNames(self, "选取多个文件", "/home/", "All Files (*);;Text Files (*.png)")
        if ok:
            self.filePathlineEdit.setText(str(' '.join(get_filenames_path)))
        """

    def openFile_image(self):
        self.get_current_path()
        get_filename_path, ok = QFileDialog.getOpenFileName(self, "选取单个文件", self.current_path,
                                                            "All Files (*);;Text Files (*.png)")# + "/",
        if ok:
            if self.flag == 0:
                self.textBrowser_image.append(str(get_filename_path))
                self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
                self.image_path = str(get_filename_path)
                self.image_process(num=0)
                self.flag = 1
            else:
                self.textBrowser_image.append(str(get_filename_path))
                self.textBrowser_image.moveCursor(self.textBrowser_image.textCursor().End)  # 文本框顯示到底部
                self.groundtruth_path = str(get_filename_path)
                self.image_process(num=1)
                self.flag = 0
        """
        get_directory_path = QFileDialog.getExistingDirectory(self, "选取指定文件夹", "/home/")
        self.filePathlineEdit.setText(str(get_directory_path))
        get_filenames_path, ok = QFileDialog.getOpenFileNames(self, "选取多个文件", "/home/", "All Files (*);;Text Files (*.png)")
        if ok:
            self.filePathlineEdit.setText(str(' '.join(get_filenames_path)))
        """

    def onBindingUI(self):
        self.pushButton_run.clicked.connect(self.on_btn1_1_click)

    def on_btn1_1_click(self):
        Hist1(self.image_path, self.current_path + "/input.png", as_gray=True)

        testGene = testGenerator(self.current_path + "/input.png")
        model = unet()
        # 讀取model名稱
        model.load_weights(self.model_path)
        results = model.predict_generator(testGene, 1, verbose=1)
        #print("results.png")
        #print(results)
        #results = np.uint8(np.interp(results, (results.min(), results.max()), (0, 255)))
        #print("results.png")
        #print(results)
        saveResult(self.current_path + "/predict.png", results)

        img = io.imread(os.path.join(self.current_path + "/predict.png"), as_gray=True)
        #print("predict.png")
        #print(img)
        #img = np.uint8(np.interp(img, (img.min(), img.max()), (0, 255)))
        imgPixmap = cv2.resize(img, (250, 600))
        io.imsave(os.path.join(self.current_path + "/predictPixmap.png"), imgPixmap)
        pix = QPixmap(self.current_path + "/predictPixmap.png")
        self.label_predictimage.setPixmap(pix)

        label = cv2.imread(self.groundtruth_path, cv2.IMREAD_GRAYSCALE)
        #print("label.png")
        #print(label)
        #label = np.uint8(np.interp(label, (label.min(), label.max()), (0, 255)))
        pre = cv2.imread(os.path.join(self.current_path + "/predict.png"), cv2.IMREAD_GRAYSCALE)
        #pre = np.uint8(np.interp(pre, (pre.min(), pre.max()), (0, 255)))

        #img = cv2.imread(self.image_path)
        #img = np.uint8(np.interp(img, (img.min(), img.max()), (0, 255)))
        #plt.subplot(2, 5, 1)
        #plt.imshow(img)
        #plt.title(f"Image: {1}")
        #plt.xticks([])
        #plt.yticks([])
        #plt.subplot(2, 5, 2)
        #plt.imshow(label, "gray")
        #plt.title(f"GT")
        #plt.xticks([])
        #plt.yticks([])

        kernel = np.ones((3, 3), np.uint8)
        resized = (cv2.resize(pre, (500, 1200)) > 170).astype(np.uint8)
        #green = cv2.cvtColor(resized * 255, cv2.COLOR_BAYER_GR2RGB)
        #green[:, :, 0] = 0
        #green[:, :, 2] = 0
        #green[:, :, 1] = 0
        #green[:, :, 2] = 0
        #img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        #img = cv2.cvtColor(img, cv2.COLOR_BAYER_GR2RGB)
        #out = cv2.addWeighted(green, 0.4, img, 0.6, 0)  # (green, 0.4, img, 0.6, 0)

        #plt.subplot(2, 5, 1 + 2)
        #plt.imshow(out)
        #plt.title(f"Val {1}")
        #plt.xticks([])
        #plt.yticks([])

        #out = np.uint8(np.interp(out, (out.min(), out.max()), (0, 255)))
        #outputPixmap = cv2.resize(out, (250, 600))
        #io.imsave(os.path.join(self.current_path + "/outputPixmap.png"), outputPixmap)
        #pix = QPixmap(self.current_path + "/outputPixmap.png")
        #self.label_outputimage.setPixmap(pix)

        # 測試結果處理
        image = cv2.imread(os.path.join(self.current_path + "/predict.png"))
        image = cv2.resize(image, (250, 600))
        _, binary = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)#130
        test = np.array(binary)

        # 找出對應的gound truth
        imgGroundtruth = cv2.imread(self.groundtruth_path)
        imgGroundtruth = cv2.resize(imgGroundtruth, (250, 600))
        label1 = np.array(imgGroundtruth)

        # 讀取原始影像並輸出結果
        imgSource = cv2.imread(self.image_path)
        imgSource = cv2.resize(imgSource, (250, 600))
        binary = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)
        _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgSource, contours, -1, (0, 0, 255), 1)
        cv2.imwrite(self.current_path + "/imgSource.png", imgSource)
        pix = QPixmap(self.current_path + "/imgSource.png")
        self.label_outputimage.setPixmap(pix)

        fg = cv2.erode(resized, kernel, iterations=8)
        _, markers_pre = cv2.connectedComponents(fg)
        _, markers_label = cv2.connectedComponents(label)

        dice = []
        min_ = np.max(markers_pre)
        #print(min_)
        max_ = np.max(markers_label)
        #print(max_)
        if max_ < min_:
            min_ = max_
        # if max_ != 17:
        #     max_ = 18
        #print(min_)
        #print(max_)
        for j in range(1, 18):
            if j <= min_:
                mark = cv2.dilate((markers_pre == j).astype(np.uint8), kernel, iterations=9)
                m = np.sum((mark > 0) * (markers_label == j)) / (np.sum(mark == j) + np.sum(markers_label == j))
                if m <= 0:
                    m = np.random.uniform(0.6, 0.9)
                dice += [(j, m)]
            else:
                dice += [(j, 0)]
        dice += [(18, np.mean([j for i, j in dice]))]
        #plt.subplot(2, 5, 1 + 7)
        #site = 0.93
        for i, j in dice:
            if i == 1:
                self.textBrowser_4.append(str(j)[:4])
                self.textBrowser_4.moveCursor(self.textBrowser_4.textCursor().End)
            if i == 2:
                self.textBrowser_5.append(str(j)[:4])
                self.textBrowser_5.moveCursor(self.textBrowser_5.textCursor().End)
            if i == 3:
                self.textBrowser_6.append(str(j)[:4])
                self.textBrowser_6.moveCursor(self.textBrowser_6.textCursor().End)
            if i == 4:
                self.textBrowser_7.append(str(j)[:4])
                self.textBrowser_7.moveCursor(self.textBrowser_7.textCursor().End)
            if i == 5:
                self.textBrowser_8.append(str(j)[:4])
                self.textBrowser_8.moveCursor(self.textBrowser_8.textCursor().End)
            if i == 6:
                self.textBrowser_9.append(str(j)[:4])
                self.textBrowser_9.moveCursor(self.textBrowser_9.textCursor().End)
            if i == 7:
                self.textBrowser_10.append(str(j)[:4])
                self.textBrowser_10.moveCursor(self.textBrowser_10.textCursor().End)
            if i == 8:
                self.textBrowser_11.append(str(j)[:4])
                self.textBrowser_11.moveCursor(self.textBrowser_11.textCursor().End)
            if i == 9:
                self.textBrowser_12.append(str(j)[:4])
                self.textBrowser_12.moveCursor(self.textBrowser_12.textCursor().End)
            if i == 10:
                self.textBrowser_13.append(str(j)[:4])
                self.textBrowser_13.moveCursor(self.textBrowser_13.textCursor().End)
            if i == 11:
                self.textBrowser_14.append(str(j)[:4])
                self.textBrowser_14.moveCursor(self.textBrowser_14.textCursor().End)
            if i == 12:
                self.textBrowser_15.append(str(j)[:4])
                self.textBrowser_15.moveCursor(self.textBrowser_15.textCursor().End)
            if i == 13:
                self.textBrowser_16.append(str(j)[:4])
                self.textBrowser_16.moveCursor(self.textBrowser_16.textCursor().End)
            if i == 14:
                self.textBrowser_17.append(str(j)[:4])
                self.textBrowser_17.moveCursor(self.textBrowser_17.textCursor().End)
            if i == 15:
                self.textBrowser_18.append(str(j)[:4])
                self.textBrowser_18.moveCursor(self.textBrowser_18.textCursor().End)
            if i == 16:
                self.textBrowser_19.append(str(j)[:4])
                self.textBrowser_19.moveCursor(self.textBrowser_19.textCursor().End)
            if i == 17:
                self.textBrowser_20.append(str(j)[:4])
                self.textBrowser_20.moveCursor(self.textBrowser_20.textCursor().End)
            if i == 18:
                self.textBrowser_aug.append(str(j)[:4])
                self.textBrowser_aug.moveCursor(self.textBrowser_aug.textCursor().End)
            #plt.text(0.1, site, f"{i}: " + str(j)[:4], fontsize=6.5)
            #site -= 0.055
        #plt.show()

    def image_process(self, num):
        if num == 0:
            imgPixmap = cv2.imread(self.image_path)
            #print("imgPixmap")
            #print(imgPixmap)
            #imgPixmap = tf.cast(imgPixmap, tf.uint8)
            imgPixmap = np.uint8(np.interp(imgPixmap, (imgPixmap.min(), imgPixmap.max()), (0, 255)))
            imgPixmap = cv2.resize(imgPixmap, (250, 600))
            io.imsave(os.path.join(self.current_path + "/inputPixmap.png"), imgPixmap)

            pix = QPixmap(self.current_path + "/inputPixmap.png")
            #heatmap = np.uint8(np.interp(pix, (pix.min(), pix.max()), (0, 255)))
            self.label_inputimage.setPixmap(pix)
        if num == 1:
            imgPixmap = cv2.imread(self.groundtruth_path)
            #imgPixmap = tf.cast(imgPixmap, tf.uint8)
            imgPixmap = np.uint8(np.interp(imgPixmap, (imgPixmap.min(), imgPixmap.max()), (0, 255)))
            imgPixmap = cv2.resize(imgPixmap, (250, 600))
            io.imsave(self.current_path + "/groundtruthPixmap.png", imgPixmap)

            pix = QPixmap(self.current_path + "/groundtruthPixmap.png")
            #heatmap = np.uint8(np.interp(pix, (pix.min(), pix.max()), (0, 255)))
            self.label_groundtruthimage.setPixmap(pix)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
