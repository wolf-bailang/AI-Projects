# ！/usr/bin/python3.6
# encoding: utf-8
"""
@author: Junbin Zhang
@email: p78083025@gs.ncku.edu.tw
@time: 2020/11/07
"""

from __future__ import print_function

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from UI import Ui_MainWindow
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5.QtGui import QPixmap

from torch.utils.data import DataLoader

import torchvision.transforms as transforms
from torchvision import datasets

import os

from model import Hyperparameters, VGG16_Classifier, model_build, predict, model_process


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
        self.pushButton_image_17.clicked.connect(self.HOME_page)
        self.pushButton_image_5.clicked.connect(self.VGG16_Classifier_page)
        self.comboBox_2.currentIndexChanged.connect(self.get_comboBox_2)
        self.pushButton_image_18.clicked.connect(self.Training_Images_click)
        self.pushButton_image_19.clicked.connect(self.Hyperparameters_click)
        self.pushButton_image_20.clicked.connect(self.Model_Structure_click)
        self.pushButton_image_21.clicked.connect(self.Accuracy_click)
        self.pushButton_image_22.clicked.connect(self.Test_click)
        self.current_path = ' '
        self.image_path = ' '
        self.comboBox_2.addItems([' ', '0-999', '1000-1999', '2000-2999', '3000-3999', '4000-4999', '5000-5999',
                                  '6000-6999', '7000-7999', '8000-8999', '9000-9999'])
        self.index = 1

    def HOME_page(self):
        self.stackedWidget.setCurrentIndex(0)

    def get_comboBox_2(self):
        self.comboBox_2.setCurrentText(self.comboBox_2.currentText())
        if self.comboBox_2.currentText() == '0-999':
            self.index = np.random.randint(0, 999)
        elif self.comboBox_2.currentText() == '1000-1999':
            self.index = np.random.randint(1000, 1999)
        elif self.comboBox_2.currentText() == '2000-2999':
            self.index = np.random.randint(2000, 2999)
        elif self.comboBox_2.currentText() == '3000-3999':
            self.index = np.random.randint(3000, 3999)
        elif self.comboBox_2.currentText() == '4000-4999':
            self.index = np.random.randint(4000, 4999)
        elif self.comboBox_2.currentText() == '5000-5999':
            self.index = np.random.randint(5000, 5999)
        elif self.comboBox_2.currentText() == '6000-6999':
            self.index = np.random.randint(6000, 6999)
        elif self.comboBox_2.currentText() == '2000-2999':
            self.index = np.random.randint(7000, 7999)
        elif self.comboBox_2.currentText() == '8000-8999':
            self.index = np.random.randint(8000, 8999)
        elif self.comboBox_2.currentText() == '9000-9999':
            self.index = np.random.randint(9000, 9999)

    def VGG16_Classifier_page(self):
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

    def Training_Images_click(self):
        self.get_current_path()
        print("==> Preparing data..")
        transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                              transforms.RandomHorizontalFlip(), transforms.ToTensor(),
                                              transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                   (0.2023, 0.1994, 0.2010)), ])
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        # specify the names of the classes
        classes_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
                        8: 'ship', 9: 'truck'}
        """"
        # plot 25 random images from training dataset
        fig, axs = plt.subplots(2, 5, figsize=(10, 10))
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            for im in range(10):
                image = inputs[im].permute(1, 2, 0)
                i = im // 5
                j = im % 5
                axs[i, j].imshow(image.numpy())  # plot the data
                axs[i, j].axis('off')
                axs[i, j].set_title(classes_dict[int(labels[im].numpy())])
            break
        """
        fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 10))
        fig.subplots_adjust(wspace=0.5, hspace=0.5)
        for batch_idx, (inputs, labels) in enumerate(trainloader):
            for im in range(10):
                image = inputs[im].permute(1, 2, 0)
                i = im // 5
                j = im % 5
                axs[i, j].imshow(image.numpy())  # plot the data
                axs[i, j].axis('off')
                axs[i, j].set_title(classes_dict[int(labels[im].numpy())], fontsize=20)
                axs[i][j].axes.get_xaxis().set_visible(False)
                axs[i][j].axes.get_yaxis().set_visible(False)
            break
        # set suptitle
        # plt.suptitle('CIFAR-10 Images')
        plt.tight_layout(pad=0, h_pad=0.5, w_pad=0.5, rect=None)
        plt.savefig(self.current_path + "/train.png")
        # plt.show()
        img = cv2.imread(self.current_path + "/train.png")
        # print(img.shape)
        cropped = img[140:840, 0:1000]  # 裁剪坐标为[y0:y1, x0:x1]
        img = cv2.resize(cropped, (480, 271))
        io.imsave(self.current_path + "/trainPixmap.png", img)
        pix = QPixmap(self.current_path + "/trainPixmap.png")
        self.label_inputimage_8.setPixmap(pix)

    def Hyperparameters_click(self):
        self.get_current_path()
        _, _, Epochs, learning_rate, batch_size, _, _, _ = Hyperparameters()
        self.textBrowser_image_2.append('Batch Size:           ' + str(batch_size))
        self.textBrowser_image_2.append('Learning Rate:    ' + str(learning_rate))
        self.textBrowser_image_2.append('Epochs:                  ' + str(Epochs))
        self.textBrowser_image_2.append('Optimizer:            SGD')
        self.textBrowser_image_2.moveCursor(self.textBrowser_image_2.textCursor().End)  # 文本框顯示到底部

    def Model_Structure_click(self):
        self.get_current_path()
        net, device = model_build(resume=True)
        for i in net._modules.items():
            self.textBrowser_image_3.append(str(i))
        self.textBrowser_image_3.append(' ')

        # 定义总参数量、可训练参数量及非可训练参数量变量
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0
        # 遍历model.parameters()返回的全局参数列表
        for param in net.parameters():
            mulValue = np.prod(param.size())  # 使用numpy prod接口计算参数数组所有元素之积
            Total_params += mulValue  # 总参数量
            if param.requires_grad:
                Trainable_params += mulValue  # 可训练参数量
            else:
                NonTrainable_params += mulValue  # 非可训练参数量
        self.textBrowser_image_3.append('Total params:        ' + str(Total_params))
        self.textBrowser_image_3.append('Trainable params:    ' + str(Trainable_params))
        self.textBrowser_image_3.append('NonTrainable params: ' + str(NonTrainable_params))
        self.textBrowser_image_3.moveCursor(self.textBrowser_image_3.textCursor().End)  # 文本框顯示到底部

    def Accuracy_click(self):
        self.get_current_path()
        VGG16_Classifier(train_model=False, resume=True)
        img = cv2.imread(self.current_path + "/Accuracy_Loss.png")
        print(img.shape)
        cropped = img[50:670, 15:526]  # 裁剪坐标为[y0:y1, x0:x1]
        img = cv2.resize(cropped, (471, 471))
        io.imsave(self.current_path + "/Accuracy_LossPixmap.png", img)
        pix = QPixmap(self.current_path + "/Accuracy_LossPixmap.png")
        self.label_inputimage_16.setPixmap(pix)

    def Test_click(self):
        self.get_current_path()
        # print('self.index= ' + str(self.index))
        transform_test = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                  (0.2023, 0.1994, 0.2010)), ])
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        # print('testset= ' + str(len(testset)))
        testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
        # testloader = DataLoader(testset, batch_size=100, sampler=None, shuffle=False, num_workers=2)
        # print('testloader= ' + str(len(testloader)))
        classes_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
                        8: 'ship', 9: 'truck'}
        net, device = model_build(resume=True)
        predict(net, testloader, device, classes_dict, self.index)
        img = cv2.imread(self.current_path + "/predict.png")
        # print(img.shape)
        cropped = img[0:1000, 240:730]  # 裁剪坐标为[y0:y1, x0:x1]
        img = cv2.resize(cropped, (331, 621))
        io.imsave(self.current_path + "/predictPixmap.png", img)
        pix = QPixmap(self.current_path + "/predictPixmap.png")
        self.label_inputimage_17.setPixmap(pix)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
