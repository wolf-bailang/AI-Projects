# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hw.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1282, 942)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(-40, -10, 1311, 901))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.stackedWidget.setFont(font)
        self.stackedWidget.setObjectName("stackedWidget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.label_4 = QtWidgets.QLabel(self.page)
        self.label_4.setGeometry(QtCore.QRect(460, 100, 401, 81))
        font = QtGui.QFont()
        font.setPointSize(17)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.pushButton_image_2 = QtWidgets.QPushButton(self.page)
        self.pushButton_image_2.setGeometry(QtCore.QRect(400, 390, 171, 71))
        self.pushButton_image_2.setObjectName("pushButton_image_2")
        self.label_5 = QtWidgets.QLabel(self.page)
        self.label_5.setGeometry(QtCore.QRect(500, 250, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.page)
        self.label_6.setGeometry(QtCore.QRect(610, 250, 81, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.pushButton_image_5 = QtWidgets.QPushButton(self.page)
        self.pushButton_image_5.setGeometry(QtCore.QRect(970, 390, 171, 71))
        self.pushButton_image_5.setObjectName("pushButton_image_5")
        self.pushButton_image_1 = QtWidgets.QPushButton(self.page)
        self.pushButton_image_1.setGeometry(QtCore.QRect(210, 390, 171, 71))
        self.pushButton_image_1.setObjectName("pushButton_image_1")
        self.label = QtWidgets.QLabel(self.page)
        self.label.setGeometry(QtCore.QRect(590, 170, 141, 51))
        font = QtGui.QFont()
        font.setPointSize(18)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.page)
        self.label_2.setGeometry(QtCore.QRect(710, 250, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.pushButton_image_3 = QtWidgets.QPushButton(self.page)
        self.pushButton_image_3.setGeometry(QtCore.QRect(590, 390, 171, 71))
        self.pushButton_image_3.setObjectName("pushButton_image_3")
        self.pushButton_image_4 = QtWidgets.QPushButton(self.page)
        self.pushButton_image_4.setGeometry(QtCore.QRect(780, 390, 171, 71))
        self.pushButton_image_4.setObjectName("pushButton_image_4")
        self.label_3 = QtWidgets.QLabel(self.page)
        self.label_3.setGeometry(QtCore.QRect(500, 20, 321, 151))
        font = QtGui.QFont()
        font.setPointSize(26)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.stackedWidget.addWidget(self.page)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.pushButton_image_12 = QtWidgets.QPushButton(self.page_3)
        self.pushButton_image_12.setGeometry(QtCore.QRect(450, 60, 161, 41))
        self.pushButton_image_12.setObjectName("pushButton_image_12")
        self.pushButton_image_13 = QtWidgets.QPushButton(self.page_3)
        self.pushButton_image_13.setGeometry(QtCore.QRect(1030, 50, 141, 81))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pushButton_image_13.setFont(font)
        self.pushButton_image_13.setObjectName("pushButton_image_13")
        self.label_inputimage_4 = QtWidgets.QLabel(self.page_3)
        self.label_inputimage_4.setGeometry(QtCore.QRect(450, 190, 800, 700))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_inputimage_4.setFont(font)
        self.label_inputimage_4.setAutoFillBackground(True)
        self.label_inputimage_4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_inputimage_4.setText("")
        self.label_inputimage_4.setObjectName("label_inputimage_4")
        self.label_inputimage_8 = QtWidgets.QLabel(self.page_3)
        self.label_inputimage_8.setGeometry(QtCore.QRect(120, 170, 255, 300))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_inputimage_8.setFont(font)
        self.label_inputimage_8.setAutoFillBackground(True)
        self.label_inputimage_8.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_inputimage_8.setText("")
        self.label_inputimage_8.setObjectName("label_inputimage_8")
        self.label_inputimage_9 = QtWidgets.QLabel(self.page_3)
        self.label_inputimage_9.setGeometry(QtCore.QRect(120, 520, 255, 300))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_inputimage_9.setFont(font)
        self.label_inputimage_9.setAutoFillBackground(True)
        self.label_inputimage_9.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_inputimage_9.setText("")
        self.label_inputimage_9.setObjectName("label_inputimage_9")
        self.label_input_6 = QtWidgets.QLabel(self.page_3)
        self.label_input_6.setEnabled(True)
        self.label_input_6.setGeometry(QtCore.QRect(200, 470, 101, 41))
        self.label_input_6.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label_input_6.setAutoFillBackground(False)
        self.label_input_6.setLocale(QtCore.QLocale(QtCore.QLocale.Chinese, QtCore.QLocale.China))
        self.label_input_6.setTextFormat(QtCore.Qt.AutoText)
        self.label_input_6.setScaledContents(False)
        self.label_input_6.setWordWrap(False)
        self.label_input_6.setObjectName("label_input_6")
        self.label_input_7 = QtWidgets.QLabel(self.page_3)
        self.label_input_7.setEnabled(True)
        self.label_input_7.setGeometry(QtCore.QRect(190, 830, 101, 41))
        self.label_input_7.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label_input_7.setAutoFillBackground(False)
        self.label_input_7.setLocale(QtCore.QLocale(QtCore.QLocale.Chinese, QtCore.QLocale.China))
        self.label_input_7.setTextFormat(QtCore.Qt.AutoText)
        self.label_input_7.setScaledContents(False)
        self.label_input_7.setWordWrap(False)
        self.label_input_7.setObjectName("label_input_7")
        self.label_input_9 = QtWidgets.QLabel(self.page_3)
        self.label_input_9.setEnabled(True)
        self.label_input_9.setGeometry(QtCore.QRect(810, 140, 101, 41))
        self.label_input_9.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label_input_9.setAutoFillBackground(False)
        self.label_input_9.setLocale(QtCore.QLocale(QtCore.QLocale.Chinese, QtCore.QLocale.China))
        self.label_input_9.setTextFormat(QtCore.Qt.AutoText)
        self.label_input_9.setScaledContents(False)
        self.label_input_9.setWordWrap(False)
        self.label_input_9.setObjectName("label_input_9")
        self.stackedWidget.addWidget(self.page_3)
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setObjectName("page_4")
        self.pushButton_image_14 = QtWidgets.QPushButton(self.page_4)
        self.pushButton_image_14.setGeometry(QtCore.QRect(440, 50, 161, 41))
        self.pushButton_image_14.setObjectName("pushButton_image_14")
        self.pushButton_image_16 = QtWidgets.QPushButton(self.page_4)
        self.pushButton_image_16.setGeometry(QtCore.QRect(1020, 40, 141, 81))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pushButton_image_16.setFont(font)
        self.pushButton_image_16.setObjectName("pushButton_image_16")
        self.label_inputimage_5 = QtWidgets.QLabel(self.page_4)
        self.label_inputimage_5.setGeometry(QtCore.QRect(90, 90, 300, 400))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_inputimage_5.setFont(font)
        self.label_inputimage_5.setAutoFillBackground(True)
        self.label_inputimage_5.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_inputimage_5.setText("")
        self.label_inputimage_5.setObjectName("label_inputimage_5")
        self.pushButton_image_15 = QtWidgets.QPushButton(self.page_4)
        self.pushButton_image_15.setGeometry(QtCore.QRect(650, 50, 161, 41))
        self.pushButton_image_15.setObjectName("pushButton_image_15")
        self.label_inputimage_6 = QtWidgets.QLabel(self.page_4)
        self.label_inputimage_6.setGeometry(QtCore.QRect(90, 500, 300, 400))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_inputimage_6.setFont(font)
        self.label_inputimage_6.setAutoFillBackground(True)
        self.label_inputimage_6.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_inputimage_6.setText("")
        self.label_inputimage_6.setObjectName("label_inputimage_6")
        self.label_inputimage_7 = QtWidgets.QLabel(self.page_4)
        self.label_inputimage_7.setGeometry(QtCore.QRect(410, 200, 900, 700))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_inputimage_7.setFont(font)
        self.label_inputimage_7.setAutoFillBackground(True)
        self.label_inputimage_7.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_inputimage_7.setText("")
        self.label_inputimage_7.setObjectName("label_inputimage_7")
        self.label_input_10 = QtWidgets.QLabel(self.page_4)
        self.label_input_10.setEnabled(True)
        self.label_input_10.setGeometry(QtCore.QRect(800, 150, 101, 41))
        self.label_input_10.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label_input_10.setAutoFillBackground(False)
        self.label_input_10.setLocale(QtCore.QLocale(QtCore.QLocale.Chinese, QtCore.QLocale.China))
        self.label_input_10.setTextFormat(QtCore.Qt.AutoText)
        self.label_input_10.setScaledContents(False)
        self.label_input_10.setWordWrap(False)
        self.label_input_10.setObjectName("label_input_10")
        self.stackedWidget.addWidget(self.page_4)
        self.page_5 = QtWidgets.QWidget()
        self.page_5.setObjectName("page_5")
        self.stackedWidget.addWidget(self.page_5)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.pushButton_image_10 = QtWidgets.QPushButton(self.page_2)
        self.pushButton_image_10.setGeometry(QtCore.QRect(1060, 80, 141, 81))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pushButton_image_10.setFont(font)
        self.pushButton_image_10.setObjectName("pushButton_image_10")
        self.pushButton_image_11 = QtWidgets.QPushButton(self.page_2)
        self.pushButton_image_11.setGeometry(QtCore.QRect(480, 90, 161, 41))
        self.pushButton_image_11.setObjectName("pushButton_image_11")
        self.label_inputimage_3 = QtWidgets.QLabel(self.page_2)
        self.label_inputimage_3.setGeometry(QtCore.QRect(220, 190, 800, 700))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_inputimage_3.setFont(font)
        self.label_inputimage_3.setAutoFillBackground(True)
        self.label_inputimage_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_inputimage_3.setText("")
        self.label_inputimage_3.setObjectName("label_inputimage_3")
        self.label_input_8 = QtWidgets.QLabel(self.page_2)
        self.label_input_8.setEnabled(True)
        self.label_input_8.setGeometry(QtCore.QRect(590, 140, 101, 41))
        self.label_input_8.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label_input_8.setAutoFillBackground(False)
        self.label_input_8.setLocale(QtCore.QLocale(QtCore.QLocale.Chinese, QtCore.QLocale.China))
        self.label_input_8.setTextFormat(QtCore.Qt.AutoText)
        self.label_input_8.setScaledContents(False)
        self.label_input_8.setWordWrap(False)
        self.label_input_8.setObjectName("label_input_8")
        self.stackedWidget.addWidget(self.page_2)
        self.page_1 = QtWidgets.QWidget()
        self.page_1.setObjectName("page_1")
        self.pushButton_image_6 = QtWidgets.QPushButton(self.page_1)
        self.pushButton_image_6.setGeometry(QtCore.QRect(190, 30, 131, 41))
        self.pushButton_image_6.setObjectName("pushButton_image_6")
        self.label_input_1 = QtWidgets.QLabel(self.page_1)
        self.label_input_1.setEnabled(True)
        self.label_input_1.setGeometry(QtCore.QRect(90, 90, 101, 41))
        self.label_input_1.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label_input_1.setAutoFillBackground(False)
        self.label_input_1.setLocale(QtCore.QLocale(QtCore.QLocale.Chinese, QtCore.QLocale.China))
        self.label_input_1.setTextFormat(QtCore.Qt.AutoText)
        self.label_input_1.setScaledContents(False)
        self.label_input_1.setWordWrap(False)
        self.label_input_1.setObjectName("label_input_1")
        self.label_inputimage = QtWidgets.QLabel(self.page_1)
        self.label_inputimage.setGeometry(QtCore.QRect(70, 180, 400, 300))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_inputimage.setFont(font)
        self.label_inputimage.setAutoFillBackground(True)
        self.label_inputimage.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_inputimage.setText("")
        self.label_inputimage.setObjectName("label_inputimage")
        self.pushButton_image_7 = QtWidgets.QPushButton(self.page_1)
        self.pushButton_image_7.setGeometry(QtCore.QRect(340, 30, 131, 41))
        self.pushButton_image_7.setObjectName("pushButton_image_7")
        self.comboBox = QtWidgets.QComboBox(self.page_1)
        self.comboBox.setGeometry(QtCore.QRect(190, 90, 131, 41))
        self.comboBox.setSizeIncrement(QtCore.QSize(0, 14))
        self.comboBox.setEditable(True)
        self.comboBox.setCurrentText("")
        self.comboBox.setMaxVisibleItems(15)
        self.comboBox.setMinimumContentsLength(5)
        self.comboBox.setDuplicatesEnabled(True)
        self.comboBox.setObjectName("comboBox")
        self.label_input_2 = QtWidgets.QLabel(self.page_1)
        self.label_input_2.setEnabled(True)
        self.label_input_2.setGeometry(QtCore.QRect(190, 490, 131, 41))
        self.label_input_2.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label_input_2.setAutoFillBackground(False)
        self.label_input_2.setLocale(QtCore.QLocale(QtCore.QLocale.Chinese, QtCore.QLocale.China))
        self.label_input_2.setTextFormat(QtCore.Qt.AutoText)
        self.label_input_2.setScaledContents(False)
        self.label_input_2.setWordWrap(False)
        self.label_input_2.setObjectName("label_input_2")
        self.label_input_3 = QtWidgets.QLabel(self.page_1)
        self.label_input_3.setEnabled(True)
        self.label_input_3.setGeometry(QtCore.QRect(800, 140, 101, 41))
        self.label_input_3.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label_input_3.setAutoFillBackground(False)
        self.label_input_3.setLocale(QtCore.QLocale(QtCore.QLocale.Chinese, QtCore.QLocale.China))
        self.label_input_3.setTextFormat(QtCore.Qt.AutoText)
        self.label_input_3.setScaledContents(False)
        self.label_input_3.setWordWrap(False)
        self.label_input_3.setObjectName("label_input_3")
        self.label_input_4 = QtWidgets.QLabel(self.page_1)
        self.label_input_4.setEnabled(True)
        self.label_input_4.setGeometry(QtCore.QRect(60, 30, 131, 41))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_input_4.setFont(font)
        self.label_input_4.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label_input_4.setAutoFillBackground(False)
        self.label_input_4.setLocale(QtCore.QLocale(QtCore.QLocale.Chinese, QtCore.QLocale.China))
        self.label_input_4.setTextFormat(QtCore.Qt.AutoText)
        self.label_input_4.setScaledContents(False)
        self.label_input_4.setWordWrap(False)
        self.label_input_4.setObjectName("label_input_4")
        self.pushButton_image_9 = QtWidgets.QPushButton(self.page_1)
        self.pushButton_image_9.setGeometry(QtCore.QRect(640, 30, 131, 41))
        self.pushButton_image_9.setObjectName("pushButton_image_9")
        self.textBrowser_image_1 = QtWidgets.QTextBrowser(self.page_1)
        self.textBrowser_image_1.setGeometry(QtCore.QRect(60, 530, 411, 191))
        self.textBrowser_image_1.setObjectName("textBrowser_image_1")
        self.pushButton_image_8 = QtWidgets.QPushButton(self.page_1)
        self.pushButton_image_8.setGeometry(QtCore.QRect(490, 30, 131, 41))
        self.pushButton_image_8.setObjectName("pushButton_image_8")
        self.textBrowser_image = QtWidgets.QTextBrowser(self.page_1)
        self.textBrowser_image.setGeometry(QtCore.QRect(340, 90, 521, 41))
        self.textBrowser_image.setObjectName("textBrowser_image")
        self.label_inputimage_2 = QtWidgets.QLabel(self.page_1)
        self.label_inputimage_2.setGeometry(QtCore.QRect(500, 180, 800, 700))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_inputimage_2.setFont(font)
        self.label_inputimage_2.setAutoFillBackground(True)
        self.label_inputimage_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_inputimage_2.setText("")
        self.label_inputimage_2.setObjectName("label_inputimage_2")
        self.label_input_5 = QtWidgets.QLabel(self.page_1)
        self.label_input_5.setEnabled(True)
        self.label_input_5.setGeometry(QtCore.QRect(220, 140, 91, 41))
        self.label_input_5.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.label_input_5.setAutoFillBackground(False)
        self.label_input_5.setLocale(QtCore.QLocale(QtCore.QLocale.Chinese, QtCore.QLocale.China))
        self.label_input_5.setTextFormat(QtCore.Qt.AutoText)
        self.label_input_5.setScaledContents(False)
        self.label_input_5.setWordWrap(False)
        self.label_input_5.setObjectName("label_input_5")
        self.pushButton_image = QtWidgets.QPushButton(self.page_1)
        self.pushButton_image.setGeometry(QtCore.QRect(1040, 50, 141, 81))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.pushButton_image.setFont(font)
        self.pushButton_image.setObjectName("pushButton_image")
        self.stackedWidget.addWidget(self.page_1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1282, 28))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(1)
        self.comboBox.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_4.setText(_translate("MainWindow", "(Computer Vision and Deep Learning)"))
        self.pushButton_image_2.setText(_translate("MainWindow", "2. Augmented Reality"))
        self.label_5.setText(_translate("MainWindow", "張軍斌"))
        self.label_6.setText(_translate("MainWindow", "資工所"))
        self.pushButton_image_5.setText(_translate("MainWindow", "5."))
        self.pushButton_image_1.setText(_translate("MainWindow", "1. Camera Calibration"))
        self.label.setText(_translate("MainWindow", "Homework 1\n"
""))
        self.label_2.setText(_translate("MainWindow", "P78083025"))
        self.pushButton_image_3.setText(_translate("MainWindow", "3. Stereo Disparity Map"))
        self.pushButton_image_4.setText(_translate("MainWindow", "4. SIFT"))
        self.label_3.setText(_translate("MainWindow", "電腦視覺與深度學習\n"
""))
        self.pushButton_image_12.setText(_translate("MainWindow", "Disparity Map"))
        self.pushButton_image_13.setText(_translate("MainWindow", "HOME"))
        self.label_input_6.setText(_translate("MainWindow", "Left Image"))
        self.label_input_7.setText(_translate("MainWindow", "Right Image"))
        self.label_input_9.setText(_translate("MainWindow", "Output Image"))
        self.pushButton_image_14.setText(_translate("MainWindow", "4.1 Keypoints"))
        self.pushButton_image_16.setText(_translate("MainWindow", "HOME"))
        self.pushButton_image_15.setText(_translate("MainWindow", "4.2 Matched Keypoints"))
        self.label_input_10.setText(_translate("MainWindow", "Output Image"))
        self.pushButton_image_10.setText(_translate("MainWindow", "HOME"))
        self.pushButton_image_11.setText(_translate("MainWindow", "Augmented_Reality"))
        self.label_input_8.setText(_translate("MainWindow", "Output Image"))
        self.pushButton_image_6.setText(_translate("MainWindow", "1.1 Find Corners"))
        self.label_input_1.setText(_translate("MainWindow", "Select Image"))
        self.pushButton_image_7.setText(_translate("MainWindow", "1.2 Find Intrinsic"))
        self.label_input_2.setText(_translate("MainWindow", "Output Parameter"))
        self.label_input_3.setText(_translate("MainWindow", "Output Image"))
        self.label_input_4.setText(_translate("MainWindow", "1. Calibration"))
        self.pushButton_image_9.setText(_translate("MainWindow", "1.4 Find Distortion"))
        self.pushButton_image_8.setText(_translate("MainWindow", "1.3 Find Extrinsic"))
        self.label_input_5.setText(_translate("MainWindow", "Input Image"))
        self.pushButton_image.setText(_translate("MainWindow", "HOME"))

