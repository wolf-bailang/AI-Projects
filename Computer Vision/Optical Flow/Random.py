from __future__ import absolute_import

from PIL import Image
import random
import math
import numpy as np
import cv2
import os
from tqdm import tqdm
import tensorflow as tf


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = tf.convert_to_tensor(img.size()[1], dtype=tf.float32) * tf.convert_to_tensor(img.size()[2], dtype=tf.float32)

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    #img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 255)
                    #img[1, x1:x1+h, y1:y1+w] = random.uniform(0, 255)
                    #img[2, x1:x1+h, y1:y1+w] = random.uniform(0, 255)
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                    #img[:, x1:x1+h, y1:y1+w] = np.random.rand(3, h, w)
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    #img[0, x1:x1+h, y1:y1+w] = np.random.rand(1, h, w)
                return img

        return img


# this labelling function is just for the visualization, we'll have separate one for preprocessing.
def label_img(img):
    category = img.split('.')[-3]
    if category == 'cat':
        return [1, 0]
    elif category == 'dog':
        return [0, 1]


# Process the data, here we're converting images into numpy array. This function takes image data, image directory, a boolean as an argument.
def process_data(img_data, data_dir, isTrain=True):
    data_df = []

    for img in tqdm(img_data):
        path = os.path.join(data_dir, img)  # Assigning path to images by concatenating directory and images
        if (isTrain):
            label = label_img(img)  # Calling label_img to assign labels to image present in training directory
        else:
            label = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        img = RE(img.copy())

        img = cv2.resize(img, (224, 224))
        data_df.append([np.array(img), np.array(label)])  # append image and labels as numpy array in data_df list
    shuffle(data_df)
    return data_df

# We will plot the images of dogs and cats and display the assigned label above image
def show_images(data, isTest=False):
    f, ax = plt.subplots(nrows=5, ncols=5, figsize=(15, 15))
    for i, data in enumerate(data[:25]):  # enumerate helps in keeping track of count of iterations
        img_num = data[1]
        img_data = data[0]
        label = np.argmax(img_num)  # to get maximum indices of an array
        if label == 1:
            str_label = 'Dog'
        elif label == 0:
            str_label = 'Cat'
        if (isTest):
            str_label = "None"
        ax[i // 5, i % 5].imshow(img_data)
        ax[i // 5, i % 5].axis('off')  # removing axis for better look
        ax[i // 5, i % 5].set_title()
    plt.show()


def get_path(directory):
    path = []
    for files in os.listdir(directory):
        print(files)
        path.append(files)
    return path


if __name__ == "__main__":
    img = cv2.imread("test.jpg")
    directory = '/home/wolf/zjb/Homework/Hw2/Q5/train1/'
    train_path = get_path(directory)
    RE = RandomErasing(1)
    train = process_data(train_path, '/home/wolf/zjb/Homework/Hw2/Q5/train1/')
    show_images(train)
