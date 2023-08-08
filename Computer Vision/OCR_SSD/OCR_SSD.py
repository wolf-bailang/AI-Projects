# -*- coding: UTF-8 -*-

"""
尝试解决SVHN任务。SVHN数据包含三种不同的数据集：train，test和extra。差异不是100％明确，但是最大的额外数据集（约500K
样本）包括以某种方式更容易识别的图像。
需要做以下准备任务：
    你需要一台的GPU机器，Tensorflow≥1.4，Keras≥2
    从这里克隆SSD_Keras项目。   https://github.com/pierluigiferrari/ssd_keras
    从此处下载coco数据集上预先培训的SSD300模型。     https://drive.google.com/open?id=1vmEF7FUsWfHquXyCqO17UaXOPpRbwsdj
    从这里克隆项目.    https://github.com/shgidi/OCR
    下载extra.tar.gz文件，其中包含SVHN数据集的额外图像。      http://ufldl.stanford.edu/housenumbers/extra.tar.gz
    更新此项目仓库中json_config.json中的所有相关路径。

@shgidi
https://github.com/shgidi/OCR/blob/master/ssd_OCR.ipynb
"""
import ipywidgets as widgets    # ipywidgets包可以实现jupyter notebook笔记本的交互式控件操作。
import os
import sys
import skimage.io       # skimage提供了io模块，这个模块是用来操作图片输入输出的。
import scipy        # 高级的科学计算库
import json

# 加载配置
with open('json_config.json') as f:
    json_conf = json.load(f)

ROOT_DIR = os.path.abspath(json_conf['ssd_folder'])     # add here mask RCNN path
sys.path.append(ROOT_DIR)

import cv2
from utils_ssd import *
import pandas as pd
from PIL import Image       # PIL图像处理标准库
from matplotlib import pyplot as plt
%matplotlib inline
%load_ext autoreload
%autoreload 2

data_folder = json_conf['data_folder']

# 数据集处理
# SVHN parsing
def read_process_h5(filename):
    """ Reads and processes the mat files provided in the SVHN dataset.
        Input: filename
        Ouptut: list of python dictionaries
    """
    f = h5py.File(filename, 'r')    #读h5文件
    groups = list(f['digitStruct'].items())
    bbox_ds = np.array(groups[0][1]).squeeze()
    names_ds = np.array(groups[1][1]).squeeze()

    data_list = []
    num_files = bbox_ds.shape[0]
    count = 0

    for objref1, objref2 in zip(bbox_ds[:10000], names_ds[:10000]):
        data_dict = {}
        # Extract image name
        names_ds = np.array(f[objref2]).squeeze()
        filename = ''.join(chr(x) for x in names_ds)
        data_dict['filename'] = filename
        # print filename
        # Extract other properties
        items1 = list(f[objref1].items())
        # Extract image label
        labels_ds = np.array(items1[1][1]).squeeze()
        try:
            label_vals = [int(f[ref][:][0, 0]) for ref in labels_ds]
        except TypeError:
            label_vals = [labels_ds]
        data_dict['labels'] = label_vals
        data_dict['length'] = len(label_vals)
        # Extract image height
        height_ds = np.array(items1[0][1]).squeeze()
        try:
            height_vals = [f[ref][:][0, 0] for ref in height_ds]
        except TypeError:
            height_vals = [height_ds]
        data_dict['height'] = height_vals
        # Extract image left coords
        left_ds = np.array(items1[2][1]).squeeze()
        try:
            left_vals = [f[ref][:][0, 0] for ref in left_ds]
        except TypeError:
            left_vals = [left_ds]
        data_dict['left'] = left_vals
        # Extract image top coords
        top_ds = np.array(items1[3][1]).squeeze()
        try:
            top_vals = [f[ref][:][0, 0] for ref in top_ds]
        except TypeError:
            top_vals = [top_ds]
        data_dict['top'] = top_vals
        # Extract image width
        width_ds = np.array(items1[4][1]).squeeze()
        try:
            width_vals = [f[ref][:][0, 0] for ref in width_ds]
        except TypeError:
            width_vals = [width_ds]
        data_dict['width'] = width_vals
        data_list.append(data_dict)
        count += 1
        print('Processed: %d/%d' % (count, num_files))
    return data_list

# 第1步：解析数据
"""
SVHN数据集用不明确的.mat格式注释。gist(https://gist.github.com/veeresht/7bf499ee6d81938f8bbdb3c6ef1855bf)提供了一个灵
活的read_process_h5脚本来将.mat文件转换为标准的json，你应该提前一步并将其转换为pascal格式，
"""
def json_to_pascal(json, filename):
    # convert json to pascal and save as csv
    pascal_list = []
    for i in json:
        for j in range(len(i['labels'])):
            pascal_list.append({'fname': i['filename'],
                                'xmin': int(i['left'][j]),
                                'xmax': int(i['left'][j] + i['width'][j]),
                                'ymin': int(i['top'][j]),
                                'ymax': int(i['top'][j] + i['height'][j]),
                                'class_id': int(i['labels'][j])})
    df_pascal = pd.DataFrame(pascal_list, dtype='str')
    df_pascal.to_csv(filename, index=False)

file_path = data_folder+'/digitStruct.mat'
p  = read_process_h5(file_path)
json_to_pascal(p, data_folder+'/pascal.csv')

## Load the trained weights file and make a copy
# Init model
task = 'svhn'
labels_path = f'{data_folder}pascal.csv'
input_format = ['class_id', 'image_name', 'xmax', 'xmin', 'ymax', 'ymin']
df = pd.read_csv(labels_path)

# 第2步：查看数据
# Explore
def viz_random_image(df):
    file = np.random.choice(df.fname)
    im = skimage.io.imread(f'{data_folder}/{file}')
    annots =  df[df.fname==file].iterrows()
    plt.figure(figsize=(6,6))
    plt.imshow(im)
    current_axis = plt.gca()
    for box in annots:
        label = box[1]['class_id']
        current_axis.add_patch(plt.Rectangle(
            (box[1]['xmin'], box[1]['ymin']), box[1]['xmax']-box[1]['xmin'],
            box[1]['ymax']-box[1]['ymin'], color='blue', fill=False, linewidth=2))
        current_axis.text(box[1]['xmin'], box[1]['ymin'], label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})
    plt.show()
# 可视化
viz_random_image(df)

# 第3步：选择策略      使用SSD检测模型
# 步骤4：加载并训练SSD模型
# 模型配置
class SVHN_Config(Config):
    batch_size = 8
    dataset_folder = data_folder
    task = task
    labels_path = labels_path
    input_format = input_format

# 实例化类
conf = SVHN_Config()
resize = Resize(height=conf.img_height, width=conf.img_width)
trans = [resize]

# 定义模型，加载权重
"""
加载预先训练过的权重。在这种情况下，将加载SSD模型在COCO数据集上训练的权重，该数据集有80个类。显然，任务只有10个类，
因此将在加载权重后重建顶层以获得正确的输出数。在init_weights函数中执行此操作。旁注：在这种情况下，每个类（边界框坐
标）的输出数量为44：4，而背景/无类别的输出数量为4。
"""
learner = SSD_finetune(conf)
learner.get_data(create_subset=True)
# SSD_Keras repo在每个epoch后保存模型，因此您可以通过将weights_destination_path行更改为等于路径来稍后加载模型
weights_destination_path=learner.init_weights()
learner.get_model(mode='training', weights_path = weights_destination_path)
model = learner.model
learner.get_input_encoder()
ssd_input_encoder = learner.ssd_input_encoder
# Training schedule definitions
adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)
model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

# Data loaders From utils 定义数据加载器
train_annotation_file=f'{conf.dataset_folder}train_pascal.csv'
val_annotation_file=f'{conf.dataset_folder}val_pascal.csv'
subset_annotation_file=f'{conf.dataset_folder}small_pascal.csv'
train_generator = learner.get_generator(conf.batch_size, trans=trans, anot_file=train_annotation_file,
                  encoder=ssd_input_encoder)
val_generator = learner.get_generator(conf.batch_size,trans=trans, anot_file=val_annotation_file,
                 returns={'processed_images','encoded_labels'}, encoder=ssd_input_encoder,val=True)
# 步骤五：训练模型
# Train
"""
训练脚本中包含了training_plot回调，以便在每个epoch后可视化随机图像
"""
learner.init_training()
histroy = learner.train(train_generator, val_generator, steps=100,epochs=80)
# normal training should go from dozens to ~10 in 1 epoch
#some times the models stalls and then is released :o

# Eval MAP (optional)
from eval_utils.average_precision_evaluator import Evaluator

# evaluate map. may need some work
class_count = 10
ev = Evaluator(learner.model,class_count, test_dataset,model_mode='training' )
./logs/ssd_20181009T1927_data_classes_1-01_loss-2.7496_val_loss-2.9780.h5
map=ev(300,300,1,data_generator_mode='resize')
map
