from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from math import ceil
import numpy as np
import pandas as pd
from time import time
import datetime
import os
from matplotlib import pyplot as plt
import re 
import pdb
import json
import sys
import os

with open('json_config.json') as f:
    json_conf = json.load(f)
    
SSD_DIR = os.path.abspath(json_conf["ssd_folder"]) # add here mask RCNN path
sys.path.append(SSD_DIR)

from models.keras_ssd7 import build_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
#from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_patch_sampling_ops import RandomMaxCropFixedAR
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation,SSDPhotometricDistortions,SSDExpand,SSDRandomCrop
from data_generator.data_augmentation_chain_variable_input_size import *
from data_generator.data_augmentation_chain_constant_input_size import *
from data_generator.object_detection_2d_patch_sampling_ops import Crop,Pad,CropPad,PatchCoordinateGenerator, RandomPatch, RandomPatchInf

from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

from eval_utils.coco_utils import get_coco_category_maps, predict_all_to_json

from sklearn.model_selection import train_test_split
import glob
import cv2
#### weight sampling imports
from models.keras_ssd300 import ssd_300
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from data_generator.object_detection_2d_geometric_ops import Resize
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,TensorBoard

import h5py
import shutil
from misc_utils.tensor_sampling_utils import sample_tensors


class Config:
    # model config from original repo
    batch_size = 8
    
    img_height = 300 # Height of the input images
    img_width = 300 # Width of the input images
    img_channels = 3 # Number of color channels of the input images
    #n_classes = num_classes
    normalize_coords = True # Whether or not the model is supposed to use coordinates relative to the image size
    two_boxes_for_ar1 = True # Whether or not you want to generate two anchor boxes for aspect ratio 1


    subtract_mean = [123, 117, 104] # The per-channel mean of the images in the dataset
    swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we should set this to `True`, but weirdly the results are better without swapping.

    scales = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets.
    # scales = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets.
    aspect_ratios = [[1.0, 2.0, 0.5],[1.0, 2.0, 0.5, 3.0, 1.0/3.0],[1.0, 2.0, 0.5, 3.0, 1.0/3.0],[1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                     [1.0, 2.0, 0.5],[1.0, 2.0, 0.5]] # The anchor box aspect ratios used in the original SSD300; the order matters
    steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
    offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
    clip_boxes = False # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
    variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are scaled as in the original implementation   

# Callback to display the target and prediciton

import keras.callbacks

class SSD_finetune:
    # a wrapper for SSD_Keras libray.
    
    def __init__(self, config):
        self.config = config
        self.val_generator = ''
        
    def get_data(self, create_subset=False):
        # first, load
        # create_subset: used to create small subser, say 100 images, to overfit
        
        print('data loading and preperations')
        df = pd.read_csv(self.config.labels_path)
        self.df =df
        self.classes = np.append([0],np.sort(df.class_id.unique()))
        #classes = np.append([0],['zero','one','two','three','four','five','six','seven',])

        if isinstance(df.class_id.unique()[0],str): # add class_id numeric if classes are strings
            print('changing strings')
            digit_dict = {j:i for i,j in enumerate(self.classes)}
            id2digit = {i:j for i,j in enumerate(self.classes)}
            cols = df.columns
            df['class_name'] = df.class_id
            df.class_id = df.class_id.apply(lambda x:digit_dict[x])
            df.to_csv(self.config.labels_path,index=False)
            
        # make splits an save files
        id_col = df.columns[np.where(np.array(self.config.input_format)=='image_name')][0]
        files = np.unique(df[id_col].values)
       
            
        self.n_classes = len(self.classes)-1
        print('class_ids',self.classes,' should be numeric')
        print('input format:', self.config.input_format)
        print(df.head(2))
        
        train_annotation_file=f'{self.config.dataset_folder}train_pascal.csv'
        val_annotation_file=f'{self.config.dataset_folder}val_pascal.csv'

        train_files, val_files = train_test_split(files, random_state=0,test_size=0.2)
        
        if create_subset:
             # take 100 (?) first files of val
            
            subset_file = f'{self.config.dataset_folder}small_pascal.csv'
            print(f'Create subset of 20 files in {subset_file}')
            df[df[id_col].isin(val_files[:20])].iloc[:,:6].to_csv(subset_file,index=False)
        
        print('split to',len(train_files),' train files',len(val_files),' val files')
        df[df[id_col].isin(train_files)].iloc[:,:6].to_csv(train_annotation_file, index=False)
        df[df[id_col].isin(val_files)].iloc[:,:6].to_csv(val_annotation_file, index=False)
        if 'class_name' in self.df.columns:
            self.id2digit={i:j for i,j in zip(self.df.drop_duplicates('class_name').class_id,self.df.drop_duplicates('class_name').class_name)}
        else:
            self.id2digit={i:j for i,j in zip(self.classes,self.classes)}
        
    def init_weights(self, classes=''):
        # saves coco trained weights in destination # that can be loaded by model
        # if passing classes, original coco classes may be passed, e.g 1=person,9=boat
            #looky here https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb
        print('\nloading weights')
        #pdb.set_trace()
        if classes=='':
            classes = self.classes
            
        weights_source_path = f'{json_conf["ssd_base_weights"]}'# /home/gidish/models/VGG_coco_SSD_300x300_iter_400000.h5'

        weights_destination_path = f'{json_conf["ssd_base_weights"]}_subsampled_{len(classes)}_classes.h5'
        shutil.copy(weights_source_path, weights_destination_path)
        weights_source_file = h5py.File(weights_source_path, 'r')
        weights_destination_file = h5py.File(weights_destination_path)
        
        classifier_names = ['conv4_3_norm_mbox_conf','fc7_mbox_conf', 'conv6_2_mbox_conf', 'conv7_2_mbox_conf',
                    'conv8_2_mbox_conf','conv9_2_mbox_conf']
        
        n_classes_source = 81

        classes_of_interest = list(classes)
        print('classes are:',classes_of_interest)
        subsampling_indices = []
        for i in range(int(324/n_classes_source)):
            indices = np.array(classes_of_interest) + i * n_classes_source
            subsampling_indices.append(indices)
        subsampling_indices = list(np.concatenate(subsampling_indices))
        print('last layer indicies',subsampling_indices)
        
        for name in classifier_names:
            # Get the trained weights for this layer from the source HDF5 weights file.
            kernel = weights_source_file[name][name]['kernel:0'].value
            bias = weights_source_file[name][name]['bias:0'].value

            # Get the shape of the kernel. We're interested in sub-sampling
            # the last dimension, 'o'.
            height, width, in_channels, out_channels = kernel.shape

            # Compute the indices of the elements we want to sub-sample.
            # Keep in mind that each classification predictor layer predicts multiple
            # bounding boxes for every spatial location, so we want to sub-sample
            # the relevant classes for each of these boxes.
            if isinstance(classes_of_interest, (list, tuple)):
                subsampling_indices = []
                for i in range(int(out_channels/n_classes_source)):
                    indices = np.array(classes_of_interest) + i * n_classes_source
                    subsampling_indices.append(indices)
                subsampling_indices = list(np.concatenate(subsampling_indices))
            elif isinstance(classes_of_interest, int):
                subsampling_indices = int(classes_of_interest * (out_channels/n_classes_source))
            else:
                raise ValueError("`classes_of_interest` must be either an integer or a list/tuple.")

            # Sub-sample the kernel and bias.
            # The `sample_tensors()` function used below provides extensive
            # documentation, so don't hesitate to read it if you want to know
            # what exactly is going on here.
            new_kernel, new_bias = sample_tensors(weights_list=[kernel, bias],
                                                  sampling_instructions=[height, width, in_channels, subsampling_indices],
                                                  axes=[[3]], # The one bias dimension corresponds to the last kernel dimension.
                                                  init=['gaussian', 'zeros'],
                                                  mean=0.0,
                                                  stddev=0.005)

            # Delete the old weights from the destination file.
            del weights_destination_file[name][name]['kernel:0']
            del weights_destination_file[name][name]['bias:0']
            # Create new datasets for the sub-sampled weights.
            weights_destination_file[name][name].create_dataset(name='kernel:0', data=new_kernel)
            weights_destination_file[name][name].create_dataset(name='bias:0', data=new_bias)

        # Make sure all data is written to our output file before this sub-routine exits.
        weights_destination_file.flush()
        conv4_3_norm_mbox_conf_kernel = weights_destination_file[classifier_names[0]][classifier_names[0]]['kernel:0']
        conv4_3_norm_mbox_conf_bias = weights_destination_file[classifier_names[0]][classifier_names[0]]['bias:0']
        
        print("Shape of the '{}' weights:".format(classifier_names[0]))
        print()
        print("kernel:\t", conv4_3_norm_mbox_conf_kernel.shape)
        print("bias:\t", conv4_3_norm_mbox_conf_bias.shape)
        return weights_destination_path
    
    def get_model(self, mode='inference', weights_path='', n_classes='', id2digit=''):
        #
        # n_classes, id2digit: for inference
        config=self.config
        if n_classes: # inference setting
            self.n_classes = n_classes
        
        if id2digit:
            self.id2digit = id2digit
            
        self.model = ssd_300(image_size=(config.img_height, config.img_width, config.img_channels),
                    n_classes=self.n_classes,
                    mode=mode,
                    l2_regularization=0.0005,
                    scales=config.scales,
                    aspect_ratios_per_layer=config.aspect_ratios,
                    two_boxes_for_ar1=config.two_boxes_for_ar1,
                    steps=config.steps,
                    offsets=config.offsets,
                    clip_boxes=config.clip_boxes,
                    variances=config.variances,
                    normalize_coords=config.normalize_coords,
                    subtract_mean=config.subtract_mean, #
                    divide_by_stddev=None,      #
                    swap_channels=config.swap_channels,
                    confidence_thresh=0.5, #
                    iou_threshold=0.45,
                    top_k=200,
                    nms_max_output_size=400,
                    return_predictor_sizes=False)

        if weights_path:
            print(f'Loading weights from {weights_path}')
            self.model.load_weights(weights_path, by_name=True)
            self.weights_path = weights_path

        #adam = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        #sgd = SGD(lr=0.001, momentum=0.9, decay=0.0, nesterov=False)
        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

        ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

        self.model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

    def get_input_encoder(self):
        config=self.config
        
        # SSD 300 layers
        predictor_sizes = [self.model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   self.model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   self.model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   self.model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   self.model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   self.model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]
        
        self.ssd_input_encoder = SSDInputEncoder(img_height=config.img_height,
                                        img_width=config.img_width,
                                        n_classes=self.n_classes,
                                        predictor_sizes=predictor_sizes,
                                        scales=config.scales,
                                        aspect_ratios_per_layer=config.aspect_ratios,
                                        two_boxes_for_ar1=config.two_boxes_for_ar1,
                                        steps=config.steps,
                                        offsets=config.offsets,
                                        clip_boxes=config.clip_boxes,
                                        variances=config.variances,
                                        matching_type='multi',
                                        pos_iou_threshold=0.5,
                                        neg_iou_limit=0.5,
                                        normalize_coords=config.normalize_coords)
        print(f'created encoder with {self.n_classes} classes')
    
    def prepare_ds(self, dataset_folder,annotation_file=''):
        # prepares dataset from folder and annotations file
        ds = DataGenerator()
        ds.parse_csv(images_dir = dataset_folder,labels_filename = annotation_file,input_format = self.config.input_format, 
                     include_classes='all',random_sample=False)
        return ds

    def get_generator(self, batch_size, trans, anot_file, encoder='',returns={'processed_images','encoded_labels'},val=False):
        
        dataset = self.prepare_ds(self.config.dataset_folder,anot_file)
        
        if encoder:
            kwargs = {'label_encoder':encoder}
        else:
            kwargs = {}
        print(f'Loaded {dataset.get_dataset_size()} images, with {trans} transformations')
        generator = dataset.generate(batch_size=batch_size, shuffle=True, transformations=trans,
                    returns=returns, keep_images_without_gt=True,**kwargs)
        if val:
            self.val_generator=generator
            
        return generator
    
    def training_plot(self, epoch, logs):
        # plots results on epoch end

        if self.val_generator:
            imgs,gt = next(self.val_generator)
            y_pred = self.model.predict(np.expand_dims(imgs[0],0))
            y_pred_decoded = decode_detections(y_pred,
                                       confidence_thresh=0.5,
                                       iou_threshold=0.45,
                                       top_k=200,
                                       input_coords='centroids',
                                       normalize_coords=True,
                                       img_height=self.config.img_height,
                                       img_width=self.config.img_width)

            plt.figure(figsize=(6,6))
            plt.imshow(imgs[0])

            current_axis = plt.gca()

            for box in y_pred_decoded[0]:
                class_id = box[0]
                confidence = box[1]
                xmin,ymin,xmax,ymax = box[2],box[3],box[4],box[5]

                label = '{}: {:.2f}'.format(self.id2digit[class_id], confidence)
                current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='blue', fill=False, linewidth=2))  
                current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'blue', 'alpha':1.0})
            plt.show()
        else:
            print('no val generator defined')
        
    def init_training(self):
        # inits callbcaks for training
        now = datetime.datetime.now()
        task = re.sub("/", '_', self.config.task)
        
        training_plot_cb = keras.callbacks.LambdaCallback(on_epoch_end=self.training_plot)
        
        checkpoint = ModelCheckpoint(filepath='./logs/ssd_{:%Y%m%dT%H%M}'.format(now)+f'_{task}_classes_{self.n_classes}-'+'{epoch:02d}_loss-{loss:.4f}\
            _val_loss-{val_loss:.4f}.h5',monitor='val_loss', verbose=1,save_best_only=True, save_weights_only=True, mode='auto',period=1)

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0, patience=10, verbose=1)

        reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss',factor=0.5, patience=10, verbose=1,epsilon=0.001, cooldown=2,  min_lr=0.00001)

        logdir = os.path.join( "./logs/{}{:%Y%m%dT%H%M}".format(self.config.task,now))

        tensor_board=TensorBoard(write_images=True,log_dir=logdir, histogram_freq=0, write_graph=True)

        self.callbacks_1 = [early_stopping,tensor_board] # reduce_learning_rate
        print('changing')
        self.callbacks_2 = [reduce_learning_rate,tensor_board, checkpoint, training_plot_cb]
        
    def train(self, train_generator, val_generator, steps=250, epochs=15):

        if self.model:
            self.model.fit_generator(generator=train_generator,steps_per_epoch=steps,epochs=epochs,
                    validation_data=val_generator, callbacks=self.callbacks_2,  validation_steps=ceil(15)) 


