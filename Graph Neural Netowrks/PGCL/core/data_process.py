# -*- coding: utf-8 -*-

# Code for paper:
# [Title]  - "GCL"
# [Author] - Junbin Zhang
# [Github] - https://github.com/

import os
import sys
import numpy as np
import pandas as pd
import json
from core.config import cfg
import random
import shutil
import gc


class DataProcess:
    def __init__(self):
        self.lables_dict = {}
        self.index = 0
        self.videos_name_list = []
        self.vid_list_file = cfg.DATA_PATH + cfg.dataset + "/splits/train.split" + cfg.split + ".bundle"
        self.vid_list_file_tst = cfg.DATA_PATH + cfg.dataset + "/splits/test.split" + cfg.split + ".bundle"
        self.features_path = cfg.DATA_PATH + cfg.dataset + "/features/"
        self.gt_path = cfg.DATA_PATH + cfg.dataset + "/groundTruth/"
        self.mapping_file = cfg.DATA_PATH + cfg.dataset + "/mapping.txt"

    def get_actions_dict(self):
        actions_file_path = open(self.mapping_file, 'r')
        # label
        actions = actions_file_path.read().split('\n')[:-1]
        actions_file_path.close()
        actions_dict = dict()
        for a in actions:
            actions_dict[a.split()[1]] = int(a.split()[0])
        index2label = dict()
        for k, v in actions_dict.items():
            index2label[str(v)] = k
        cfg.label2index_dict = actions_dict
        cfg.index2label_dict = index2label

    def get_videos_name(self):
        files = os.listdir(self.gt_path)
        videos_name_path = cfg.DATA_PATH + cfg.dataset + "/videos_name.txt"
        with open(videos_name_path, 'w') as f:
            for file in files:
                video_name = file[: -4]
                f.write(video_name + '\n')
        f.close()

    def split_dataset(self):
        train_ratio = cfg.split_ratio[0]
        validation_ratio = cfg.split_ratio[1]
        test_ratio = cfg.split_ratio[2]
        # reading videos_name_dict file
        videos_name_path = cfg.DATA_PATH+cfg.dataset+"/videos_name.txt"
        videos_name = np.loadtxt(videos_name_path, dtype=str)

        train_dataset = videos_name[: int(train_ratio*len(videos_name))]
        np.savetxt(cfg.DATA_PATH+cfg.dataset+'/trainset.txt', train_dataset, fmt='%s')
        validation_dataset = videos_name[int(train_ratio*len(videos_name)): int((1-validation_ratio)*len(videos_name))]
        np.savetxt(cfg.DATA_PATH+cfg.dataset+'/validationset.txt', validation_dataset, fmt='%s')
        test_dataset = videos_name[int((1-test_ratio)*len(videos_name)): len(videos_name)]
        np.savetxt(cfg.DATA_PATH+cfg.dataset+'/testset.txt', test_dataset, fmt='%s')







    def get_label_videos_name_dict_breakfast(self):
        actions_dict = dict()
        num = 0

        name = {'coffee': 1,    # n=167
                'juice': 1,   # # n=162
                'milk': 1,    # # n=187
                'tea': 1,     # # n=184
                'cereals': 1,     # # n=184
                'friedegg': 1,   # # n=173
                'pancake': 1,     # # n=157
                'salat': 1,      #  # n=163
                'sandwich': 1,    # # n=169
                'scrambledegg': 1   #  # n=166
                }

        files = os.listdir(self.gt_path)  # 得到文件夹下的所有文件
        videos_name_path = cfg.DATA_PATH + cfg.dataset + "/videos_name.txt"
        with open(videos_name_path, 'w') as f:
            for file in files:
                if file[0] != 'P':
                    continue
                temp = file.split('_')
                if temp[3][: -4] in name:
                    video_name = file[: -4]
                    f.write(video_name + '\n')
                else:
                    continue
                actions_file_path = open(self.gt_path+file, 'r')
                # label
                actions = actions_file_path.read().split('\n')[:-1]
                actions_file_path.close()
                for a in actions:
                    if a in actions_dict:
                        continue
                    else:
                        actions_dict[a] = num
                        num += 1
        f.close()
        index2label = dict()
        for k, v in actions_dict.items():
            index2label[str(v)] = k
        cfg.label2index_dict = actions_dict
        cfg.index2label_dict = index2label
        # videos_name.append(video_name)
        # f.write(str(videos_name) + '\n')

    def get_lable_videos_name_dict_gtea(self):
        actions_dict = dict()
        num = 0
        name = {'Cheese': 1,
                'coffee': 1,
                'CofHoney': 1,
                'Hotdog': 1,
                'Pealate': 1,
                'Peanut': 1,
                'Tea': 1,
                }
        files = os.listdir(self.gt_path)  # 得到文件夹下的所有文件
        videos_name_path = cfg.DATA_PATH + cfg.dataset + "/videos_name.txt"
        with open(videos_name_path, 'w') as f:
            for file in files:
                if file[0] != 'S':
                    continue
                temp = file.split('_')
                if temp[2][: -4] in name:

                    video_name = file[: -4]
                    f.write(video_name + '\n')

                else:
                    continue
                actions_file_path = open(self.gt_path + file, 'r')
                # label
                actions = actions_file_path.read().split('\n')[:-1]
                actions_file_path.close()
                for a in actions:
                    if a in actions_dict:
                        continue
                    else:
                        actions_dict[a] = num
                        num += 1
        f.close()
        index2label = dict()
        for k, v in actions_dict.items():
            index2label[str(v)] = k
        cfg.label2index_dict = actions_dict
        cfg.index2label_dict = index2label

    def load_txt(self):
        # path = cfg.DATA_PATH + '/segmentation_fine'  # 文件夹目录
        path = cfg.DATA_PATH + '/segmentation_coarse'  # 文件夹目录
        folders = os.listdir(path)  # 得到文件夹下的所有文件
        # print(folders)
        for folder in folders:  # 遍历文件夹
            files = os.listdir(path + '/' + folder)  # 得到文件夹下的所有文件名称
            # print(str(folder) +' '+ str(len(files)))
            # i = 0
            for file in files:  # 遍历文件夹
                if '.txt' not in file:     # [-4:]
                    # print(file[-4:])
                    sourcePath = path + '/' + folder
                    # targetPath = cfg.DATA_PATH + '/xml_fine'
                    targetPath = cfg.DATA_PATH + '/xml_coarse'
                    self.copyfile(sourcePath, targetPath, file)
                    continue
                sourcePath = path + '/' + folder
                # targetPath = cfg.DATA_PATH + '/txt_fine'
                targetPath = cfg.DATA_PATH + '/txt_coarse'
                self.copyfile(sourcePath, targetPath, file)
                self.videos_name_list.append(str(file[:-4]))
                position = path + '/' + folder + '/' + file
                # print(position)
                with open(position, "r", encoding='utf-8') as f:  # 打开文件
                    line = f.readline()  # 读取文件
                    while line:
                        line_data = line.split(' ')
                        lable = line_data[1]
                        # print(lable)
                        self.get_lables_dict(lable)
                        line = f.readline()  # 读取文件
                f.close()
        # print(len(self.videos_name_list))
        self.get_videos_name_dict()
        # i += 1
        # print(i)

    def get_lables_dict(self, lable):
        if lable not in self.lables_dict:
            dict_temp = {lable: self.index}
            self.index += 1
            self.lables_dict.update(dict_temp)
        with open(cfg.DATA_PATH + '/fine_segmentation_lables_dict2.json', 'w') as f:
        # with open(cfg.DATA_PATH + '/coarse_segmentation_lables_dict.json', 'w') as f:
            json.dump(self.lables_dict, f)
        f.close()
        # print('get_lables_dict end')

    def get_videos_name_dict(self):
        files = os.listdir(cfg.DATA_PATH + '/' + 'bf_kinetics_feat')  # 得到文件夹下的所有文件名称
        # print(len(files))
        with open(cfg.DATA_PATH + '/videos_name_dict.txt', 'w') as f:
            # print(len(self.videos_name_list))
            for line in self.videos_name_list:
                for j in range(len(files)):
                    if line == files[j][:-4]:
                        f.write(line + '\n')
        f.close()

    def load_json(self, file):
        with open(file) as json_file:
            json_data = json.load(json_file)
            return json_data

    def copyfile(self, sourcePath, targetPath, filename):
        shutil.copy(sourcePath + '/' + filename, targetPath + '/' + filename)

    '''
    def load_npy(self, file):
        loadData = np.load(file)
        return loadData

    def data_load(self):
        path = '/home/cpslabzjb/zjb/projects/zjb/GCL/data/lables_dict.json'
        lables_dict = self.load_json(path)
        self.lables_dict = lables_dict
        lable_index = self.lables_dict[lable]
    
    def split_dataset1(self, train_ratio=0.6, test_ratio=0.2, validation_ratio=0.2):
        # reading videos_name_dict file
        file = cfg.DATA_PATH + '/videos_name_dict'
        if os.path.exists(file + '.txt'):
            ###self.videos_name_list = np.loadtxt(file + '.txt', dtype=np.int32)#
            with open(file + '.txt', "r", encoding='utf-8') as f:  # 打开文件
                # self.videos_name_list.append(f.readline())  # 读取文件
                line = f.readline()  # 读取文件
                while line:
                    line_data = line.split('\n')
                    self.videos_name_list.append(line_data[0])
                    line = f.readline()  # 读取文件
            f.close()
        else:
            print('no videos_name_dict.txt')
        #print(videos_name_list)
        #videos_name_list.split(',')
        # print(len(videos_name_list))
        #for i in range(0, len(videos_name_list)):
        #    self.videos_name_list.append(videos_name_list[i])
        print(self.videos_name_list)
        # print(type(self.videos_name_list))
        np.random.shuffle(self.videos_name_list)
        print(self.videos_name_list)
        print(len(self.videos_name_list))
        test_dataset = []
        validation_dataset = []
        train_dataset = []
        num = int(test_ratio * len(self.videos_name_list))
        size = len(self.videos_name_list)
        for i in range(0, num):
            index = random.randint(0, size)
            test_dataset.append(self.videos_name_list[index])
            temp = self.videos_name_list.pop(index)
            size = len(self.videos_name_list)
        #for i in range(0, len(test_dataset)):
        #    self.videos_name_list.remove(test_dataset[i])
        #print(test_dataset)
        #print(len(test_dataset))
        #print(len(self.videos_name_list))
        for i in range(0, int(validation_ratio * len(self.videos_name_list))):
            num = random.randint(0, len(self.videos_name_list))
            validation_dataset.append(self.videos_name_list[num])
        for i in range(0, len(validation_dataset)):
            self.videos_name_list.pop(i)
        #print(validation_dataset)
        #print(len(validation_dataset))
        #print(len(self.videos_name_list))
        train_dataset = self.videos_name_list
        #print(train_dataset)
        #print(len(train_dataset))
    '''

    def load_dataset(self, datasetname):
        '''
        videos = [{"name": "g", "frame": [[[0, 0], 't00'], [[1, 1], 't01'], [[2, 2], 't02'], [[3, 3], 't03']]},
                  {"name": "e", "frame": [[[3, 3], 't10'], [[4, 4], 't11'], [[5, 5], 't12'], [[6, 6], 't13']]},
                  {"name": "a", "frame": [[[6, 6], 't20'], [[7, 7], 't21'], [[8, 8], 't22'], [[9, 9], 't23']]}]
        '''
        path = cfg.DATA_PATH + '/fine_segmentation_lables_dict.json'
        self.lables_dict = self.load_json(path)

        videos = []
        path = cfg.DATA_PATH + '/' + datasetname + '/'  # 文件夹目录
        files = os.listdir(path)  # 得到文件夹下的所有文件
        for file in files:
            position = path + '/' + file
            # print(position)
            loadData = np.load(position)
            # print('loadData '+str(len(loadData)))
            i = -4
            while file[i] != '_':
                i -= 1
            name = file[i + 1: -4]
            video = {"name": name, "frame": []}
            with open(cfg.DATA_PATH + '/txt' + '/' + file[:-4] + '.txt', "r", encoding='utf-8') as f:  # 打开文件
                line = f.readline()  # 读取文件
                while line:
                    line_data = line.split(' ')
                    frame_num = line_data[0].split('-')
                    startPoint = int(frame_num[0])
                    # print('startPoint= '+str(startPoint))
                    endPoint = min(int(frame_num[1]), len(loadData))
                    # print('endPoint= ' + str(endPoint))
                    frame_lable = self.lables_dict[line_data[1]]  # line_data[1]
                    # print('frame_lable= ' + str(frame_lable))
                    # print(lable)
                    '''
                    for index in range(startPoint, endPoint):
                        frame_feature = loadData[index]
                        video["frame"].append([frame_feature, frame_lable])
                    '''
                    # frame_feature = []
                    for index in range(startPoint, endPoint):
                        frame_feature = loadData[index]
                    video["frame"].append([frame_feature, frame_lable])

                    line = f.readline()  # 读取文件
            f.close()
            # print(video)
            videos.append(video)
        #for video in videos:
        #    print(video)
        #    print(' ')
        return videos

    def load_dataset_one(self, path, file, lables_dict):
        '''
        videos = [{"name": "g", "frame": [[[0, 0], 't00'], [[1, 1], 't01'], [[2, 2], 't02'], [[3, 3], 't03']]},
                  {"name": "e", "frame": [[[3, 3], 't10'], [[4, 4], 't11'], [[5, 5], 't12'], [[6, 6], 't13']]},
                  {"name": "a", "frame": [[[6, 6], 't20'], [[7, 7], 't21'], [[8, 8], 't22'], [[9, 9], 't23']]}]
        paths = cfg.DATA_PATH + '/fine_segmentation_lables_dict.json'
        self.lables_dict = self.load_json(paths)
        '''
        self.lables_dict = lables_dict
        loadData = np.load(path + '/' + file)
        # print('loadData '+str(len(loadData)))
        i = -4
        while file[i] != '_':
            i -= 1
        name = file[: -4]     # file[i + 1: -4]
        video = {"name": name, "frame": []}
        # print(video)
        with open(cfg.DATA_PATH + '/txt_fine' + '/' + file[:-4] + '.txt', "r", encoding='utf-8') as f:  # 打开文件coarse
            line = f.readline()  # 读取文件
            while line:
                line_data = line.split(' ')
                frame_num = line_data[0].split('-')
                startPoint = int(frame_num[0])
                # print('startPoint= '+str(startPoint))
                endPoint = min(int(frame_num[1]), len(loadData))
                # print('endPoint= ' + str(endPoint))
                frame_lable = self.lables_dict[line_data[1]]  # line_data[1]
                # print('frame_lable= ' + str(frame_lable))
                # print(lable)

                for index in range(startPoint, endPoint):
                    frame_feature = loadData[index]
                    video["frame"].append([frame_feature, frame_lable])
                '''
                frame_feature = []
                for index in range(startPoint, endPoint):
                    # frame_feature = loadData[index]
                    frame_feature.append(loadData[index])
                video["frame"].append([frame_feature, frame_lable])
                '''
                line = f.readline()  # 读取文件
        f.close()
        return video


if __name__ == '__main__':
    # DataProcess().split_dataset(train_ratio=0.8, test_ratio=0.1, validation_ratio=0.1)

    # 设置类别的数量
    num_classes = 5
    # 需要转换的整数
    arr = [3]
    # 将整数转为一个10位的one hot编码
    label_one_hot = np.eye(num_classes)[arr]
    print(label_one_hot[0])
    DataProcess().load_txt()

    # videos = DataProcess().load_dataset('train')

    '''
    path = "/home/cpslabzjb/zjb/projects/zjb/GCL/data/BreakfastActions"  # 文件夹目录
    folders = os.listdir(path)  # 得到文件夹下的所有文件
    for folder in folders:  # 遍历文件夹
        files = os.listdir(path + '/' + folder)  # 得到文件夹下的所有文件名称
        print(files)
    '''
    print('end')
