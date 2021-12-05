# -*- coding: utf-8 -*-

# Code for paper:
# [Title]  - "GCL"
# [Author] - Junbin Zhang
# [Github] - https://github.com/

import os
import sys
import time
import copy
import json
import torch
import numpy as np
import gc
import warnings
warnings.filterwarnings("ignore")
############################################################
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T  # 在计算机视觉领域是一种很常见的数据增强。
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from terminaltables import AsciiTable

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR


import core.utils as utils
import ASFormer.eval as ASFormer
import core.train as train
import core.test as test
from core.model import GCN, build_optimizer
from core.data_process import DataProcess
from core.graph import GraphGenerate_pyg, GraphGenerate_nx, NodeEmebedding
from core.config import cfg
from core.node_classification_planetoid_tensor import GNN
# from core.node2vec import NodeEmebeddingNode2Vec
# import snap
#############################################################


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_id
    worker_init_fn = None
    if cfg.seed >= 0:
        utils.set_seed(cfg.seed)
        worker_init_fn = np.random.seed(cfg.seed)
    # utils.set_path(cfg)
    # utils.save_config(cfg)

    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get labels
    DataProcess().get_actions_dict()
    cfg.num_classes = len(cfg.label2index_dict)
    print(cfg.label2index_dict)
    print(cfg.num_classes)
    # {'take': 0, 'open': 1, 'pour': 2, 'close': 3, 'shake': 4, 'scoop': 5, 'stir': 6, 'put': 7, 'fold': 8, 'spread': 9,
    #  'background': 10}
    DataProcess().get_videos_name()
    DataProcess().split_dataset()
    '''
    # nx
    GraphGenerate_nx().video_to_graph_nx('train')
    GraphGenerate_nx().video_to_graph_nx('validation')
    GraphGenerate_nx().video_to_graph_nx('test')
    
    # pyg
    # GraphGenerate_pyg().frame_to_graph_one_pyg()
    dataset = GraphGenerate_pyg().graph_to_dataset_pyg()
    print(dataset[0])
    # print(dataset[1])
    '''
    # cfg.num_node_features = 2048 + 128

    # Data
    print('==> Preparing data..')
    '''
    # trainset
    graphs = []
    files = os.listdir(cfg.DATA_PATH+cfg.dataset+'/trainset/')  # 得到文件夹下的所有文件
    for i in range(len(files)):
        graphs.append(torch.load(cfg.DATA_PATH+cfg.dataset+'/trainset/'+str(files[i])))
    # validationset
    files = os.listdir(cfg.DATA_PATH + cfg.dataset + '/validationset/')  # 得到文件夹下的所有文件
    for i in range(len(files)):
        graphs.append(torch.load(cfg.DATA_PATH + cfg.dataset + '/validationset/' + str(files[i])))
    # testset
    files = os.listdir(cfg.DATA_PATH + cfg.dataset + '/testset/')  # 得到文件夹下的所有文件
    for i in range(len(files)):
        graphs.append(torch.load(cfg.DATA_PATH + cfg.dataset + '/testset/' + str(files[i])))
    DeepSNAP_graph_dataset = GraphDataset(graphs=graphs, task='node', minimum_node_per_graph=0)
    dataset_train, dataset_val, dataset_test = DeepSNAP_graph_dataset.split(transductive=False, split_ratio=[0.8, 0.1, 0.1])
    train_loader = DataLoader(dataset_train, collate_fn=Batch.collate(), batch_size=cfg.batch_size[0], shuffle=False)
    val_loader = DataLoader(dataset_val, collate_fn=Batch.collate(), batch_size=cfg.batch_size[1], shuffle=False)
    test_loader = DataLoader(dataset_test, collate_fn=Batch.collate(), batch_size=cfg.batch_size[2], shuffle=False)
    print("train dataset: {}".format(dataset_train))
    print("validation dataset: {}".format(dataset_val))
    print("test dataset: {}".format(dataset_test))    
    '''

    ''''''
    # trainset
    graphs = []
    files = os.listdir(cfg.DATA_PATH + cfg.dataset + '/trainset/')  # 得到文件夹下的所有文件
    for i in range(len(files)):
        graphs.append(torch.load(cfg.DATA_PATH + cfg.dataset + '/trainset/' + str(files[i])))
    DeepSNAP_graph_dataset = GraphDataset(graphs=graphs, task='node', minimum_node_per_graph=0)
    train_loader = DataLoader(DeepSNAP_graph_dataset, collate_fn=Batch.collate(), batch_size=cfg.batch_size[0], shuffle=False)
    # validationset
    graphs = []
    files = os.listdir(cfg.DATA_PATH + cfg.dataset + '/validationset/')  # 得到文件夹下的所有文件
    for i in range(len(files)):
        graphs.append(torch.load(cfg.DATA_PATH + cfg.dataset + '/validationset/' + str(files[i])))
    DeepSNAP_graph_dataset = GraphDataset(graphs=graphs, task='node', minimum_node_per_graph=0)
    val_loader = DataLoader(DeepSNAP_graph_dataset, collate_fn=Batch.collate(), batch_size=cfg.batch_size[1], shuffle=False)
    # testset
    graphs = []
    files = os.listdir(cfg.DATA_PATH + cfg.dataset + '/testset/')  # 得到文件夹下的所有文件
    for i in range(len(files)):
        graphs.append(torch.load(cfg.DATA_PATH + cfg.dataset + '/testset/' + str(files[i])))
    DeepSNAP_graph_dataset = GraphDataset(graphs=graphs, task='node', minimum_node_per_graph=0)
    test_loader = DataLoader(DeepSNAP_graph_dataset, collate_fn=Batch.collate(), batch_size=cfg.batch_size[2], shuffle=False)
    del graphs
    del files
    del DeepSNAP_graph_dataset
    gc.collect()


    '''
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=None)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)
    '''

    model = GNN(cfg.num_node_features, cfg.hidden_dim, cfg.num_classes, cfg).to(cfg.device)
    if cfg.device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    scheduler, optimizer = build_optimizer(cfg, model.parameters())
    print('Model Size: ', sum(parameter.numel() for parameter in model.parameters()))
    # cfg.mode = 'test'
    if cfg.mode == 'train':
        print('=> training model...')
        train.train(model, optimizer, train_loader, val_loader, test_loader, cfg, cfg.num_node_features,
                    cfg.num_classes, cfg.device)
        # print('train end')
        cfg.mode = 'test'

    if cfg.mode == 'test':
        # print('=> loading model: {}'.format('best_model.pth'))
        model.load_state_dict(torch.load(os.path.join(cfg.best_model_path, 'best_model.pth')))
        print('=> testing model...')
        test_acc = test.test(test_loader, model, cfg.device)
        print('test_acc = '+str(test_acc[0]))
        # print('test end')
        cfg.mode = 'predict'

    if cfg.mode == 'predict':
        # print('=> loading model: {}'.format('best_model.pth'))
        model.load_state_dict(torch.load(os.path.join(cfg.best_model_path, 'best_model.pth')))
        print('=> predict model...')
        # predict_acc, pred = \
        test.predict(test_loader, model, cfg.device)
        # print('predict_acc = '+str(predict_acc))
        cfg.mode = 'eval'
        # print('predict end')

    if cfg.mode == 'eval':
        acc_all = 0.
        edit_all = 0.
        f1s_all = [0., 0., 0.]
        recog_path = cfg.DATA_PATH + cfg.dataset + "/results/"
        file_list = cfg.DATA_PATH + cfg.dataset + "/groundTruth/P50_webcam02_P50_coffee.txt"
        acc_all, edit_all, f1s_all = ASFormer.func_eval(cfg.dataset, recog_path, file_list)
        print("Acc: %.4f  Edit: %.4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
    ''''''

    '''
    # Mean aggregation
    best_model = None
    best_val = 0
    
    model = GNN(cfg.num_node_features, cfg.hidden_dim, cfg.num_classes, cfg).to(cfg.device)
    scheduler, optimizer = build_optimizer(cfg, model.parameters())

    if cfg.MODE == 'train':
        for epoch in range(cfg.epochs):
            loss = train.snap_train(model, optimizer, train_loader)
            accs, best_model, best_val = test(model, hetero_graph, [train_loader, val_loader, test_loader], best_model, best_val)
            print(
                f"Epoch {epoch + 1}: loss {round(loss, 5)}, "
                f"train micro {round(accs[0][0] * 100, 2)}%, train macro {round(accs[0][1] * 100, 2)}%, "
                f"valid micro {round(accs[1][0] * 100, 2)}%, valid macro {round(accs[1][1] * 100, 2)}%, "
                f"test micro {round(accs[2][0] * 100, 2)}%, test macro {round(accs[2][1] * 100, 2)}%"
            )
        print('train end')
    if cfg.MODE == 'test':
        best_accs, _, _ = test.snap_test(best_model, hetero_graph, [train_idx, val_idx, test_idx])
        print(
            f"Best model: "
            f"train micro {round(best_accs[0][0] * 100, 2)}%, train macro {round(best_accs[0][1] * 100, 2)}%, "
            f"valid micro {round(best_accs[1][0] * 100, 2)}%, valid macro {round(best_accs[1][1] * 100, 2)}%, "
            f"test micro {round(best_accs[2][0] * 100, 2)}%, test macro {round(best_accs[2][1] * 100, 2)}%"
        )
        print('test end')
    '''

if __name__ == '__main__':
    print("PyTorch has version {}".format(torch.__version__))
    if cfg.mode == 'test':
        # cfg.MODEL_FILE = cfg.model_dir
        print('test')
    else:
        print('train')
    print(AsciiTable([['CoLA - Compare to Localize Actions']]).table)
    main()
