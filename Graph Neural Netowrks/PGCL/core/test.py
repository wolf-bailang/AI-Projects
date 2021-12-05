# -*- coding: utf-8 -*-

# Code for paper:
# [Title]  - "GCL"
# [Author] - Junbin Zhang
# [Github] - https://github.com/

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import numpy as np
from core.config import cfg
from torch.utils.data import DataLoader
import os
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch


@torch.no_grad()
def test(test_loader, model, device='cuda'):
    model.eval()
    # print('test_loader=' + str(len(test_loader)))
    for batch in test_loader:
        torch.cuda.empty_cache()
        batch.to(device)
        logits = model(batch)
        pred = logits[batch.node_label_index].max(1)[1]
        acc = pred.eq(batch.node_label).sum().item()
        total = batch.node_label_index.shape[0]
        acc /= total
    return acc, pred

@torch.no_grad()
def test_one(model, device='cuda', name='train'):
    model.eval()
    model.to(device)
    # acc = 0.0
    if name == 'train':
        batch_size = cfg.batch_size[0]
    if name == 'validation':
        batch_size = cfg.batch_size[1]
    if name == 'test':
        batch_size = cfg.batch_size[2]
    files = os.listdir(cfg.DATA_PATH+cfg.dataset+'/'+name+'_batch/')  # 得到文件夹下的所有文件
    for i in range(len(files)):
        DeepSNAP_graph_dataset = torch.load(cfg.DATA_PATH+cfg.dataset+'/'+name+'_batch/'+str(i)+'.bin')
        test_loader = DataLoader(DeepSNAP_graph_dataset, collate_fn=Batch.collate(), batch_size=batch_size, shuffle=False)
        for batch in test_loader:
            torch.cuda.empty_cache()
            batch.to(device)
            logits = model(batch)
            pred = logits[batch.node_label_index].max(1)[1]
            acc = pred.eq(batch.node_label).sum().item()
            total = batch.node_label_index.shape[0]
            acc /= total
    return acc

@torch.no_grad()
def test_three(model, device='cuda', name='train'):
    model.eval()
    model.to(device)
    # acc = 0.0
    if name == 'train':
        batch_size = cfg.batch_size[1]
    if name == 'validation':
        batch_size = cfg.batch_size[1]
    if name == 'test':
        batch_size = cfg.batch_size[2]
    files = os.listdir('/mnt/TOSHIBAdata2T/'+cfg.dataset+'/three/'+name+'_graph/')  # 得到文件夹下的所有文件
    # files.sort(key=None, reverse=False)
    for i in range(0, len(files), batch_size):
        graphs = []
        if name == 'train':
            if i < batch_size*(int(len(files) / batch_size)):
                for j in range(batch_size):
                    graphs.append(
                        torch.load('/mnt/TOSHIBAdata2T/'+cfg.dataset+'/three/'+name+'_graph/' + str(files[i + j])))  # +'.bin'
            else:
                for j in range(len(files) - (batch_size*(int(len(files) / batch_size)))):
                    graphs.append(torch.load('/mnt/TOSHIBAdata2T/'+cfg.dataset+'/three/'+name+'_graph/' + str(files[i + j])))
        else:
            graphs = torch.load('/mnt/TOSHIBAdata2T/'+cfg.dataset+'/three/'+name+'_graph/'+str(files[i]))   # +'.bin'
        DeepSNAP_graph_dataset = GraphDataset(graphs=graphs, task='node', minimum_node_per_graph=0)
        test_loader = DataLoader(DeepSNAP_graph_dataset, collate_fn=Batch.collate(), batch_size=batch_size, shuffle=False)
        for batch in test_loader:
            torch.cuda.empty_cache()
            batch.to(device)
            logits = model(batch)
            pred = logits[batch.node_label_index].max(1)[1]
            acc = pred.eq(batch.node_label).sum().item()
            total = batch.node_label_index.shape[0]
            acc /= total
    return acc

@torch.no_grad()
def predict(test_loader, model, device='cuda'):
    model.eval()
    # with torch.no_grad():
    model.to(device)
    import time
    time_start = time.time()
    for batch in test_loader:
        batch.to(device)
        logits = model(batch)
        # print(logits)
        # tensor([[ -1.9995, -14.0217,  -8.5506,  ..., -14.7283, -14.3700, -13.6380],
        #         [ -1.7551, -17.4630, -10.3406,  ..., -18.3565, -17.9021, -16.9502],
        #         [ -1.9145, -15.1572,  -9.2544,  ..., -15.8375, -15.4975, -14.6809],
        #         ...,
        #         [ -3.0307,  -5.7998,  -5.5706,  ...,  -5.6701,  -5.6205,  -5.5863],
        #         [ -2.9672,  -5.8683,  -5.5883,  ...,  -5.6966,  -5.6806,  -5.5979],
        #         [ -2.6783,  -6.0715,  -5.6673,  ...,  -5.8317,  -5.8361,  -5.7243]],
        #        device='cuda:0', grad_fn=<LogSoftmaxBackward>)
        # torch.Size([1024, 178])

        pred = logits[batch.node_label_index].max(1)[1]
        # print(pred)
        # tensor([  0, 159,   0,  ...,   0,   0,   0], device='cuda:0')
        # torch.Size([1024])
        recognition = []
        for index in pred.cpu().detach().numpy():
            recognition.append(cfg.index2label_dict[str(int(index))])
        f_name = batch.Name[0]
        f_ptr = open(cfg.DATA_PATH + cfg.dataset + "/results/" + f_name, "w")
        f_ptr.write("### Frame level recognition: ###\n")
        f_ptr.write(' '.join(recognition))
        f_ptr.close()
    time_end = time.time()

@torch.no_grad()
def predict_one(model, device='cuda',name='train'):
    model.eval()
    model.to(device)
    #import time
    #time_start = time.time()
    if name == 'train':
        batch_size = cfg.batch_size[0]
    if name == 'validation':
        batch_size = cfg.batch_size[1]
    if name == 'test':
        batch_size = cfg.batch_size[2]
    files = os.listdir(cfg.DATA_PATH + cfg.dataset + '/' + name + '_batch/')  # 得到文件夹下的所有文件
    for i in range(len(files)):
        DeepSNAP_graph_dataset = torch.load(cfg.DATA_PATH + cfg.dataset + '/' + name + '_batch/' + str(i) + '.bin')
        test_loader = DataLoader(DeepSNAP_graph_dataset, collate_fn=Batch.collate(), batch_size=batch_size,
                                 shuffle=False)
        for batch in test_loader:
            torch.cuda.empty_cache()
            batch.to(device)
            logits = model(batch)
            # print(logits)
            # tensor([[ -1.9995, -14.0217,  -8.5506,  ..., -14.7283, -14.3700, -13.6380],
            #         [ -1.7551, -17.4630, -10.3406,  ..., -18.3565, -17.9021, -16.9502],
            #         [ -1.9145, -15.1572,  -9.2544,  ..., -15.8375, -15.4975, -14.6809],
            #         ...,
            #         [ -3.0307,  -5.7998,  -5.5706,  ...,  -5.6701,  -5.6205,  -5.5863],
            #         [ -2.9672,  -5.8683,  -5.5883,  ...,  -5.6966,  -5.6806,  -5.5979],
            #         [ -2.6783,  -6.0715,  -5.6673,  ...,  -5.8317,  -5.8361,  -5.7243]],
            #        device='cuda:0', grad_fn=<LogSoftmaxBackward>)
            # torch.Size([1024, 178])

            pred = logits[batch.node_label_index].max(1)[1]
            # print(pred)
            # tensor([  0, 159,   0,  ...,   0,   0,   0], device='cuda:0')
            # torch.Size([1024])
            recognition = []
            for index in pred:
                recognition.append(cfg.index2label_dict[str(int(index))])
            f_name = batch.Name[0]
            f_ptr = open(cfg.DATA_PATH+cfg.dataset+"/results/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()
        #time_end = time.time()

@torch.no_grad()
def predict_three(model, device='cuda', name='train'):
    model.eval()
    model.to(device)
    #import time
    #time_start = time.time()
    if name == 'train':
        batch_size = cfg.batch_size[0]
    if name == 'validation':
        batch_size = cfg.batch_size[1]
    if name == 'test':
        batch_size = cfg.batch_size[2]
    files = os.listdir('/mnt/TOSHIBAdata2T/'+cfg.dataset+'/three/' + name + '_graph/')  # 得到文件夹下的所有文件
    # files.sort(key=None, reverse=False)
    for i in range(len(files)):
        graphs = torch.load('/mnt/TOSHIBAdata2T/'+cfg.dataset+'/three/' + name + '_graph/' + str(files[i]))  # + '.bin'
        DeepSNAP_graph_dataset = GraphDataset(graphs=graphs, task='node', minimum_node_per_graph=0)
        test_loader = DataLoader(DeepSNAP_graph_dataset, collate_fn=Batch.collate(), batch_size=batch_size,
                                 shuffle=False)
        for batch in test_loader:
            torch.cuda.empty_cache()
            batch.to(device)
            logits = model(batch)
            # print(logits)
            # tensor([[ -1.9995, -14.0217,  -8.5506,  ..., -14.7283, -14.3700, -13.6380],
            #         [ -1.7551, -17.4630, -10.3406,  ..., -18.3565, -17.9021, -16.9502],
            #         [ -1.9145, -15.1572,  -9.2544,  ..., -15.8375, -15.4975, -14.6809],
            #         ...,
            #         [ -3.0307,  -5.7998,  -5.5706,  ...,  -5.6701,  -5.6205,  -5.5863],
            #         [ -2.9672,  -5.8683,  -5.5883,  ...,  -5.6966,  -5.6806,  -5.5979],
            #         [ -2.6783,  -6.0715,  -5.6673,  ...,  -5.8317,  -5.8361,  -5.7243]],
            #        device='cuda:0', grad_fn=<LogSoftmaxBackward>)
            # torch.Size([1024, 178])

            pred = logits[batch.node_label_index].max(1)[1]
            # print(pred)
            # tensor([  0, 159,   0,  ...,   0,   0,   0], device='cuda:0')
            # torch.Size([1024])
            # print(pred.cpu().detach().numpy())
            recognition = []
            for index in pred.cpu().detach().numpy():
                recognition.append(cfg.index2label_dict[str(int(index))])
            f_name = batch.Name[0]
            f_ptr = open(cfg.DATA_PATH+cfg.dataset+"/results/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()
        #time_end = time.time()


if __name__ == '__main__':
    print('test')
