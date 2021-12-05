# -*- coding: utf-8 -*-

# Code for paper:
# [Title]  - "GCL"
# [Author] - Junbin Zhang
# [Github] - https://github.com/

import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn
import torch.optim as optim
from torch_geometric.data import Data

from torch.utils.data import DataLoader
from torch_geometric.datasets import Planetoid, TUDataset
from sklearn.metrics import *
from torch.nn import Sequential, Linear, ReLU
from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch

from core.data_process import DataProcess
from core.graph import NodeEmebedding
from core.config import cfg
import os
import sys
import core.test as test
import gc


def snap_train(model, optimizer, train_loader, device):
    model.train()
    for batch in train_loader:
        batch.to(cfg.device)
        optimizer.zero_grad()
        preds = model(cfg.num_node_feature, batch.edge_index)
        loss = model.loss(preds, batch.node_label)
        loss.backward()
        optimizer.step()
    return loss.item()


def train(model, optimizer, train_loader, val_loader, test_loader, cfg, num_node_features, num_classes, device="cpu"):
    best_model_para_dict = None
    best_val = 0
    best_loss = 1000
    best = []

    for epoch in range(cfg.epochs):
        torch.cuda.empty_cache()
        total_loss = 0
        model.train()
        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.node_label
            '''
            print('pred')
            print(np.shape(pred[batch.node_label_index]))
            print(pred[batch.node_label_index])
            print(np.shape(label))
            '''
            loss = model.loss(pred[batch.node_label_index], label)
            total_loss += float(loss)
            loss.backward()
            optimizer.step()
        train_acc, _ = test.test(train_loader, model, device)
        val_acc, _ = test.test(val_loader, model, device)
        test_acc, _ = test.test(test_loader, model, device)

        print("Epoch: {} Train: {:.4f} Validation: {:.4f} Test: {:.4f} Loss: {:.4f}".format(
            epoch + 1, train_acc*100, val_acc*100, test_acc*100, total_loss))
        # if best_loss > total_loss:
        #     best_loss = total_loss
        if val_acc > best_val:
            best_val = val_acc
            best = [epoch + 1, train_acc*100, val_acc*100, test_acc*100, total_loss]
            # best_model = copy.deepcopy(model)
            best_model_para_dict = model.state_dict()

    torch.save(best_model_para_dict, os.path.join(cfg.best_model_path, 'best_model.pth'))
    print("best_model   Epoch {}:  Train: {:.4f}  Validation: {:.4f}  Test: {:.4f} Loss: {:.4f}".format(
        best[0], best[1], best[2], best[3], best[4]))

def train_one(model, optimizer, cfg, num_node_features, num_classes, device="cpu"):
    best_model_para_dict = None
    best_val = 0
    # best_loss = 1000
    best = []
    for epoch in range(cfg.epochs):
        total_loss = 0
        model.train()
        files = os.listdir(cfg.DATA_PATH+cfg.dataset+'/trainset/')  # 得到文件夹下的所有文件
        num = len(files)
        del files
        for i in range(num):
            DeepSNAP_graph_dataset = torch.load(cfg.DATA_PATH+cfg.dataset+'/train_batch/'+str(i)+'.bin')
            train_loader = DataLoader(DeepSNAP_graph_dataset, collate_fn=Batch.collate(), batch_size=cfg.batch_size[0], shuffle=False)
            for batch in train_loader:
                torch.cuda.empty_cache()
                batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                label = batch.node_label
                '''
                print('pred')
                print(np.shape(pred[batch.node_label_index]))
                print(pred[batch.node_label_index])
                print('label')
                print(np.shape(label))
                print(label)
                '''
                loss = model.loss(pred[batch.node_label_index], label)
                # total_loss += loss
                loss.backward()
                optimizer.step()
                total_loss += float(loss)     #    loss.detach()  loss.cpu().detach().numpy()
            del DeepSNAP_graph_dataset
            del train_loader
            gc.collect()
        # gc.collect()
        torch.cuda.empty_cache()
        train_acc = test.test_one(model, device, name='train')
        val_acc = test.test_one(model, device, name='validation')
        # test_acc = test.test_one(model, device, name='test')

        print("Epoch: {} Train: {:.4f} Validation: {:.4f} Loss: {:.4f}".format(    # Test: {:.4f}
            epoch + 1, train_acc*100, val_acc*100, total_loss))       # test_acc*100,
        # if best_loss > total_loss:
        #     best_loss = total_loss
        if val_acc > best_val:
            best_val = val_acc
            best = [epoch + 1, train_acc*100, val_acc*100, total_loss]    # test_acc*100,
            # best_model = copy.deepcopy(model)
            best_model_para_dict = model.state_dict()
        # print(best)
    torch.save(best_model_para_dict, os.path.join(cfg.best_model_path, 'best_model.pth'))
    print("best_model   Epoch {}:  Train: {:.4f}  Validation: {:.4f}  Loss: {:.4f}".format(    # Test: {:.4f}
        best[0], best[1], best[2], best[3]))   #, best[4]

def train_three(model, optimizer, cfg, num_node_features, num_classes, device="cpu"):
    best_model_para_dict = None
    best_val = 0
    # best_loss = 1000
    best = []
    for epoch in range(cfg.epochs):
        total_loss = 0
        model.train()
        files = os.listdir('/mnt/TOSHIBAdata2T/'+cfg.dataset+'/three/train_graph/')  # 得到文件夹下的所有文件
        # files.sort(key=None, reverse=False)
        for i in range(0, len(files), cfg.batch_size[0]):
            graphs = []
            if i < cfg.batch_size[0]*(int(len(files)/cfg.batch_size[0])):
                for j in range(cfg.batch_size[0]):
                    graphs.append(torch.load('/mnt/TOSHIBAdata2T/'+cfg.dataset+'/three/train_graph/'+str(files[i+j])))  # +'.bin'
            else:
                for j in range(len(files) - (cfg.batch_size[0]*(int(len(files)/cfg.batch_size[0])))):
                    graphs.append(torch.load('/mnt/TOSHIBAdata2T/'+cfg.dataset+'/three/train_graph/'+str(files[i+j])))
            DeepSNAP_graph_dataset = GraphDataset(graphs=graphs, task='node', minimum_node_per_graph=0)
            train_loader = DataLoader(DeepSNAP_graph_dataset, collate_fn=Batch.collate(), batch_size=cfg.batch_size[0], shuffle=False)
            for batch in train_loader:
                batch.to(device)
                optimizer.zero_grad()
                pred = model(batch)
                label = batch.node_label
                '''
                print('pred')
                print(np.shape(pred[batch.node_label_index]))
                print(pred[batch.node_label_index])
                print('label')
                print(np.shape(label))
                print(label)
                '''
                loss = model.loss(pred[batch.node_label_index], label)
                # total_loss += loss
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().detach().numpy()
            # del graphs
            # del train_loader
        # gc.collect()
        # torch.cuda.empty_cache()
        train_acc = test.test_three(model, device, name='train')
        val_acc = test.test_three(model, device, name='validation')
        test_acc = test.test_three(model, device, name='test')

        print("Epoch: {} Train: {:.4f} Validation: {:.4f} Test: {:.4f} Loss: {:.4f}".format(    #
            epoch + 1, train_acc*100, val_acc*100, test_acc*100, total_loss))       #
        # if best_loss > total_loss:
        #     best_loss = total_loss
        if val_acc > best_val:
            best_val = val_acc
            best = [epoch + 1, train_acc*100, val_acc*100, test_acc*100, total_loss]    #
            # best_model = copy.deepcopy(model)
            best_model_para_dict = model.state_dict()
        # print(best)
    torch.save(best_model_para_dict, os.path.join(cfg.best_model_path, 'best_model.pth'))
    print("best_model   Epoch {}:  Train: {:.4f}  Validation: {:.4f} Test: {:.4f}  Loss: {:.4f}".format(    #
        best[0], best[1], best[2], best[3], best[4]))   #


if __name__ == '__main__':
    print('train')
