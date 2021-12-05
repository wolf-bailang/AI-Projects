# https://github.com/snap-stanford/deepsnap/blob/master/examples/node_classification/node_classification_planetoid_tensor.py

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
from core.graph import GraphGenerate_nx, GraphGenerate_pyg, NodeEmebedding
from core.config import cfg
import os
import sys
from core.node2vec import NodeEmebeddingNode2Vec
import copy


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    return optimizer


class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(GNN, self).__init__()
        self.dropout = args.dropout
        self.num_layers = args.num_layers

        conv_model = self.build_conv_model(args.model)
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))

        for l in range(args.num_layers - 2):
            self.convs.append(conv_model(hidden_dim, hidden_dim))
        self.convs.append(conv_model(hidden_dim, output_dim))

    def build_conv_model(self, model_type):
        if model_type == 'GCN':
            return pyg_nn.GCNConv
        elif model_type == 'GAT':
            return pyg_nn.GATConv
        elif model_type == "GraphSage":
            return pyg_nn.SAGEConv
        else:
            raise ValueError(
                "Model {} unavailable, please add it to GNN.build_conv_model.".format(model_type))

    def forward(self, data):
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch

        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[len(self.convs) - 1](x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred, label):
        '''
        print('pred')
        print(np.shape(pred))
        print(pred)
        print('label')
        print(np.shape(label))
        print(label)
        '''
        # pred = pred.squeeze()
        # label = label.squeeze()
        return F.nll_loss(pred, label)      # CrossEntropyLoss与nll_loss类似


def train(train_loader, val_loader, test_loader, args, num_node_features, num_classes,
          device="cpu"):
    best_model = None
    best_val = 0
    best = []

    model_cls = GNN
    model = model_cls(num_node_features, args.hidden_dim, num_classes, args).to(device)
    opt = build_optimizer(args, model.parameters())
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for batch in train_loader:
            batch.to(device)
            opt.zero_grad()
            pred = model(batch)
            label = batch.node_label
            # print('pred')
            # print(np.shape(pred[batch.node_label_index]))
            # print(pred[batch.node_label_index])
            # print('label')
            # print(np.shape(label))
            # print(label)
            loss = model.loss(pred[batch.node_label_index], label)
            total_loss += loss
            loss.backward()
            opt.step()
        train_acc = test(train_loader, model, device)
        val_acc = test(val_loader, model, device)
        test_acc = test(test_loader, model, device)

        print("Epoch {}: Train: {:.4f}, Validation: {:.4f}. Test: {:.4f}, Loss: {:.4f}".format(
            epoch + 1, train_acc, val_acc, test_acc, total_loss))
        if val_acc > best_val:
            best_val = val_acc
            best = [train_acc, val_acc, test_acc, total_loss]
            best_model = copy.deepcopy(model)
    print("Train: {:.4f}, Validation: {:.4f}. Test: {:.4f}, Loss: {:.4f}".format(
        best[0], best[1], best[2], best[3]))
    '''
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()
        for i in range(len(train_loader)):
            for batch in train_loader[i]:
                batch.to(device)
                opt.zero_grad()
                pred = model(batch)
                label = batch.node_label
                
                print('pred')
                print(np.shape(pred[batch.node_label_index]))
                print(pred[batch.node_label_index])
                print('label')
                print(np.shape(label))
                print(label)
                
                loss = model.loss(pred[batch.node_label_index], label)
                total_loss += loss
                loss.backward()
                opt.step()
            
            train_acc = test(train_loader, model, device)
            val_acc = test(val_loader, model, device)
            test_acc = test(test_loader, model, device)
            
            # train_acc = test(train_loader, model, device)
            train_acc = test(train_loader[i], model, device)
            val_acc = test(val_loader[i], model, device)
            test_acc = test(test_loader[i], model, device)
            print("Epoch {}: Train: {:.4f}, Validation: {:.4f}. Test: {:.4f}, Loss: {:.4f}".format(
                epoch + 1, train_acc, val_acc, test_acc, total_loss))
    '''


@torch.no_grad()
def test(loader, model, device='cuda'):
    model.eval()
    for batch in loader:
        batch.to(device)
        logits = model(batch)
        pred = logits[batch.node_label_index].max(1)[1]
        acc = pred.eq(batch.node_label).sum().item()
        total = batch.node_label_index.shape[0]
        acc /= total
    return acc


def arg_parse():
    parser = argparse.ArgumentParser(description='Node classification arguments.')

    parser.add_argument('--device', type=str,
                        help='CPU / GPU device.')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs to train.')
    parser.add_argument('--dataset', type=str,
                        help='Node classification dataset. Cora, CiteSeer, PubMed')
    parser.add_argument('--model', type=str,
                        help='GCN, GAT, GraphSAGE.')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size for node classification.')
    parser.add_argument('--hidden_dim', type=int,
                        help='Hidden dimension of GNN.')
    parser.add_argument('--num_layers', type=int,
                        help='Number of graph convolution layers.')
    parser.add_argument('--opt', type=str,
                        help='Optimizer such as adam, sgd, rmsprop or adagrad.')
    parser.add_argument('--weight_decay', type=float,
                        help='Weight decay.')
    parser.add_argument('--dropout', type=float,
                        help='The dropout ratio.')
    parser.add_argument('--lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--split', type=str,
                        help='Randomly split dataset, or use fixed split in PyG. fixed, random')
    parser.set_defaults(
        device='cuda:0',    #'cpu'
        epochs=100,
        dataset='Cora',
        model='GCN',        # 'GAT' "GraphSage"
        batch_size=8,
        hidden_dim=128,    # 32
        num_layers=2,
        opt='adam',         #  'sgd' 'rmsprop'  'adagrad'
        weight_decay=5e-4,
        dropout=0.0,
        lr=0.001,
        split='fixed'     # 'random'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()
    print("PyTorch has version {}".format(torch.__version__))
    '''
    'Cora'    
Data(edge_attr=[10556, 1], edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])
num_classes: 7
num_nodes: 2708
tensor([[0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        ...,
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.],
        [0., 0., 0.,  ..., 0., 0., 0.]])
tensor([3, 4, 4,  ..., 3, 3, 3])
Index fields: train_mask ignored.
Index fields: val_mask ignored.
Index fields: test_mask ignored.
GraphDataset(1)
Batch(batch=[2708], directed=[1], edge_feature=[10556, 1], edge_index=[2, 10556], edge_label_index=[2, 10556], node_feature=[2708, 1433], node_label=[2166], node_label_index=[2166], task=[1])
Cora train dataset: GraphDataset(1)
Cora validation dataset: GraphDataset(1)
Cora test dataset: GraphDataset(1)
Original Cora has 2708 nodes
After the split, Cora has 2166 training nodes
After the split, Cora has 270 validation nodes
After the split, Cora has 272 test nodes
num_node_features = 1433
num_classes = 7
Graph(directed=[1], edge_feature=[10556, 1], edge_index=[2, 10556], edge_label_index=[2, 10556], node_feature=[2708, 1433], node_label=[2166], node_label_index=[2166], task=[])
pred
torch.Size([2166, 7])
tensor([[-1.9974, -1.9573, -1.8758,  ..., -2.0442, -1.9040, -1.9123],
        [-1.9636, -1.9543, -1.9261,  ..., -1.9417, -1.9667, -1.9317],
        [-1.8961, -2.0466, -2.0010,  ..., -1.9164, -1.9239, -1.8464],
        ...,
        [-2.0490, -1.9118, -2.0232,  ..., -1.8193, -1.9843, -1.9450],
        [-1.9973, -2.0730, -1.9984,  ..., -1.8604, -1.9151, -1.8739],
        [-1.9226, -2.0120, -1.9142,  ..., -1.9687, -1.9211, -1.8545]],
       device='cuda:0', grad_fn=<IndexBackward>)
label
torch.Size([2166])
tensor([4, 3, 2,  ..., 4, 2, 4], device='cuda:0')
    '''
    '''
    args.dataset = 'Cora'
    if args.dataset in ['Cora', 'CiteSeer', 'Pubmed', 'ENZYMES']:
        pyg_dataset = Planetoid('./planetoid', args.dataset, transform=T.TargetIndegree())  # load some format of graph data
        # pyg_dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', transform=T.TargetIndegree())
    else:
        raise ValueError("Unsupported dataset.")
    print(len(pyg_dataset))
    print(pyg_dataset[0])
    print("num_classes: {}".format(pyg_dataset.num_classes))
    print("num_nodes: {}".format(pyg_dataset[0].num_nodes))
    print(pyg_dataset[0].x)
    print(pyg_dataset[0].y)

    if args.split == 'random':
        graphs = GraphDataset.pyg_to_graphs(pyg_dataset, verbose=True, fixed_split=False, tensor_backend=True)
        dataset = GraphDataset(graphs, task='node')  # node, edge, link_pred, graph
        # transductive split, inductive split
        dataset_train, dataset_val, dataset_test = dataset.split(transductive=True, split_ratio=[0.8, 0.1, 0.1])
    else:
        graphs_train, graphs_val, graphs_test = GraphDataset.pyg_to_graphs(pyg_dataset, verbose=True, fixed_split=True, tensor_backend=True)
        dataset_train, dataset_val, dataset_test = GraphDataset(graphs_train, task='node'), GraphDataset(graphs_val, task='node'), GraphDataset(graphs_test, task='node')

    print(dataset)
    train_loader = DataLoader(dataset_train, collate_fn=Batch.collate(), batch_size=16)  # basic data loader
    print('train_loader')
    print(len(train_loader))
    for ba in train_loader:
        print(ba)
    val_loader = DataLoader(dataset_val, collate_fn=Batch.collate(), batch_size=16)  # basic data loader
    test_loader = DataLoader(dataset_test, collate_fn=Batch.collate(), batch_size=16)  # basic data loader
    print("Cora train dataset: {}".format(dataset_train))
    print("Cora validation dataset: {}".format(dataset_val))
    print("Cora test dataset: {}".format(dataset_test))
    print("Original Cora has {} nodes".format(dataset_train.num_nodes[0]))
    # The nodes in each set can be find in node_label_index
    print("After the split, Cora has {} training nodes".format(dataset_train[0].node_label_index.shape[0]))
    print("After the split, Cora has {} validation nodes".format(dataset_val[0].node_label_index.shape[0]))
    print("After the split, Cora has {} test nodes".format(dataset_test[0].node_label_index.shape[0]))
    num_node_features = dataset_train.num_node_features
    print('num_node_features = '+str(num_node_features))
    num_classes = dataset_train.num_node_labels
    print('num_classes = ' + str(num_classes))
    print(dataset_train[0])

    train(train_loader, val_loader,test_loader, args, num_node_features, num_classes, args.device)
    '''

    ''' 
Graph(G=[], Name=[], edge_index=[2, 981], edge_label_index=[2, 981], node_feature=[982, 2048], node_label=[982], node_label_index=[982], task=[], weight=[981])
Cora train dataset: GraphDataset(1)
Cora validation dataset: GraphDataset(1)
Cora test dataset: GraphDataset(1)
Original Cora has 982 nodes
After the split, Cora has 589 training nodes
After the split, Cora has 196 validation nodes
After the split, Cora has 197 test nodes
num_node_features = 2048
num_classes = 22
Graph(G=[], Name=[], edge_index=[2, 981], edge_label_index=[2, 981], node_feature=[982, 2048], node_label=[589], node_label_index=[589], task=[], weight=[981])
pred
torch.Size([589, 22])
tensor([[-73.0595, -52.8558,  -0.4920,  ..., -36.8668, -27.0465, -78.3512],
        [-74.7480, -53.8454,  -0.1340,  ..., -35.4519, -25.6062, -77.7338],
        [-73.7293, -52.0483,  -0.8282,  ..., -35.1678, -27.7338, -76.8065],
        ...,
        [-74.3861, -52.4679,  -0.6228,  ..., -37.8069, -28.5896, -77.8759],
        [-72.5142, -52.3990,  -0.4109,  ..., -35.6369, -24.4989, -76.0149],
        [-73.8150, -51.9875,  -0.8609,  ..., -36.6067, -28.0340, -76.7156]],
       device='cuda:0', grad_fn=<IndexBackward>)
label
torch.Size([589])
tensor([ 14,  49, 160, 160, 162, 155, 160,  15,  52, 161,  52,  14, 160,  52,
          0, 159, 157, 159,  49, 160, 155, 155, 160, 158, 160, 159,  49,  32,
        160,  14,  49, 159, 160,  52,  49, 159, 160,   0, 160,  66, 159,   0,
         95, 160,   0,  51, 158,   0, 160,  49,   0, 125,  96,  66, 160, 159,
         14,   0,  49,   0, 159,  49,   0, 159, 160,  50,  14,  50,   0,  49,
        125,  66,  50,  52,  32,  49,  49,   0,  52,  52,  52, 162,  52, 160,
        125,  52,   0, 159, 160, 158,   0, 155, 160, 160, 159,  66, 159, 125,
         52, 160, 158,  53,  48,   0, 159,  14, 160, 125,  14,  14, 125, 125,
          0, 160,  53,  15, 155, 160,  50, 160, 159,  50,  50,  49, 158,  32,
         52,  51, 160, 160, 160, 158, 159, 155, 160,  49, 159, 155, 162,   0,
        159,  50,   0, 155, 158,  52,   0, 158, 159, 159, 158,  52,  32,  52,
         49, 160,  96, 158, 160, 160, 162, 159, 159, 160, 159, 155,   0, 160,
         49, 159, 158, 158,   0, 160,   0, 160,  53,   0,  96, 155, 159,   0,
         66,  15,  66,  49, 160, 155,  14,  66,  53,  53, 162,   0, 160,  52,
        159,  52,  49, 160,  95, 159,   0, 159,  53,  52,   0, 159,  14,  52,
        160, 160,  66, 159, 160,   0, 159, 160,  52, 155,  14, 160,  14,   0,
          0, 159, 160,  52,   0,  49, 160,  50,  53,  53, 160,  51,  52, 125,
        162,  53,  52,  48,  52,  32,  48,  53,   0,  51,  52,  14,  51, 155,
        160,  49, 160,  32,  53,  14, 158,  51, 160, 162,  49, 161,  53,  52,
         14,  53,  49,   0,   0, 158, 158,   0, 159,  52,  49, 160, 160,  66,
          0, 159,   0,  49, 162, 159,  51, 159, 160, 158,  52, 160, 159,  52,
        160, 159,   0,  49,  52,  49,  49, 159, 159, 160, 160,  52,   0,  52,
        158, 159,  95, 160, 155,  49, 160, 158, 125, 160, 159,   0, 155, 125,
        125, 160,   0,  14, 159,   0, 160,  52, 160,   0,  52, 160,   0,  32,
        125, 125,  52, 160, 155, 160, 160,   0,   0,   0, 160,   0, 160,  96,
        158,   0, 159,   0,  32, 155,   0,  50,   0, 125,  49, 155, 159,  52,
        159,   0,  32,  49, 160, 160, 159,   0,  52,  95,   0,  50,  14, 157,
        155,  49, 160, 160, 160, 160,  49,  52, 156, 160,  52,  14,  14,   0,
        160, 158,   0, 160,  66,   0, 159, 159,  48,  53,  48,  49, 158, 160,
        162, 160,  51, 156, 160, 160, 159, 159,  52,  14,  14,  50,  49, 160,
         53, 159, 155,  51,  14, 125, 159, 160,  52,  52,  96,  14,   0,  49,
         14, 160, 155, 160,  51, 160,  50, 158, 160,   0, 155,  53,   0, 162,
        162,  32,  14, 155, 155,   0, 160, 159, 159,  50,   0,  52,  52,  49,
         52, 158, 160, 158,  14,  52,  51, 156, 156,  52,  51,  53, 160,   0,
        160,  32, 160, 155,  49, 160,   0,  53, 160,  66, 160, 160,  53,  51,
         49, 160, 160, 158, 158,  95,  95,  96,  49, 156,  52, 160,  50, 160,
        160, 125, 158, 155, 160,  15, 160, 160, 160, 155,  32, 160,   0,   0,
         14, 158,  14,  52,  52,  49,  50, 158,   0,  51, 159,  50,  52,  50,
        159,   0,  66, 155, 160,   0, 159, 160,  52, 160, 160, 125, 160,  48,
        160, 159,  51, 159, 160, 160, 159, 155, 158,  52, 155, 160,   0, 160,
         53,  53, 160,   0, 159, 160, 159, 159,  32,  52, 162,  14,  96,   0,
         32,  52, 160, 159, 162,  14, 158, 159,  51,  14,  32, 159, 160,  50,
        160], device='cuda:0')
    '''
    # graph = []
    # graphs_temp = []
    graphs = []
    # DataProcess().split_dataset(train_ratio=0.6, test_ratio=0.2, validation_ratio=0.2)
    paths = cfg.DATA_PATH + '/fine_segmentation_lables_dict.json'     # coarse
    lables_dict = DataProcess().load_json(paths)
    path = cfg.DATA_PATH + '/' + 'milk'+'/'  # 文件夹目录train       I3D_feature_fine
    files = os.listdir(path)  # 得到文件夹下的所有文件

    num = 0
    for file in files[0:10]:
    # for file in files:
        print('vidoe num = '+str(num))
        num += 1
        video = DataProcess().load_dataset_one(path, file, lables_dict)
        #################################################
        graph = GraphGenerate_nx(video).frame_to_graph_one_nx()
        # graph = GraphGenerate_pyg(video).frame_to_graph_one_pyg()
        # print(graph)
        # Data(edge_index=[2, 10556], test_mask=[2708], train_mask=[2708], val_mask=[2708], x=[2708, 1433], y=[2708])
        # graphs = GraphDataset.pyg_to_graphs(dataset=graph, verbose=True, fixed_split=False, tensor_backend=True, netlib=None)
        # # graph = NodeEmebedding().node_embedding(graph, 8)train_label =
        '''graphs.append(Graph(graph))
        '''
        graph_temp = Graph(graph)
        emb = NodeEmebeddingNode2Vec().NodeEmebedding(data=graph_temp)
        # NodeEmebedding().visualize_emb(graph, emb)
        # print(graphs[0].node_feature)
        # print(graphs[0].node_feature.shape)
        node_f = torch.zeros(len(graph_temp.node_feature), 2048+128)
        for i in range(len(graph_temp.node_feature)):
            node_f[i] = torch.cat([graph_temp.node_feature[i], emb.embedding.weight.data.cpu()[i]], dim=0)
        # print(node_f.shape)
        # print(emb.embedding.weight.data.cpu())
        graph_temp.node_feature = node_f
        # print(graphs[0].node_feature)
        graphs.append(graph_temp)
        #################################################Graph()
        # break

    print(graphs[0])
    # print(len(graphs))
    DeepSNAP_graph_dataset = GraphDataset(graphs=graphs, task='node', minimum_node_per_graph=0)
    # print(DeepSNAP_graph_dataset)

    # dataset_train, dataset_val, dataset_test = DeepSNAP_graph_dataset.split(transductive=True, split_ratio=[0.3, 0.3, 0.4])
    # Train: 0.6303, Validation: 0.8034. Test: 0.0000, Loss: 42.7436
    # Train: 0.0652, Validation: 0.5450. Test: 0.1126, Loss: 39.6708
    # 111 Train: 0.3455, Validation: 0.5088. Test: 0.2878, Loss: 24.6393
    # dataset_train, dataset_val, dataset_test = DeepSNAP_graph_dataset.split(transductive=False, split_ratio=[0.3, 0.3, 0.4])
    # Train: 0.3975, Validation: 0.1335.Test: 0.1933, Loss: 8.0172
    # Train: 0.5655, Validation: 0.2451.Test: 0.0338, Loss: 4.2018
    # 111 Train: 0.6682, Validation: 0.3831. Test: 0.3143, Loss: 2.9436

    # dataset_train, dataset_val, dataset_test = DeepSNAP_graph_dataset.split(transductive=True, split_ratio=[0.8, 0.1, 0.1])
    # 111 Train: 0.6701, Validation: 0.7600. Test: 0.5413, Loss: 19.6987
    dataset_train, dataset_val, dataset_test = DeepSNAP_graph_dataset.split(transductive=False, split_ratio=[0.8, 0.2, 0.0])
    # 111 Train: 0.5452, Validation: 0.3009. Test: 0.3187, Loss: 12.9422

    train_loader = DataLoader(dataset_train, collate_fn=Batch.collate(), batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset_val, collate_fn=Batch.collate(), batch_size=1, shuffle=True)
    test_loader = DataLoader(dataset_test, collate_fn=Batch.collate(), batch_size=1, shuffle=True)
    print("Cora train dataset: {}".format(dataset_train))
    print("Cora validation dataset: {}".format(dataset_val))
    print("Cora test dataset: {}".format(dataset_test))
    print("Original Cora has {} nodes".format(dataset_train.num_nodes[0]))
    # The nodes in each set can be find in node_label_index
    print("After the split, Cora has {} training nodes".format(dataset_train[0].node_label_index.shape[0]))
    print("After the split, Cora has {} validation nodes".format(dataset_val[0].node_label_index.shape[0]))
    print("After the split, Cora has {} test nodes".format(dataset_test[0].node_label_index.shape[0]))

    num_node_features = dataset_train.num_node_features
    print('num_node_features = '+str(num_node_features))
    num_classes = 178   # dataset_train.num_node_labels
    print('num_classes = ' + str(dataset_train.num_node_labels))
    # print(dataset_train[0])

    train(train_loader, val_loader, test_loader, args, num_node_features, num_classes, args.device)

    '''
    # DataProcess().split_dataset(train_ratio=0.6, test_ratio=0.2, validation_ratio=0.2)
    videos = []
    path = cfg.DATA_PATH + '/' + 'train' + '/'  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件
    for file in files:
        videos = DataProcess().load_dataset111(path, file)
        # print('111111111111111111')
        for i in videos:
            graph_datasets = GraphGenerate([i]).frame_to_graph()  # [videos[0]]videos
            # print(graph_datasets[0])
            DeepSNAP_graph_dataset = GraphGenerate([i]).networkx_to_DeepSNAP_graph(graph_datasets)  # [videos[0]]videos
            # print(DeepSNAP_graph_dataset[0])
            dataset_train, dataset_val, dataset_test = DeepSNAP_graph_dataset.split(transductive=True,
                                                                                    split_ratio=[0.6, 0.2, 0.2])
            train_loader = DataLoader(dataset_train, collate_fn=Batch.collate(), batch_size=1)  # basic data loader
            
            print('train_loader')
            print(type(trainset_loader))
            print(len(trainset_loader))
            for ba in trainset_loader:
                print(ba)
                
            val_loader = DataLoader(dataset_val, collate_fn=Batch.collate(), batch_size=1)  # basic data loader
            test_loader = DataLoader(dataset_test, collate_fn=Batch.collate(), batch_size=1)  # basic data loader
            print("Cora train dataset: {}".format(dataset_train))
            print("Cora validation dataset: {}".format(dataset_val))
            print("Cora test dataset: {}".format(dataset_test))
            print("Original Cora has {} nodes".format(dataset_train.num_nodes[0]))
            # The nodes in each set can be find in node_label_index
            print("After the split, Cora has {} training nodes".format(dataset_train[0].node_label_index.shape[0]))
            print("After the split, Cora has {} validation nodes".format(dataset_val[0].node_label_index.shape[0]))
            print("After the split, Cora has {} test nodes".format(dataset_test[0].node_label_index.shape[0]))

            num_node_features = dataset_train.num_node_features
            print('num_node_features = ' + str(num_node_features))
            num_classes = 178  # dataset_train.num_node_labels
            print('num_classes = ' + str(num_classes))
            print(dataset_train[0])
            train(train_loader, val_loader, test_loader, args, num_node_features, num_classes, args.device)

    # test(loader, model, device='cuda')
    '''

    '''
    # DataProcess().split_dataset(train_ratio=0.6, test_ratio=0.2, validation_ratio=0.2)
    videos = DataProcess().load_dataset('test')
    graph_datasets = GraphGenerate([videos[2]]).frame_to_graph()  # videos
    # print(graph_datasets[0])
    DeepSNAP_graph_dataset = GraphGenerate([videos[2]]).networkx_to_DeepSNAP_graph(graph_datasets)  # videos
    # print(DeepSNAP_graph_dataset[0])
    dataset_train, dataset_val, dataset_test = DeepSNAP_graph_dataset.split(transductive=True,
                                                                                    split_ratio=[0.6, 0.2, 0.2])
    train_loader = DataLoader(dataset_train, collate_fn=Batch.collate(), batch_size=1, shuffle=True)  # basic data loader
    
    print('train_loader')
    print(type(trainset_loader))
    print(len(trainset_loader))
    for ba in trainset_loader:
        print(ba)
    
    val_loader = DataLoader(dataset_val, collate_fn=Batch.collate(), batch_size=1, shuffle=True)  # basic data loader
    test_loader = DataLoader(dataset_test, collate_fn=Batch.collate(), batch_size=1, shuffle=True)  # basic data loader
    print("Cora train dataset: {}".format(dataset_train))
    print("Cora validation dataset: {}".format(dataset_val))
    print("Cora test dataset: {}".format(dataset_test))
    print("Original Cora has {} nodes".format(dataset_train.num_nodes[0]))
    # The nodes in each set can be find in node_label_index
    print("After the split, Cora has {} training nodes".format(dataset_train[0].node_label_index.shape[0]))
    print("After the split, Cora has {} validation nodes".format(dataset_val[0].node_label_index.shape[0]))
    print("After the split, Cora has {} test nodes".format(dataset_test[0].node_label_index.shape[0]))

    num_node_features = dataset_train.num_node_features
    print('num_node_features = ' + str(num_node_features))
    num_classes = 178  # dataset_train.num_node_labels
    print('num_classes = ' + str(num_classes))
    print(dataset_train[0])
    train(train_loader, val_loader, test_loader, args, num_node_features, num_classes, args.device)
    '''
