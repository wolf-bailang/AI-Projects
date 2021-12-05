# -*- coding: utf-8 -*-

# Code for paper:
# [Title]  - "GCL"
# [Author] - Junbin Zhang
# [Github] - https://github.com/

import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from typing import Union, Tuple, Optional  # 类型检查，防止运行时出现参数和返回值类型不符合。

import torch
import torch_scatter  # 通过src和index两个张量来获得一个新的张量。scatter()一般可以用来对标签进行one-hot编码，
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag

import networkx as nx

import torch_geometric
import torch_geometric.transforms as T
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)
# PRW：https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html
# The PyG built-in GCNConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
# from torch_geometric.data import DataLoader

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from deepsnap.graph import Graph
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset

from core.data_process import DataProcess
from core.graph import GraphGenerate_nx, NodeEmebedding


def build_optimizer(cfg, parameters):
    weight_decay = cfg.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, parameters)
    if cfg.optimizer == 'adam':
        optimizer = optim.Adam(filter_fn, lr=cfg.lr, weight_decay=weight_decay)
    elif cfg.optimizer == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=cfg.lr, momentum=0.95, weight_decay=weight_decay)
    elif cfg.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=cfg.lr, weight_decay=weight_decay)
    elif cfg.optimizer == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=cfg.lr, weight_decay=weight_decay)
    if cfg.optimizer_scheduler == 'none':
        return None, optimizer
    elif cfg.optimizer_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.optimizer_decay_step,
                                              gamma=cfg.optimizer_decay_rate)
    elif cfg.optimizer_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.optimizer_restart)
    return scheduler, optimizer


# GNN Layers
class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, emb=False):
        super(GNNStack, self).__init__()
        conv_model = self.build_conv_model(args.model_type)  # GraphSage / GAT
        self.convs = nn.ModuleList()
        self.convs.append(conv_model(input_dim, hidden_dim))
        # Python assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers - 1):
            self.convs.append(conv_model(args.heads * hidden_dim, hidden_dim))
        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(args.heads * hidden_dim, hidden_dim), nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, output_dim))
        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.emb = emb

    def build_conv_model(self, model_type):
        if model_type == 'GraphSage':
            return GraphSage
        elif model_type == 'GAT':
            # When applying GAT with num heads > 1, one needs to modify the
            # input and output dimension of the conv layers (self.convs),
            # to ensure that the input dim of the next layer is num heads
            # multiplied by the output dim of the previous layer.
            # In case you want to play with multiheads, you need to change the for-loop when builds up self.convs to be
            # self.convs.append(conv_model(hidden_dim * num_heads, hidden_dim)),
            # and also the first nn.Linear(hidden_dim * num_heads, hidden_dim) in post-message-passing.
            return GAT

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout)
        x = self.post_mp(x)
        if self.emb == True:
            return x
        # Applies a softmax followed by a logarithm.
        # https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.log_softmax
        return F.log_softmax(x, dim=1)

    # The negative log likelihood loss. https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#nll_loss
    def loss(self, pred, label):
        # nll_loss(negative log likelihood loss)：最大似然 / log似然代价函数.CrossEntropyLoss: 交叉熵损失函数。交叉熵描述了两个概率
        # 分布之间的距离，当交叉熵越小说明二者之间越接近。
        return F.nll_loss(pred, label)


# GCN Model
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout, return_embeds=False):
        # Implement this function that initializes self.convs,
        # self.bns, and self.softmax.
        super(GCN, self).__init__()
        # A list of GCNConv layers
        self.convs = None
        # A list of 1D batch normalization layers
        self.bns = None
        # The log softmax layer
        self.softmax = None
        # use torch.nn.ModuleList for self.convs and self.bns
        # self.convs has num_layers GCNConv layers
        # self.bns has num_layers - 1 BatchNorm1d layers
        # use torch.nn.LogSoftmax for self.softmax
        # The parameters can set for GCNConv include 'in_channels' and
        # 'out_channels'. More information please refer to the documentation:
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels=input_dim, out_channels=hidden_dim)] +
            [GCNConv(in_channels=hidden_dim, out_channels=hidden_dim)
             for i in range(num_layers - 2)] +
            [GCNConv(in_channels=hidden_dim, out_channels=output_dim)]
        )
        # The only parameter you need to set for BatchNorm1d is 'num_features'
        # More information please refer to the documentation:
        # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        self.bns = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(num_features=hidden_dim)
            for i in range(num_layers - 1)
        ])
        self.softmax = torch.nn.LogSoftmax()
        # Probability of an element to be zeroed
        self.dropout = dropout
        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        # Implement this function that takes the feature tensor x,
        # edge_index tensor adj_t and returns the output tensor as
        # shown in the figure.
        out = None
        # Construct the network as showing in the figure
        # torch.nn.functional.relu and torch.nn.functional.dropout are useful
        # More information please refer to the documentation:
        # https://pytorch.org/docs/stable/nn.functional.html
        # Don't forget to set F.dropout training to self.training
        # If return_embeds is True, then skip the last softmax layer
        for conv, bn in zip(self.convs[:-1], self.bns):
            x1 = F.relu(bn(conv(x, adj_t)))
            if self.training:
                x1 = F.dropout(x1, p=self.dropout)
            x = x1
        x = self.convs[-1](x, adj_t)  # GCNVonv
        out = x if self.return_embeds else self.softmax(x)
        return out


class GraphSage(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True,
                 bias=False, **kwargs):
        super(GraphSage, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.lin_l = None
        self.lin_r = None
        # Define the layers needed for the message and update functions below.
        # self.lin_l is the linear transformation that you apply to embedding for central node.
        # self.lin_r is the linear transformation that you apply to aggregated message from neighbors.
        self.lin_l = nn.Linear(self.in_channels, self.out_channels)
        self.lin_r = nn.Linear(self.in_channels, self.out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, size=None):
        """"""
        out = None
        # Implement message passing, as well as any post-processing (our update rule).
        # First call propagate function to conduct the message passing.
        # See there for more information: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        # We use the same representations for central (x_central) and
        # neighbor (x_neighbor) nodes, which means you'll pass x=(x, x) to propagate.
        # Update our node embedding with skip connection.
        # If normalize is set, do L-2 normalization (defined in torch.nn.functional)
        prop = self.propagate(edge_index, x=(x, x), size=size)
        out = self.lin_l(x) + self.lin_r(prop)
        if self.normalize:
            out = F.normalize(out, p=2)
        return out

    def message(self, x_j):
        out = None
        # Implement message function here.
        out = x_j
        return out

    def aggregate(self, inputs, index, dim_size=None):
        out = None
        # The axis along which to index number of nodes.
        node_dim = self.node_dim  # node_dim的情况可以看PyG那个文档，是MessagePassing的参数：indicates along which axis to propagate.
        # Implement aggregate function here.
        # See here as how to use torch_scatter.scatter:
        # https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html#torch_scatter.scatter
        out = torch_scatter.scatter(inputs, index, node_dim, dim_size=dim_size, reduce='mean')  # 'sum'
        return out


class GAT(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=2, negative_slope=0.2, dropout=0., **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.lin_l = None
        self.lin_r = None
        self.att_l = None
        self.att_r = None
        # Define the layers needed for the message functions below.
        # self.lin_l is the linear transformation that you apply to embeddings BEFORE message passing.
        # Pay attention to dimensions of the linear layers, since we're using multi-head attention.
        self.lin_l = nn.Linear(self.in_channels, self.out_channels * self.heads)
        self.lin_r = self.lin_l
        # Define the attention parameters \overrightarrow{a_l/r}^T in the above intro.
        # You have to deal with multi-head scenarios.
        # Use nn.Parameter instead of nn.Linear
        self.att_l = nn.Parameter(torch.zeros(self.heads, self.out_channels))
        self.att_r = nn.Parameter(torch.zeros(self.heads, self.out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size=None):
        H, C = self.heads, self.out_channels
        # Implement message passing, as well as any pre- and post-processing (our update rule).
        # First apply linear transformation to node embeddings, and split that into multiple heads. We use the same
        # representations for source and target nodes, but apply different linear weights (W_l and W_r)
        # Calculate alpha vectors for central nodes (alpha_l) and neighbor nodes (alpha_r).
        # Call propagate function to conduct the message passing.
        # Remember to pass alpha = (alpha_l, alpha_r) as a parameter.
        # See there for more information: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        # Transform the output back to the shape of N * d.
        x_l = self.lin_l(x).reshape(-1, H, C)
        x_r = self.lin_r(x).reshape(-1, H, C)
        alpha_l = self.att_l * x_l
        alpha_r = self.att_r * x_r
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r), size=size)
        out = out.reshape(-1, H * C)
        '''
        x_l=self.lin_l(x)  #𝐖𝐥ℎ𝑖
        x_r=self.lin_r(x)  #𝐖rℎj
        x_l=x_l.view(-1,H,C)  #N x H x C
        x_r=x_r.view(-1,H,C)
        alpha_l = (x_l * self.att_l).sum(axis=1)  #*是逐元素相乘（每个特征对应的所有节点一样处理？）。sum的维度是H（聚合）。
        #最终维度是N*C？
        #alpha_l就是a^T * Wl * hi
        alpha_r = (x_r * self.att_r).sum(axis=1)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=(alpha_l, alpha_r),size=size)
        out = out.view(-1, H * C)  #N*D(D=H*C)
        #print(list(self.lin_l.parameters()))
        #print(list(self.lin_r.parameters()))
        #print(list(self.lin_l.parameters())==list(self.lin_r.parameters()))
        #print(hash(self.lin_l))
        #print(hash(self.lin_r))
        #这几行注释是给下一个cell用的，目的是看一下lin_l和lin_r这两层是不是真的一样（按照浅拷贝的定义来讲就应该一样）
        #用id也行……哎这个辨别的方式倒是很多啦无所谓
        #然后我发现真的一样……但是似乎本意又应该是不一样？
        #但是torch_geometric的GAT实现又确实也是这么写的：https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/gat_conv.html
        #如果in_channels是一个数就一样，如果是一堆数就不一样？震惊，这是为什么呢？
        #以后再回来研究
        '''
        return out

    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):
        # Implement message function. Putting the attention in message instead of in update is a little tricky.
        # Calculate the final attention weights using alpha_i and alpha_j, and apply leaky Relu.
        # Calculate softmax over the neighbor nodes for all the nodes. Use torch_geometric.utils.softmax instead of the
        # one in Pytorch.
        # Apply dropout to attention weights (alpha).
        # Multiply embeddings and attention weights. As a sanity check, the output should be of shape E * H * d.
        # ptr (LongTensor, optional): If given, computes the softmax based on sorted inputs in CSR representation.
        # You can simply pass it to softmax.
        alpha = F.leaky_relu(alpha_i + alpha_j, negative_slope=self.negative_slope)
        if ptr:
            att_weight = F.softmax(alpha_i + alpha_j, ptr)
        else:
            att_weight = torch_geometric.utils.softmax(alpha_i + alpha_j, index)
        att_weight = F.dropout(att_weight, p=self.dropout)
        out = att_weight * x_j
        '''
        #alpha：[E, C]
        alpha = alpha_i + alpha_j  #leakyrelu的对象
        alpha = F.leaky_relu(alpha,self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        #https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch-geometric-utils
        #https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/utils/softmax.py
        #没仔细看，反正参数是这些参数
        alpha = F.dropout(alpha, p=self.dropout, training=self.training).unsqueeze(1)  #[E,1,C]
        out = x_j * alpha  #通过计算得到的alpha来计算节点信息聚合值（我大概理解就是得到h_i^'）  #[E,H,C]
        '''
        return out

    def aggregate(self, inputs, index, dim_size=None):
        # Implement aggregate function here.
        # See here as how to use torch_scatter.scatter:
        # https://pytorch-scatter.readthedocs.io/en/latest/_modules/torch_scatter/scatter.html
        # Pay attention to "reduce" parameter is different from that in GraphSage.
        out = torch_scatter.scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
        return out

'''
def train(dataset, args):
    # print("Node task. test set size:", np.sum(dataset[0]['train_mask'].numpy()))
    # test_loader = loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)  # shuffle=True用于打乱数据集,每次都会以不同的顺序返回。
    test_loader = loader = dataset
    # build model
    model = GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes, args)
    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    losses = []
    test_accs = []
    for epoch in range(args.epochs):
        total_loss = 0
        model.train()  # 将模型设置为训练状态,作用:使Dropout,batchnorm知道后有不同表现
        for batch in loader:
            opt.zero_grad()  # 清空过往梯度；
            pred = model(batch)
            label = batch.y
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]
            loss = model.loss(pred, label)
            loss.backward()  # 反向传播，计算当前梯度；
            opt.step()  # 根据梯度更新网络参数
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)
        losses.append(total_loss)

        if epoch % 10 == 0:
            test_acc = test(test_loader, model)
            test_accs.append(test_acc)
        else:
            test_accs.append(test_accs[-1])
    return test_accs, losses


def test(loader, model, is_validation=True):
    model.eval()

    correct = 0
    for data in loader:
        with torch.no_grad():
            # max(dim=1) returns values, indices tuple; only need indices
            pred = model(data).max(dim=1)[1]
            label = data.y

        mask = data.val_mask if is_validation else data.test_mask
        # node classification: only evaluate on nodes in test set
        pred = pred[mask]
        label = data.y[mask]

        correct += pred.eq(label).sum().item()

    total = 0
    for data in loader.dataset:
        total += torch.sum(data.val_mask if is_validation else data.test_mask).item()
    return correct / total


class objectview(object):
    def __init__(self, d):
        self.__dict__ = d
'''


class DeepsnapGNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, args):
        super(DeepsnapGNN, self).__init__()
        self.num_layers = args["num_layers"]

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_size, hidden_size))
        for l in range(self.num_layers - 1):
            self.convs.append(SAGEConv(hidden_size, hidden_size))
        self.post_mp = nn.Linear(hidden_size, output_size)

    def forward(self, data):
        x, edge_index, batch = data.node_feature, data.edge_index, data.batch

        for i in range(len(self.convs) - 1):
            x = self.convs[i](x, edge_index)
            x = F.leaky_relu(x)
        x = self.convs[-1](x, edge_index)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, pred, label):
        return F.nll_loss(pred, label)





def train(train_loader, val_loader, test_loader, args, num_node_features, num_classes,
          device="cpu"):
    model = DeepsnapGNN(num_node_features, args['hidden_size'], num_classes, args).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=5e-4)

    for epoch in range(args['epochs']):
        total_loss = 0
        model.train()
        for batch in train_loader:
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            label = batch.node_label
            loss = model.loss(pred[batch.node_label_index], label)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_acc = test(train_loader, model, device)
        val_acc = test(val_loader, model, device)
        test_acc = test(test_loader, model, device)
        print("Epoch {}: Train: {:.4f}, Validation: {:.4f}. Test: {:.4f}, Loss: {:.4f}".format(
            epoch + 1, train_acc, val_acc, test_acc, total_loss))


def test(loader, model, device='cpu'):  # 'cuda'
    model.eval()
    for batch in loader:
        batch.to(device)
        logits = model(batch)
        pred = logits[batch.node_label_index].max(1)[1]
        acc = pred.eq(batch.node_label).sum().item()
        total = batch.node_label_index.shape[0]
        acc /= total
    return acc


if __name__ == '__main__':
    print("PyTorch has version {}".format(torch.__version__))
    print("torch_geometric has version {}".format(torch_geometric.__version__))

    args = {
        "device": 'cpu',  # 'cuda' if torch.cuda.is_available() else 'cpu',
        "hidden_size": 178,
        "epochs": 100,
        "lr": 0.01,
        "num_layers": 2,
        "dataset": "Cora",
    }

    videos = DataProcess().load_dataset('train')
    graph_datasets = GraphGenerate(videos).frame_to_graph()  # [videos[0]]
    DeepSNAP_graph_dataset = GraphGenerate(videos).networkx_to_DeepSNAP_graph(graph_datasets)  # [videos[0]]
    print(DeepSNAP_graph_dataset[0])
    dataset_train, dataset_val, dataset_test = DeepSNAP_graph_dataset.split(transductive=False,
                                                                            split_ratio=[0.8, 0.1, 0.1])
    '''
    pyg_dataset = Planetoid('./tmp/cora', args["dataset"])
    graphs_train, graphs_val, graphs_test = \
        GraphDataset.pyg_to_graphs(pyg_dataset, fixed_split=True)
    
    dataset_train, dataset_val, dataset_test = \
        GraphDataset(graphs_train, task='node'), GraphDataset(graphs_val, task='node'), \
        GraphDataset(graphs_test, task='node')
    '''
    train_loader = DataLoader(dataset_train, collate_fn=Batch.collate(), batch_size=1)
    val_loader = DataLoader(dataset_val, collate_fn=Batch.collate(), batch_size=1)
    test_loader = DataLoader(dataset_test, collate_fn=Batch.collate(), batch_size=1)
    num_node_features = dataset_train.num_node_features
    print(num_node_features)
    num_classes = dataset_train.num_node_labels
    print(num_classes)
    train(train_loader, val_loader, test_loader, args, num_node_features, num_classes, args["device"])

    '''
    for args in [
        {'model_type': 'GraphSage', 'dataset': 'cora', 'num_layers': 2, 'heads': 1, 'batch_size': 32, 'hidden_dim': 32,
         'dropout': 0.5, 'epochs': 500, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3,
         'lr': 0.01},
    ]:
        args = objectview(args)
        for model in ['GraphSage', 'GAT']:
            args.model_type = model

            # Match the dimension.
            if model == 'GAT':
              args.heads = 2
            else:
              args.heads = 1

            if args.dataset == 'cora':
                # dataset = Planetoid(root='/tmp/cora', name='Cora')
                videos = DataProcess().load_dataset('train')
                graph_datasets = GraphGenerate(videos).frame_to_graph()  # [videos[0]]
                dataset = DeepSNAP_graph_dataset = GraphGenerate(videos).networkx_to_DeepSNAP_graph(graph_datasets)  # [videos[0]]
            else:
                raise NotImplementedError("Unknown dataset")
            test_accs, losses = train(dataset, args)

            print("Maximum accuracy: {0}".format(max(test_accs)))
            print("Minimum loss: {0}".format(min(losses)))

            plt.title(dataset.name)
            plt.plot(losses, label="training loss" + " - " + args.model_type)
            plt.plot(test_accs, label="test accuracy" + " - " + args.model_type)
        plt.legend()
        plt.show()
    '''
    print('end')
