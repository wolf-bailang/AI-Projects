# -*- coding: utf-8 -*-

# Code for paper:
# [Title]  - "GCL"
# [Author] - Junbin Zhang
# [Github] - https://github.com/


import torch
import torch.nn as nn
from torch.optim import SGD
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy as np
from core.data_process import DataProcess
import torch_geometric
from deepsnap.graph import Graph
from deepsnap.dataset import GraphDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_networkx
# Helper function for visualization.
# %matplotlib inline
from core.config import cfg
import gc
from core.my_own_dataset import MyOwnDataset
import torch_geometric.transforms as T
# from core.node2vec import NodeEmebeddingNode2Vec
import numba as nb


# Visualization function for NX graph or PyTorch tensor
def visualize(graph, colors, epoch=None, loss=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    if torch.is_tensor(graph):
        h = graph.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=colors, cmap="Set2")
        if epoch is not None and loss is not None:
            plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    else:
        nx.draw_networkx(graph, pos=nx.spring_layout(graph, seed=42), with_labels=True, node_color=colors, cmap="Set2")
    plt.show()


class GraphGenerate_nx:
    def __init__(self):
        # Create a graph dataset
        self.graph_datasets = []
        self.color = []

    def get_color(self):
        return self.color

    def create_graph(self):
        # Create an undirected graph G
        """
        undirected_graph = nx.Graph()
        return undirected_graph
        """
        # Create a directed graph
        directed_graph = nx.DiGraph()
        # print(graph.is_directed())
        return directed_graph

    # Add graph level attribute
    def add_graph_attribute(self, graph, name):
        graph.graph["Name"] = name
        # print(graph.graph)

    # Add one node with node level attributes
    def add_node(self, graph, index, feature, label):
        graph.add_node(index, node_feature=torch.tensor(feature, dtype=torch.float32), node_label=label)
        # Get attributes of the node 0
        # node_0_attr = graph.nodes[0]
        # print("Node 0 has the attributes {}".format(node_0_attr))
        # Get number of nodes
        # num_nodes = graph.number_of_nodes()
        # print("G has {} nodes".format(num_nodes))

    def add_edge(self, graph, index, weight=0.5):
        # Add one edge with edge weight 0.5
        graph.add_edge(index - 1, index, weight=weight)

    def add_edges(self, graph, index, weight=0.5, startnode=0, endnode=0):
        # Add edges with edge weight 0.5
        graph.add_edge(startnode, endnode, weight=weight)

    def frame_to_graph(self, video_name):
        node_idx = 0
        templabel = 0
        # j = 0
        # self.color = []
        new_graph = self.create_graph()
        self.add_graph_attribute(new_graph, video_name)
        video_path = cfg.DATA_PATH+cfg.dataset+"/groundTruth/"+video_name+".txt"
        labels = np.loadtxt(video_path, dtype=str)
        video_feature_path = cfg.DATA_PATH+cfg.dataset+"/features/"+video_name+".npy"
        features = np.load(video_feature_path)
        frame_features = features.T
        for i in range(len(labels)):
            frame_feature = frame_features[i]
            frame_label = cfg.label2index_dict[labels[i]]
            self.add_node(new_graph, node_idx, frame_feature, frame_label)
            if templabel == frame_label:
                weight = 1
                # self.color.append(j)
            else:
                weight = 0
                templabel = frame_label
                # j += 1
                # self.color.append(j)
            if node_idx != 0:
                self.add_edge(new_graph, node_idx, weight=weight)
            node_idx += 1
        # print(self.color)
        # visualize(new_graph, colors=self.color, epoch=None, loss=None)
        # print(new_graph.edges)
        return new_graph

    def frame_to_graph_edges(self, video_name):
        node_idx = 0
        templabel = 0
        startnode = 0
        endnode = 0
        # j = 0
        # self.color = []
        new_graph = self.create_graph()
        self.add_graph_attribute(new_graph, video_name)
        video_path = cfg.DATA_PATH+cfg.dataset+"/groundTruth/"+video_name+".txt"
        labels = np.loadtxt(video_path, dtype=str)
        video_feature_path = cfg.DATA_PATH+cfg.dataset+"/features/"+video_name+".npy"
        features = np.load(video_feature_path)
        frame_features = features.T
        for i in range(len(labels)):
            frame_feature = frame_features[i]
            frame_label = cfg.label2index_dict[labels[i]]
            self.add_node(new_graph, node_idx, frame_feature, frame_label)
            if templabel == frame_label:
                weight = 1
                endnode = node_idx
                # self.color.append(j)
            else:
                weight = 0
                templabel = frame_label
                startnode = node_idx
                endnode = node_idx
                # j += 1
                # self.color.append(j)
            if node_idx != 0:
                for num in range(startnode, endnode):
                    self.add_edges(new_graph, node_idx, weight=weight, startnode=num, endnode=endnode)
            node_idx += 1
        # self.visualize(new_graph, colors=self.color, epoch=None, loss=None)
        # print(new_graph.edges)
        return new_graph

    def video_to_graph_nx(self, name):
        videos_name_path = cfg.DATA_PATH+cfg.dataset+"/"+name+"set.txt"
        videos_name = np.loadtxt(videos_name_path, dtype=str)
        for video_name in videos_name:
            # graph = self.frame_to_graph(video_name)
            graph = self.frame_to_graph_edges(video_name)
            # print(graph)
            '''
            # NodeEmebedding
            graph_temp = Graph(graph)
            # emb = NodeEmebeddingNode2Vec().NodeEmebedding(data=graph_temp)
            emb = np.load(cfg.DATA_PATH+cfg.dataset+"/emb/Node2Vec2_edges_all/"+video_name+'.npy')
            node_f = torch.zeros(len(graph_temp.node_feature), 2048 + 128)
            for i in range(len(graph_temp.node_feature)):
                node_f[i] = torch.cat([graph_temp.node_feature[i], torch.Tensor(emb[i])], dim=0)
            graph_temp.node_feature = node_f
            self.graph_datasets.append(graph_temp)
            cfg.num_node_features = 2048 + 128
            '''
            torch.save(graph, cfg.DATA_PATH+cfg.dataset+'/'+name+'set/'+str(video_name)+'.bin')
        print('video to graph end')







    def graph_to_batch_to_save(self, name='train'):
        videos_name_path = cfg.DATA_PATH+cfg.dataset+"/"+name+"set.txt"
        videos_name = np.loadtxt(videos_name_path, dtype=str)
        if name == 'train':
            batch_size = cfg.batch_size[0]
        if name == 'validation':
            batch_size = cfg.batch_size[1]
        if name == 'test':
            batch_size = cfg.batch_size[2]
        for i in range(int(len(videos_name) / batch_size)):
            dataset = GraphGenerate_nx().video_to_graph_nx_one(videos_name[i * batch_size: (i + 1) * batch_size])
            DeepSNAP_graph_dataset = GraphDataset(graphs=dataset, task='node', minimum_node_per_graph=0)
            torch.save(DeepSNAP_graph_dataset, cfg.DATA_PATH+cfg.dataset+'/'+name+'set/'+str(i)+'.bin')
        if len(videos_name) % batch_size != 0:
            # print(len(videos_name) % batch_size)
            # print(len(videos_name) - int(len(videos_name) % batch_size))
            dataset = GraphGenerate_nx().video_to_graph_nx_one(
                videos_name[len(videos_name) - (int(len(videos_name) % batch_size)):])
            DeepSNAP_graph_dataset = GraphDataset(graphs=dataset, task='node', minimum_node_per_graph=0)
            torch.save(DeepSNAP_graph_dataset, cfg.DATA_PATH + cfg.dataset+'/'+name+'set/'+str(int(len(videos_name)/batch_size))+'.bin')

    # @nb.vectorize(nopython=True)
    # @nb.jit(nopython=True)
    def video_to_graph_nx_1(self):
        num = 1
        videos_name_path = cfg.DATA_PATH + cfg.dataset + "/videos_name.txt"
        videos_name = np.loadtxt(videos_name_path, dtype=str)
        # for video_name in videos_name[0:1]:
        for video_name in videos_name:
            graph = self.frame_to_graph(video_name)
            # graph = self.frame_to_graph_edges(video_name)
            ''''''
            graph_temp = Graph(graph)
            # emb = NodeEmebeddingNode2Vec().NodeEmebedding(data=graph_temp)
            emb = np.load(cfg.DATA_PATH+cfg.dataset+"/emb/Node2Vec2_edges/"+video_name+'.npy')
            node_f = torch.zeros(len(graph_temp.node_feature), 2048 + 128)
            for i in range(len(graph_temp.node_feature)):
                # print(emb[i])
                # print(emb[i].shape)
                node_f[i] = torch.cat([graph_temp.node_feature[i], torch.Tensor(emb[i])], dim=0)
                # print(node_f[i])
                # print(node_f[i].shape)
            graph_temp.node_feature = node_f
            self.graph_datasets.append(graph_temp)
            cfg.num_node_features = 2048 + 128
            # self.graph_datasets.append(graph)
            print('video = '+str(num))
            num += 1
        return self.graph_datasets

    def frame_to_graph_one_nx(self):
        new_graph = self.create_graph()
        name = self.videos["name"]
        self.add_graph_attribute(new_graph, name)
        index = 0
        templable = 0
        j = 0
        self.color = []
        for frame in self.videos["frame"]:
            frame_feature = frame[0]
            frame_lable = frame[1]
            self.add_node(new_graph, index, frame_feature, frame_lable)
            if templable == frame_lable:
                weight = 1.0
                self.color.append(j)
            else:
                weight = 0.0
                templable = frame_lable
                j += 1
                self.color.append(j)
            if index != 0:
                self.add_edge(new_graph, index, weight=weight)
            index += 1
        # Draw the graph
        # nx.draw(new_graph, with_labels=True)
        # print(new_graph.number_of_nodes())
        # 使用 range 函数创建列表对象,使用迭代器创建 ndarray
        # color = np.fromiter(iter(range(new_graph.number_of_nodes())), dtype=int)
        # color = np.fromiter(iter(range(j)), dtype=int)

        # self.visualize(new_graph, color=self.color, epoch=None, loss=None)
        self.videos.clear()
        # print(new_graph.graph["Name"])
        return new_graph


    def networkx_to_DeepSNAP_graph(self, dataset_graphs):
        for graph in dataset_graphs:
            # print(graph)
            self.graph_datasets.append(Graph(graph))    # networkx to DeepSNAP graph
        DeepSNAP_graph_dataset = GraphDataset(graphs=self.graph_datasets, task='node', minimum_node_per_graph=0)      # 节点分类任务
        print('')
        return DeepSNAP_graph_dataset
        # return dataset_graphs


class GraphGenerate_pyg:
    def __init__(self):
        # self.videos = videos
        # Create a graph dataset
        self.graph_datasets = []

    # Visualization function for NX graph or PyTorch tensor
    def visualize(self, h, color, epoch=None, loss=None):
        plt.figure(figsize=(7, 7))
        plt.xticks([])
        plt.yticks([])
        G = h
        if torch.is_tensor(h):
            h = h.detach().cpu().numpy()
            plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
            if epoch is not None and loss is not None:
                plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
        else:
            nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                             node_color=color, cmap="Set2")
        plt.show()

    def frame_to_graph_one_pyg(self):
        graph_idx_path = cfg.DATA_PATH+cfg.dataset+"/my_dataset/raw/"+cfg.dataset+".graph_idx"
        note_lable_path = cfg.DATA_PATH+cfg.dataset+"/my_dataset/raw/"+cfg.dataset+".note_lable"
        note_feature_path = cfg.DATA_PATH+cfg.dataset+"/my_dataset/raw/"+cfg.dataset+".note_feature"
        edges_path = cfg.DATA_PATH+cfg.dataset+"/my_dataset/raw/"+cfg.dataset+".edges"
        graph_idx = 1
        videos_name_path = cfg.DATA_PATH+cfg.dataset+"/videos_name.txt"
        videos_name = np.loadtxt(videos_name_path, dtype=str)
        for video_name in videos_name[0:10]:
            print(video_name)
            # P14_cam01_P14_scrambledegg P09_cam01_P09_coffee P30_cam01_P30_milk P23_stereo01_P23_tea
            # P03_cam01_P03_sandwich P31_cam02_P31_friedegg P46_cam02_P46_friedegg P52_webcam02_P52_coffee
            # P50_webcam02_P50_coffee P53_webcam01_P53_coffee
            index = 0
            video_path = cfg.DATA_PATH+cfg.dataset+"/groundTruth/"+video_name+".txt"
            lables = np.loadtxt(video_path, dtype=str)
            for lable in lables:
                with open(graph_idx_path, 'a+') as f:
                    f.write(str(graph_idx) + '\n')
                f.close()

                with open(note_lable_path, 'a+') as f:
                    f.write(lable + '\n')
                f.close()

                if index != 0:
                    with open(edges_path, 'a+') as f:
                        f.write(str(index-1)+','+str(index) + '\n')
                    f.close()
                index += 1

            video_feature_path = cfg.DATA_PATH+cfg.dataset+"/features/"+video_name+".npy"
            frame_features = np.load(video_feature_path)
            # print(frame_features.shape)
            # (2048, 2852)
            for frame_feature in frame_features.T:
                with open(note_feature_path, 'a+') as f:
                    for i in range(len(frame_feature) - 1):
                        f.write(str(frame_feature[i]) + ',')
                    f.write(str(frame_feature[-1]) + '\n')
                f.close()
            graph_idx += 1

    def graph_to_dataset_pyg(self):
        transform = T.Compose(
            [T.RandomNodeSplit('train_rest', num_splits=1, num_train_per_class=8, num_val=1, num_test=1),
             T.TargetIndegree(), T.ToDevice(device='cuda', attrs=[], non_blocking=True)])
        dataset = MyOwnDataset(root=cfg.DATA_PATH+cfg.dataset+'/my_dataset', pre_transform=T.GCNNorm())      #.process()transform=transform,
        # dataset1 = MyOwnDataset(root=cfg.DATA_PATH+cfg.dataset+'/my_dataset', transform=None, pre_transform=None).get(0)
        # # print(type(data))
        # dataset = transform(data[0])
        # data = torch.load(cfg.DATA_PATH+cfg.dataset+'/my_dataset/processed/data.pt')
        return dataset


class NodeEmebedding:
    def __init__(self):
        # Please do not change / reset the random seed
        torch.manual_seed(1)

    def graph_to_edge_list(self, graph):
        # the edge list of an nx.Graph. The returned edge_list should be a list of tuples where each tuple is a tuple
        # representing an edge connected by two nodes.
        edge_list = []
        for edge in graph.edges():
            edge_list.append(edge)
        # print(edge_list)
        return edge_list

    def edge_list_to_tensor(self, edge_list):
        # transforms the edge_list to tensor. The input edge_list is a list of tuples and the resulting tensor should
        # have the shape [2 x len(edge_list)].
        edge_index = torch.tensor([])
        edge_index = torch.LongTensor(edge_list).t()
        # edge_index = torch.tensor(edge_list).transpose(1, 0)
        # print(edge_index)
        return edge_index

    def sample_negative_edges(self, graph, num_neg_samples):
        # returns a list of negative edges. The number of sampled negative edges is num_neg_samples. You do not
        # need to consider the corner case when the number of possible negative edges is less than num_neg_samples.
        # It should be ok as long as your implementation works on the karate club network. In this implementation,
        # self loop should not be considered as either a positive or negative edge. Also, notice that the karate club
        # network is an undirected graph, if (0, 1) is a positive edge, do you think (1, 0) can be a negative one?
        neg_edge_list = []
        # 得到图中所有不存在的边（这个函数只会返回一侧，不会出现逆边）
        non_edges_one_side = list(enumerate(nx.non_edges(graph)))
        # print('non_edges_one_side')
        # print(non_edges_one_side)
        # 取样num_neg_samples长度的索引
        neg_edge_list_indices = random.sample(range(0, len(non_edges_one_side)), num_neg_samples)
        # 抽取逻辑是按照non_edges_one_side的索引来抽取边
        for i in neg_edge_list_indices:
            neg_edge_list.append(non_edges_one_side[i][1])
        # print('neg_edge_list')
        # print(neg_edge_list)
        return neg_edge_list

    def create_node_emb(self, num_nodes=34, embedding_dim=16):
        # create the node embedding matrix. A torch.nn.Embedding layer will be returned. You do not need to change the
        # values of num_node and embedding_dim. The weight matrix of returned layer should be initialized under uniform
        # distribution.
        emb = None
        emb = nn.Embedding(num_embeddings=num_nodes, embedding_dim=embedding_dim)
        # print(emb)
        # Embedding初始化本来就是均匀分布。不过在这里应该是要用manual_seed来维持可复现性
        emb.weight.data = torch.rand(num_nodes, embedding_dim)
        # print('emb')
        # print(emb)
        return emb

    def visualize_emb(self, graph, emb):
        temp = ''
        j = 0
        # X = emb.weight.data.numpy()
        X = emb.embedding.weight.data.cpu()
        pca = PCA(n_components=2)
        components = pca.fit_transform(X)
        plt.figure(figsize=(6, 6))
        club1_x = []
        club1_y = []
        club2_x = []
        club2_y = []
        club3_x = []
        club3_y = []
        label = []
        color = []
        '''
            for node in graph.nodes(data=True):
                # print('node')
                print(node)
                if node[1]['node_label'] == 0:
                    club1_x.append(components[node[0]][0])
                    club1_y.append(components[node[0]][1])
                elif node[1]['node_label'] == 1:
                    club2_x.append(components[node[0]][0])
                    club2_y.append(components[node[0]][1])
                else:
                    club3_x.append(components[node[0]][0])
                    club3_y.append(components[node[0]][1])
            plt.scatter(club1_x, club1_y, color="red", label='0')
            plt.scatter(club2_x, club2_y, color="blue", label='2')
            plt.scatter(club3_x, club3_y, color="green", label='3')
        '''
        for node in graph.nodes(data=True):
            # print('node')
            # print(node)
            club1_x.append(components[node[0]][0])
            club1_y.append(components[node[0]][1])
            label.append(node[1]['node_label'])
            if node[1]['node_label'] == temp:
                color.append(j)
            else:
                temp = node[1]['node_label']
                color.append(j)
                j += 1
        # print(str(len(club1_x))+' '+str(len(club1_y))+' '+str(len(color))+' '+str(len(label)))
        # for i in range(len(club1_x)):
        plt.scatter(club1_x, club1_y, c=color, label=None, cmap="Set1")
        # plt.scatter(h[:, 0], h[:, 1], s=140, c=color)
        plt.legend()
        plt.show()

    def accuracy(self, pred, label):
        # the accuracy function takes the pred tensor (the resulting tensor after sigmoid) and the label tensor
        # (torch.LongTensor). Predicted value greater than 0.5 will be classified as label 1. Else it will be
        # classified as label 0. The returned accuracy should be rounded to 4 decimal places.  For example, accuracy
        # 0.82956 will be rounded to 0.8296.
        accu = 0.0
        # accuracy=预测与实际一致的结果数/所有结果数
        # pred tensor和label tensor都是[78*2(156)]大小的tensor
        pred = pred > 0.5
        print('label')
        print(label)
        accu = (pred == label).sum().item() / (pred.shape[0])
        accu = round(accu, 4)
        return accu

    def train(self, emb, loss_fn, sigmoid, train_label, train_edge):
        # Train the embedding layer here. You can also change epochs and learning rate. In general, need to implement:
        # (1) Get the embeddings of the nodes in train_edge
        # (2) Dot product the embeddings between each node pair
        # (3) Feed the dot product result into sigmoid
        # (4) Feed the sigmoid output into the loss_fn
        # (5) Print both loss and accuracy of each epoch (as a sanity check, the loss should decrease during training)
        # print(emb.weight.data.numpy())
        epochs = 5
        learning_rate = 0.1
        optimizer = SGD(emb.parameters(), lr=learning_rate, momentum=0.9)
        for i in range(epochs):
            # print('train')
            optimizer.zero_grad()
            train_node_emb = emb(train_edge)  # [2,156,16]
            dot_product_result = train_node_emb[0].mul(train_node_emb[1])  # 点对之间对应位置嵌入相乘，[156,16]
            dot_product_result = torch.sum(dot_product_result, 1)  # 加起来，构成点对之间向量的点积，[156]
            sigmoid_result = sigmoid(dot_product_result)  # 将这个点积结果经过激活函数映射到0,1之间
            loss_result = loss_fn(sigmoid_result, train_label)
            loss_result.backward()
            optimizer.step()
            # if i % 10 == 0:  # 其实这个应该每一轮都打印一遍的，但是我嫌太大了就十轮打印一遍了
                # print(loss_result)
                # print(self.accuracy(sigmoid_result, train_label))
        # print(emb.weight.data.numpy())

    def node_embedding(self, graph, embedding_dim):
        pos_edge_list = self.graph_to_edge_list(graph)
        # [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 60), (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 68), (68, 69), (69, 70), (70, 71), (71, 72), (72, 73), (73, 74), (74, 75), (75, 76), (76, 77), (77, 78), (78, 79), (79, 80), (80, 81), (81, 82), (82, 83), (83, 84), (84, 85), (85, 86), (86, 87), (87, 88), (88, 89), (89, 90), (90, 91), (91, 92), (92, 93), (93, 94), (94, 95), (95, 96), (96, 97), (97, 98), (98, 99), (99, 100), (100, 101), (101, 102), (102, 103), (103, 104), (104, 105), (105, 106), (106, 107), (107, 108), (108, 109), (109, 110), (110, 111), (111, 112), (112, 113), (113, 114), (114, 115), (115, 116), (116, 117), (117, 118), (118, 119), (119, 120), (120, 121), (121, 122), (122, 123), (123, 124), (124, 125), (125, 126), (126, 127), (127, 128), (128, 129), (129, 130), (130, 131), (131, 132), (132, 133), (133, 134), (134, 135), (135, 136), (136, 137), (137, 138), (138, 139), (139, 140), (140, 141), (141, 142), (142, 143), (143, 144), (144, 145), (145, 146), (146, 147), (147, 148), (148, 149), (149, 150), (150, 151), (151, 152), (152, 153), (153, 154), (154, 155), (155, 156), (156, 157), (157, 158), (158, 159), (159, 160), (160, 161), (161, 162), (162, 163), (163, 164), (164, 165), (165, 166), (166, 167), (167, 168), (168, 169), (169, 170), (170, 171), (171, 172), (172, 173), (173, 174), (174, 175), (175, 176), (176, 177), (177, 178), (178, 179), (179, 180), (180, 181), (181, 182), (182, 183), (183, 184), (184, 185), (185, 186), (186, 187), (187, 188), (188, 189), (189, 190), (190, 191), (191, 192), (192, 193), (193, 194), (194, 195), (195, 196), (196, 197), (197, 198), (198, 199), (199, 200), (200, 201), (201, 202), (202, 203), (203, 204), (204, 205), (205, 206), (206, 207), (207, 208), (208, 209), (209, 210), (210, 211), (211, 212), (212, 213), (213, 214), (214, 215), (215, 216), (216, 217), (217, 218), (218, 219), (219, 220), (220, 221), (221, 222), (222, 223), (223, 224), (224, 225), (225, 226), (226, 227), (227, 228), (228, 229), (229, 230), (230, 231), (231, 232), (232, 233), (233, 234), (234, 235), (235, 236), (236, 237), (237, 238), (238, 239), (239, 240), (240, 241), (241, 242), (242, 243), (243, 244), (244, 245), (245, 246), (246, 247), (247, 248), (248, 249), (249, 250), (250, 251), (251, 252), (252, 253), (253, 254), (254, 255), (255, 256), (256, 257), (257, 258), (258, 259), (259, 260), (260, 261), (261, 262), (262, 263), (263, 264), (264, 265), (265, 266), (266, 267), (267, 268), (268, 269), (269, 270), (270, 271), (271, 272), (272, 273), (273, 274), (274, 275), (275, 276), (276, 277), (277, 278), (278, 279), (279, 280), (280, 281), (281, 282), (282, 283), (283, 284), (284, 285), (285, 286), (286, 287), (287, 288), (288, 289), (289, 290), (290, 291), (291, 292), (292, 293), (293, 294), (294, 295), (295, 296), (296, 297), (297, 298), (298, 299), (299, 300), (300, 301), (301, 302), (302, 303), (303, 304), (304, 305), (305, 306), (306, 307), (307, 308), (308, 309), (309, 310), (310, 311), (311, 312), (312, 313), (313, 314), (314, 315), (315, 316), (316, 317), (317, 318), (318, 319), (319, 320), (320, 321), (321, 322), (322, 323), (323, 324), (324, 325), (325, 326), (326, 327), (327, 328), (328, 329), (329, 330), (330, 331), (331, 332), (332, 333), (333, 334), (334, 335), (335, 336), (336, 337), (337, 338), (338, 339), (339, 340), (340, 341), (341, 342), (342, 343), (343, 344), (344, 345), (345, 346), (346, 347), (347, 348), (348, 349), (349, 350), (350, 351), (351, 352), (352, 353), (353, 354), (354, 355), (355, 356), (356, 357), (357, 358), (358, 359), (359, 360), (360, 361), (361, 362), (362, 363), (363, 364), (364, 365), (365, 366), (366, 367), (367, 368), (368, 369), (369, 370), (370, 371), (371, 372), (372, 373), (373, 374), (374, 375), (375, 376), (376, 377), (377, 378), (378, 379), (379, 380), (380, 381), (381, 382), (382, 383), (383, 384), (384, 385), (385, 386), (386, 387), (387, 388), (388, 389), (389, 390), (390, 391), (391, 392), (392, 393), (393, 394), (394, 395), (395, 396), (396, 397), (397, 398), (398, 399), (399, 400), (400, 401), (401, 402), (402, 403), (403, 404), (404, 405), (405, 406), (406, 407), (407, 408), (408, 409), (409, 410), (410, 411), (411, 412), (412, 413), (413, 414), (414, 415), (415, 416), (416, 417), (417, 418), (418, 419), (419, 420), (420, 421), (421, 422), (422, 423), (423, 424), (424, 425), (425, 426), (426, 427), (427, 428), (428, 429), (429, 430), (430, 431), (431, 432), (432, 433), (433, 434), (434, 435), (435, 436), (436, 437), (437, 438), (438, 439), (439, 440), (440, 441), (441, 442), (442, 443), (443, 444), (444, 445), (445, 446), (446, 447), (447, 448), (448, 449), (449, 450), (450, 451), (451, 452), (452, 453), (453, 454), (454, 455), (455, 456), (456, 457), (457, 458), (458, 459), (459, 460), (460, 461), (461, 462), (462, 463), (463, 464), (464, 465), (465, 466), (466, 467), (467, 468), (468, 469), (469, 470), (470, 471), (471, 472), (472, 473), (473, 474), (474, 475), (475, 476), (476, 477), (477, 478), (478, 479), (479, 480), (480, 481), (481, 482), (482, 483), (483, 484), (484, 485), (485, 486), (486, 487), (487, 488), (488, 489), (489, 490), (490, 491), (491, 492), (492, 493), (493, 494), (494, 495), (495, 496), (496, 497), (497, 498), (498, 499), (499, 500), (500, 501), (501, 502), (502, 503), (503, 504), (504, 505), (505, 506), (506, 507), (507, 508), (508, 509), (509, 510), (510, 511), (511, 512), (512, 513), (513, 514), (514, 515), (515, 516), (516, 517), (517, 518), (518, 519), (519, 520), (520, 521), (521, 522), (522, 523), (523, 524), (524, 525), (525, 526), (526, 527), (527, 528), (528, 529), (529, 530), (530, 531), (531, 532), (532, 533), (533, 534), (534, 535), (535, 536), (536, 537), (537, 538), (538, 539), (539, 540), (540, 541), (541, 542), (542, 543), (543, 544), (544, 545), (545, 546), (546, 547), (547, 548), (548, 549), (549, 550), (550, 551), (551, 552), (552, 553), (553, 554), (554, 555), (555, 556), (556, 557), (557, 558), (558, 559), (559, 560), (560, 561), (561, 562), (562, 563), (563, 564), (564, 565), (565, 566), (566, 567), (567, 568), (568, 569), (569, 570), (570, 571), (571, 572), (572, 573), (573, 574), (574, 575), (575, 576), (576, 577), (577, 578), (578, 579), (579, 580), (580, 581), (581, 582), (582, 583), (583, 584), (584, 585), (585, 586), (586, 587), (587, 588), (588, 589), (589, 590), (590, 591), (591, 592), (592, 593), (593, 594), (594, 595), (595, 596), (596, 597), (597, 598), (598, 599), (599, 600), (600, 601), (601, 602), (602, 603), (603, 604), (604, 605), (605, 606), (606, 607), (607, 608), (608, 609), (609, 610), (610, 611), (611, 612), (612, 613), (613, 614), (614, 615), (615, 616), (616, 617), (617, 618), (618, 619), (619, 620), (620, 621), (621, 622), (622, 623), (623, 624), (624, 625), (625, 626), (626, 627), (627, 628), (628, 629), (629, 630), (630, 631), (631, 632), (632, 633), (633, 634), (634, 635), (635, 636), (636, 637), (637, 638), (638, 639), (639, 640), (640, 641), (641, 642), (642, 643), (643, 644), (644, 645), (645, 646), (646, 647), (647, 648), (648, 649), (649, 650), (650, 651), (651, 652), (652, 653), (653, 654), (654, 655), (655, 656), (656, 657), (657, 658), (658, 659), (659, 660), (660, 661), (661, 662), (662, 663), (663, 664), (664, 665), (665, 666), (666, 667), (667, 668), (668, 669), (669, 670), (670, 671), (671, 672), (672, 673), (673, 674), (674, 675)]
        print(pos_edge_list)
        pos_edge_index = self.edge_list_to_tensor(pos_edge_list)
        # tensor([[  0,   1,   2,  ..., 672, 673, 674],
        #         [  1,   2,   3,  ..., 673, 674, 675]])
        print(pos_edge_index)
        # print("The pos_edge_index tensor has shape {}".format(pos_edge_index.shape))
        # print("The pos_edge_index tensor has sum value {}".format(torch.sum(pos_edge_index)))

        # Sample 78 negative edges
        neg_edge_list = self.sample_negative_edges(graph, len(pos_edge_list))
        # [(424, 417), (548, 93), (373, 465), (616, 307), (546, 161), (5, 167), (373, 87), (590, 453), (70, 295), (233, 201), (69, 398), (525, 142), (371, 18), (194, 623), (421, 654), (594, 357), (200, 117), (156, 340), (157, 588), (451, 149), (20, 664), (177, 214), (128, 140), (580, 309), (173, 130), (337, 376), (152, 561), (407, 579), (271, 668), (624, 196), (393, 72), (423, 312), (44, 374), (501, 196), (201, 92), (654, 158), (379, 140), (507, 363), (340, 33), (396, 37), (187, 402), (586, 282), (292, 448), (587, 142), (542, 218), (63, 150), (487, 323), (324, 580), (13, 112), (122, 342), (643, 299), (420, 96), (402, 595), (332, 310), (199, 299), (113, 452), (110, 480), (0, 155), (411, 530), (404, 394), (186, 492), (421, 466), (426, 152), (2, 161), (401, 27), (119, 14), (498, 368), (169, 432), (107, 452), (592, 358), (565, 177), (551, 445), (431, 266), (656, 335), (235, 127), (224, 258), (381, 570), (532, 172), (550, 472), (371, 455), (393, 245), (445, 653), (478, 12), (214, 429), (171, 99), (594, 46), (247, 294), (334, 67), (390, 291), (509, 208), (644, 355), (291, 282), (66, 622), (100, 520), (189, 14), (190, 440), (555, 672), (513, 455), (274, 248), (118, 77), (485, 285), (616, 300), (157, 60), (167, 66), (352, 8), (140, 74), (317, 305), (204, 516), (11, 518), (547, 249), (553, 272), (32, 496), (232, 520), (588, 113), (371, 24), (221, 109), (583, 201), (64, 117), (152, 209), (396, 385), (64, 379), (6, 601), (497, 56), (593, 135), (217, 631), (536, 2), (332, 213), (224, 474), (208, 326), (243, 395), (3, 451), (440, 669), (533, 300), (453, 237), (0, 401), (258, 438), (323, 8), (339, 269), (397, 50), (584, 352), (388, 101), (369, 190), (232, 372), (207, 318), (589, 195), (659, 235), (442, 157), (425, 288), (251, 302), (661, 64), (33, 502), (163, 526), (468, 527), (133, 521), (334, 165), (561, 553), (445, 300), (421, 72), (417, 315), (270, 179), (357, 186), (301, 444), (1, 360), (556, 386), (40, 301), (294, 8), (62, 233), (508, 630), (446, 137), (574, 298), (266, 602), (62, 64), (475, 529), (516, 667), (255, 379), (606, 256), (298, 131), (235, 404), (65, 368), (290, 347), (124, 82), (528, 612), (379, 560), (6, 239), (390, 280), (67, 504), (127, 102), (407, 492), (447, 521), (545, 421), (228, 140), (92, 167), (265, 613), (68, 269), (565, 507), (138, 520), (295, 123), (23, 379), (641, 73), (420, 299), (77, 87), (486, 572), (310, 180), (411, 170), (53, 65), (441, 347), (69, 522), (585, 40), (465, 494), (530, 257), (622, 459), (92, 152), (534, 655), (673, 224), (288, 11), (338, 625), (193, 351), (289, 543), (98, 568), (201, 270), (511, 433), (288, 179), (266, 576), (133, 93), (355, 234), (266, 77), (494, 553), (70, 534), (501, 99), (475, 426), (134, 93), (347, 420), (437, 276), (419, 626), (406, 60), (563, 616), (478, 178), (362, 644), (171, 214), (424, 66), (605, 168), (597, 377), (236, 344), (52, 285), (538, 362), (512, 311), (320, 72), (495, 470), (324, 650), (265, 170), (517, 494), (297, 32), (60, 103), (33, 591), (537, 167), (39, 601), (276, 335), (584, 674), (284, 193), (61, 164), (226, 589), (442, 646), (570, 503), (67, 605), (25, 200), (474, 648), (57, 110), (599, 616), (662, 145), (210, 392), (603, 464), (490, 417), (24, 48), (543, 458), (274, 67), (431, 61), (159, 44), (30, 155), (532, 413), (179, 561), (632, 311), (371, 584), (566, 41), (43, 93), (234, 211), (229, 121), (260, 487), (597, 378), (370, 647), (287, 443), (22, 254), (335, 293), (458, 493), (261, 173), (217, 188), (619, 342), (173, 0), (235, 0), (567, 78), (85, 214), (523, 309), (180, 139), (588, 473), (322, 516), (577, 118), (649, 529), (225, 130), (569, 354), (295, 190), (612, 26), (543, 595), (286, 42), (523, 476), (372, 286), (249, 472), (451, 48), (304, 579), (219, 190), (349, 6), (30, 437), (587, 275), (41, 171), (673, 537), (386, 134), (54, 149), (292, 524), (647, 360), (443, 168), (357, 417), (674, 205), (94, 504), (302, 436), (332, 125), (6, 590), (114, 22), (346, 585), (646, 412), (210, 622), (115, 126), (74, 661), (393, 283), (451, 265), (548, 536), (182, 417), (101, 419), (582, 467), (69, 83), (397, 671), (284, 673), (361, 647), (573, 541), (518, 212), (38, 170), (467, 350), (311, 30), (545, 465), (367, 605), (600, 397), (581, 424), (53, 271), (190, 354), (632, 21), (331, 90), (289, 457), (604, 88), (560, 438), (63, 357), (379, 253), (599, 531), (592, 13), (476, 246), (408, 419), (97, 272), (183, 550), (91, 175), (120, 656), (596, 432), (111, 299), (121, 560), (130, 299), (346, 644), (335, 501), (209, 653), (249, 168), (377, 641), (256, 394), (322, 193), (129, 631), (316, 80), (101, 494), (121, 263), (442, 277), (252, 560), (242, 181), (308, 3), (482, 448), (219, 611), (166, 183), (229, 95), (561, 79), (228, 239), (660, 427), (388, 232), (564, 2), (510, 523), (357, 1), (345, 474), (60, 81), (403, 180), (289, 501), (557, 24), (124, 655), (214, 323), (488, 550), (339, 329), (169, 353), (599, 369), (579, 623), (381, 202), (585, 206), (132, 101), (136, 182), (632, 451), (195, 313), (224, 532), (440, 198), (350, 442), (394, 21), (126, 497), (603, 128), (401, 348), (349, 165), (29, 304), (382, 530), (522, 212), (666, 651), (516, 506), (619, 460), (303, 420), (262, 319), (512, 232), (445, 529), (10, 353), (200, 114), (319, 256), (94, 21), (298, 657), (340, 134), (329, 673), (432, 556), (669, 52), (270, 30), (573, 536), (499, 334), (200, 575), (633, 375), (149, 544), (69, 587), (543, 527), (411, 626), (495, 570), (320, 394), (209, 3), (662, 50), (146, 491), (646, 221), (533, 48), (11, 8), (196, 595), (104, 313), (1, 645), (327, 118), (521, 363), (641, 4), (554, 179), (440, 563), (285, 661), (7, 233), (390, 80), (473, 671), (26, 48), (583, 568), (108, 390), (550, 60), (410, 471), (418, 468), (522, 519), (521, 358), (147, 485), (7, 317), (426, 4), (577, 337), (522, 556), (37, 18), (419, 418), (665, 472), (83, 459), (511, 647), (311, 267), (172, 281), (351, 306), (118, 560), (153, 383), (57, 646), (195, 341), (422, 99), (143, 455), (321, 146), (357, 135), (96, 182), (590, 68), (336, 483), (309, 641), (306, 75), (573, 110), (250, 471), (373, 474), (135, 201), (245, 269), (425, 454), (370, 197), (306, 334), (575, 403), (224, 252), (259, 21), (330, 86), (473, 205), (241, 638), (626, 482), (616, 536), (487, 63), (255, 601), (264, 530), (303, 423), (610, 614), (184, 154), (362, 178), (132, 356), (274, 229), (306, 54), (169, 370), (61, 665), (477, 280), (277, 225), (369, 297), (206, 277), (144, 660), (383, 44), (377, 540), (620, 256), (392, 314), (118, 431), (387, 72), (142, 233), (185, 230), (217, 570), (425, 272), (309, 286), (388, 248), (137, 615), (658, 93), (124, 658), (470, 237), (458, 624), (384, 346), (403, 411), (398, 232), (465, 430), (593, 368), (297, 212), (81, 323), (341, 429), (564, 135), (20, 184), (614, 616), (595, 401), (68, 99), (89, 84), (253, 397), (55, 289), (651, 589), (662, 87), (517, 629), (298, 93), (397, 17), (312, 374), (237, 128), (449, 170), (432, 570), (26, 338), (587, 79), (397, 240), (319, 514), (636, 245), (61, 368), (560, 190), (152, 235), (572, 565), (146, 354), (608, 623), (384, 447), (483, 502), (664, 292), (505, 523), (400, 274), (246, 50), (639, 642), (329, 569), (43, 498), (13, 520), (524, 321), (591, 628), (231, 337), (197, 100), (229, 344), (601, 252), (563, 615), (444, 194), (34, 583), (470, 617), (77, 186), (561, 435), (574, 96), (629, 311), (71, 373), (361, 503), (373, 136), (268, 574), (76, 548), (500, 621), (625, 580), (181, 112), (630, 106), (60, 126), (611, 413), (236, 186), (308, 380), (114, 241), (668, 156), (265, 225), (272, 390), (98, 495), (186, 238), (315, 441), (518, 411), (250, 285), (393, 59), (658, 212), (324, 243), (68, 249), (173, 180), (424, 447), (216, 237), (112, 121), (116, 484), (196, 298), (506, 8), (354, 601), (635, 340), (441, 120), (178, 116), (395, 182), (632, 51), (389, 597), (669, 152), (562, 143), (325, 537), (668, 431), (96, 41), (567, 408), (633, 454), (441, 172), (616, 480), (76, 588), (364, 231)]
        print(neg_edge_list)
        # Transform the negative edge list to tensor
        neg_edge_index = self.edge_list_to_tensor(neg_edge_list)
        # tensor([[424, 548, 373,  ..., 616,  76, 364],
        #         [417,  93, 465,  ..., 480, 588, 231]])
        print(neg_edge_index)
        # print("The neg_edge_index tensor has shape {}".format(neg_edge_index.shape))
        '''
        # Which of following edges can be negative ones?
        edge_1 = (7, 1)
        edge_2 = (1, 33)
        edge_3 = (33, 22)
        edge_4 = (0, 4)
        edge_5 = (4, 2)
        ## 1: For each of the 5 edges, print whether it can be negative edge
        # 如果边在图中，就认为不行
        print('edge_1' + (" can't" if graph.has_edge(edge_1[0], edge_1[1]) else ' can') + ' be negative edge')
        print('edge_2' + (" can't" if graph.has_edge(edge_2[0], edge_2[1]) else ' can') + ' be negative edge')
        print('edge_3' + (" can't" if graph.has_edge(edge_3[0], edge_3[1]) else ' can') + ' be negative edge')
        print('edge_4' + (" can't" if graph.has_edge(edge_4[0], edge_4[1]) else ' can') + ' be negative edge')
        print('edge_5' + (" can't" if graph.has_edge(edge_5[0], edge_5[1]) else ' can') + ' be negative edge')
        '''
        num_nodes = graph.number_of_nodes()
        emb = self.create_node_emb(num_nodes, embedding_dim)
        # ids = torch.LongTensor([0, 3])
        # Print the embedding layer
        # print("Embedding: {}".format(emb))
        # An example that gets the embeddings for node 0 and 3
        # print(emb(ids))
        # Visualize the initial random embeddding
        self.visualize_emb(graph, emb)

        loss_fn = nn.BCELoss()
        sigmoid = nn.Sigmoid()

        # Generate the positive and negative labels
        pos_label = torch.ones(pos_edge_index.shape[1],)    # torch.tensor([1., 1., 1.])

        # for graph
        #     pos_label.append()
        # pos_label = torch.tensor(pos_label)

        # print('pos_label')
        # print(pos_label)
        neg_label = torch.zeros(neg_edge_index.shape[1],)    # torch.tensor([0., 0., 0.])
        # print('neg_label')
        # print(neg_label)
        # Concat positive and negative labels into one tensor
        train_label = torch.cat([pos_label, neg_label], dim=0)
        # print('train_label')
        # print(train_label)
        # Concat positive and negative edges into one tensor
        # Since the network is very small, we do not split the edges into val/test sets
        train_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        # print('train_edge')
        # print(train_edge)

        self.train(emb, loss_fn, sigmoid, train_label, train_edge)
        # Visualize the final learned embedding
        self.visualize_emb(graph, emb)
        X = emb.weight.data.numpy()
        print(X)
        # print(len(emb))
        print(X[0])
        return graph


class GraphSampling:
    def __init__(self, graph_datasets):
        self.graph_datasets = graph_datasets

    def sampling(self, k_hop):
        for i in range(0, len(self.graph_datasets)):
            graph = self.graph_datasets[i]
            '''
            print(graph.is_directed())
            for node in graph.nodes(data=True):
                print(node)
            '''


if __name__ == '__main__':
    DataProcess().get_lable_videos_name_dict()
    DataProcess().split_dataset()
    GraphGenerate_nx().video_to_graph_nx_three(name='train')

    '''
    videos = [{"name": "g", "frame": [[[0, 1], 't00'], [[1, 2], 't01'], [[2, 3], 't02']]},
              {"name": "e", "frame": [[[1, 2], 't1']]},
              {"name": "a", "frame": [[[2, 3, 2], 't2']]},
              ]
    
    videos = [{"name": "g", "frame": [[[0, 0], 't00'], [[1, 1], 't01'], [[2, 2], 't02'], [[3, 3], 't03']]},
              {"name": "e", "frame": [[[3, 3], 't10'], [[4, 4], 't11'], [[5, 5], 't12'], [[6, 6], 't13']]},
              {"name": "a", "frame": [[[6, 6], 't20'], [[7, 7], 't21'], [[8, 8], 't22'], [[9, 9], 't23']]}]
    
    print("PyTorch has version {}".format(torch.__version__))
    print(torch.cuda.is_available())
    print(torch_geometric.__version__)

    videos = DataProcess().load_dataset('test')
    # print(len(videos))
    graph_datasets = GraphGenerate_nx(videos=videos).frame_to_graph()     # [videos[0]]
    print(graph_datasets)
    # for graph in graph_datasets:
    # print(type(graph))
    # nx.draw(graph, with_labels=False)
    # NodeEmebedding().node_embedding(graph, 8)
    '''
    '''
    DeepSNAP_graph_dataset = GraphGenerate(videos).networkx_to_DeepSNAP_graph(graph_datasets)
    # print(DeepSNAP_graph_dataset[0])
    # Graph(G=[], Name=[], edge_index=[2, 46], edge_label_index=[2, 46], node_feature=[47, 2048], node_label=[47],
    #      node_label_index=[47], task=[], weight=[46])

    dataset_train, dataset_val, dataset_test = DeepSNAP_graph_dataset.split(transductive=False, split_ratio=[0.8, 0.1, 0.1])
    
    print(train, val, test)
    
    GraphDataset(514) GraphDataset(64) GraphDataset(65)
    
    dataset_train, dataset_val, dataset_test = DeepSNAP_graph_dataset.split(transductive=True, split_ratio=[0.8, 0.1, 0.1])
    
    print("Cora train dataset: {}".format(dataset_train))
    print("Cora validation dataset: {}".format(dataset_val))
    print("Cora test dataset: {}".format(dataset_test))
    print("Original Cora has {} nodes".format(DeepSNAP_graph_dataset.num_nodes[1]))#
    # The nodes in each set can be find in node_label_index
    print("After the split, Cora has {} training nodes".format(dataset_train[1].node_label_index.shape[0]))
    print("After the split, Cora has {} validation nodes".format(dataset_val[1].node_label_index.shape[0]))
    print("After the split, Cora has {} test nodes".format(dataset_test[1].node_label_index.shape[0]))
    
    Cora train dataset: GraphDataset(643)
    Cora validation dataset: GraphDataset(643)
    Cora test dataset: GraphDataset(643)
    Original Cora has 47 nodes
    After the split, Cora has 37 training nodes
    After the split, Cora has 4 validation nodes
    After the split, Cora has 6 test nodes
    '''

    # num_classes = graph_datasets.num_classes
    # num_features = graph_datasets.num_features
    #name = 'ENZYMES'
    # print("{} dataset has {} classes".format(name, num_classes))ex
    # print("{} dataset has {} features".format(name, num_features))

    # GraphSampling(graph_datasets=graph_datasets).sampling(k_hop=1)


