import torch
import torch_geometric.transforms as T
import os
import pandas as pd
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from torch_geometric.utils.convert import to_networkx
import networkx as nx
from tqdm import tqdm
from core.config import cfg


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
    # def __init__(self, transform=None, pre_transform=None):
    #     super().__init__(None, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # return ['some_file_1', 'some_file_2', ...]
        # names = ['x', 'tx', 'allx', 'y', 'ty', 'ally', 'graph', 'test.index']
        # return ['ind.{}.{}'.format(self.dataname.lower(), name) for name in names]
        return [cfg.dataset+'.edges', cfg.dataset+'.graph_idx', cfg.dataset+'.note_lable', cfg.dataset+'.note_feature']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        # download_url(url, self.raw_dir)
        # ...
        # print('download')
        pass

    def process(self):
        # Read data into huge `Data` list.
        # data_list = [...]
        # Read the files' content as Pandas DataFrame. Nodes and graphs ids are based on the file row-index, we adjust
        # the DataFrames indices by starting from 1 instead of 0.
        path = os.path.join(self.raw_dir, cfg.dataset+'.note_feature')
        node_attrs = pd.read_csv(path, sep=',', header=None)
        node_attrs.index += 1
        path = os.path.join(self.raw_dir, cfg.dataset+'.edges')
        edge_index = pd.read_csv(path, sep=',', names=['source', 'target'])
        edge_index.index += 1
        path = os.path.join(self.raw_dir, cfg.dataset+'.graph_idx')
        graph_idx = pd.read_csv(path, sep=',', names=['idx'])     #
        graph_idx.index += 1
        path = os.path.join(self.raw_dir, cfg.dataset+'.note_lable')
        graph_labels = pd.read_csv(path, sep=',', names=['label'])      #
        graph_labels.index += 1
        # In the loop we extract the nodes' embeddings, edges connectivity for and label for a graph, process the
        # information and put it in a Data object, then we add the object to a list
        data_list = []
        ids_list = graph_idx['idx'].unique()
        ########################################
        # for g_idx in tqdm(ids_list[0:2]):
        ########################################
        for g_idx in tqdm(ids_list):
            node_ids = graph_idx.loc[graph_idx['idx'] == g_idx].index
            # Node features
            attributes = node_attrs.loc[node_ids, :]
            # Edges info
            edges = edge_index.loc[edge_index['source'].isin(node_ids)]
            edges_ids = edges.index
            ########################################
            # Graph label
            # label = graph_labels.loc[g_idx]
            label = graph_labels.loc[node_ids[0]: node_ids[-1], :]
            ########################################
            # Normalize the edges indices
            edge_idx = torch.tensor(edges.to_numpy().transpose(), dtype=torch.long)
            map_dict = {v.item(): i for i, v in enumerate(torch.unique(edge_idx))}
            map_edge = torch.zeros_like(edge_idx)
            for k, v in map_dict.items():
                map_edge[edge_idx == k] = v
            # Convert the DataFrames into tensors
            attrs = torch.tensor(attributes.to_numpy(), dtype=torch.float)
            pad = torch.zeros((attrs.shape[0], 0), dtype=torch.float)
            x = torch.cat((attrs, pad), dim=-1)
            edge_idx = map_edge.long()
            np_label = label.to_numpy()
            ###########################################################
            # y = torch.tensor(np_lab if np_lab[0] == 1 else [0], dtype=torch.long)
            y = []
            for i in np_label:
                y.append(cfg.actions_dict[i[0]])
            # graph = Data(x=x, edge_index=edge_idx, y=torch.tensor([[1], [2], [3], [4]]))
            data = Data(x=x, edge_index=edge_idx, y=torch.tensor(y, dtype=torch.long))
            ############################################################
            '''
            if self.pre_filter is not None:
                # print('pre_filter')
                # print(data_list[0])
                data = self.pre_filter(data)
                # print(data_list[0])
            if self.pre_transform is not None:
                # print('pre_transform')
                # print(data_list[0])
                # Data(x=[2852, 2048], edge_index=[2, 2850], y=[2852])
                data = self.pre_transform(data)
                # print(data_list[0])
                # Data(x=[2852, 2048], edge_index=[2, 5702], y=[2852], edge_weight=[5702])
            data_list.append(data)
        # print(data_list)
        torch.save(self.collate(data_list), self.processed_paths[0])
            '''
            # data_list.append(data)
            data_list = [data]

            # datasets = []
            if self.pre_filter is not None:
                # print('pre_filter')
                # print(data_list[0])
                data_list = [data for data in data_list if self.pre_filter(data)]
                # print(data_list[0])
            if self.pre_transform is not None:
                # print('pre_transform')
                # print(data_list[0])
                # Data(x=[2852, 2048], edge_index=[2, 2850], y=[2852])
                data_list = [self.pre_transform(data) for data in data_list]
                # datasets.append(self.pre_transform(data) for data in data_list)
                # print(data_list[0])
                # Data(x=[2852, 2048], edge_index=[2, 5702], y=[2852], edge_weight=[5702])
            torch.save(self.collate(data_list), self.processed_paths[g_idx-1])
        # print(self.pre_transform(data_list))
        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])
        '''
        for i in range(len(data_list)):
            # data, slices = self.collate(dataset)
            # torch.save((data, slices), self.processed_paths[0])
            print(self.processed_dir)
            torch.save(self.collate([data_list[i]]), os.join(self.processed_dir, 'data_{}.pt'.format(i)))
        '''


    def process1(self):
        # Read data into huge `Data` list.
        # data_list = [...]
        # Read the files' content as Pandas DataFrame. Nodes and graphs ids
        # are based on the file row-index, we adjust the DataFrames indices
        # by starting from 1 instead of 0.
        path = os.path.join(self.raw_dir, 'FRANKENSTEIN.node_attrs')
        node_attrs = pd.read_csv(path, sep=',', header=None)
        node_attrs.index += 1
        path = os.path.join(self.raw_dir, 'FRANKENSTEIN.edges')
        edge_index = pd.read_csv(path, sep=',', names=['source', 'target'])
        edge_index.index += 1
        path = os.path.join(self.raw_dir, 'FRANKENSTEIN.graph_idx')
        graph_idx = pd.read_csv(path, sep=',', names=['idx'])
        graph_idx.index += 1
        path = os.path.join(self.raw_dir, 'FRANKENSTEIN.graph_labels')
        graph_labels = pd.read_csv(path, sep=',', names=['label'])
        graph_labels.index += 1
        # In the loop we extract the nodes' embeddings, edges connectivity for
        # and label for a graph, process the information and put it in a Data
        # object, then we add the object to a list
        data_list = []
        ids_list = graph_idx['idx'].unique()
        ########################################
        # for g_idx in tqdm(ids_list[0:1]):
        ########################################
        for g_idx in tqdm(ids_list):
            node_ids = graph_idx.loc[graph_idx['idx'] == g_idx].index
            # Node features
            attributes = node_attrs.loc[node_ids, :]
            # Edges info
            edges = edge_index.loc[edge_index['source'].isin(node_ids)]
            edges_ids = edges.index
            # Graph label
            label = graph_labels.loc[g_idx]
            ########################################
            # label = graph_labels.loc[:node_ids]
            # print(label)
            ########################################
            # Normalize the edges indices
            edge_idx = torch.tensor(edges.to_numpy().transpose(), dtype=torch.long)
            map_dict = {v.item(): i for i, v in enumerate(torch.unique(edge_idx))}
            map_edge = torch.zeros_like(edge_idx)
            for k, v in map_dict.items():
                map_edge[edge_idx == k] = v
            # Convert the DataFrames into tensors
            attrs = torch.tensor(attributes.to_numpy(), dtype=torch.float)
            pad = torch.zeros((attrs.shape[0], 4), dtype=torch.float)
            x = torch.cat((attrs, pad), dim=-1)
            edge_idx = map_edge.long()
            np_lab = label.to_numpy()
            ###########################################################
            # y = torch.tensor(np_lab if np_lab[0] == 1 else [0], dtype=torch.long)
            y = []
            for i in range(0, x.shape[0]):
                y.append(np_lab[0] if np_lab[0] == 1 else 0)
            graph = Data(x=x, edge_index=edge_idx, y=torch.tensor([[1], [2], [3], [4]]))
            # graph = Data(x=x, edge_index=edge_idx, y=torch.tensor(y, dtype=torch.long))
            ############################################################
            data_list.append(graph)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
            # print(data_list)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_paths[0]))      # self.processed_dir
        # data = torch.load(os.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


if __name__ == '__main__':
    # transform = T.Compose([T.AddTrainValTestMask('train_rest', num_val=500, num_test=500), T.TargetIndegree(), ])
    # dataset = MyOwnDataset(root='../FRANKENSTEIN', transform=transform, pre_transform=T.GCNNorm())
    transform = T.Compose([T.RandomNodeSplit('train_rest', num_splits=1, num_train_per_class=10, num_val=2, num_test=2),
                           T.TargetIndegree(), T.ToDevice(device='cuda', attrs=[], non_blocking=True)])
    dataset1 = MyOwnDataset(root='../FRANKENSTEIN', transform=transform, pre_transform=T.GCNNorm())# .process()
    dataset2 = MyOwnDataset(root='../FRANKENSTEIN', transform=None, pre_transform=None).get(0)
    # print(dataset)
    print(dataset1[0])
    dataset = transform(dataset1[0])
    print(dataset)
    data, slices = torch.load('../FRANKENSTEIN/processed/data.pt')
    print(data)
    print(slices)
