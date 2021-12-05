import argparse

import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.utils import to_undirected

from ogb.nodeproppred import PygNodePropPredDataset

from core.config import cfg
from core.graph import GraphGenerate_pyg, GraphGenerate_nx
from deepsnap.graph import Graph
import numpy as np
from core.data_process import DataProcess


class NodeEmebeddingNode2Vec:
    def __init__(self):
        # Please do not change / reset the random seed
        torch.manual_seed(1)

    def save_embedding(self, model):
        torch.save(model.embedding.weight.data.cpu(), './embedding.pt')

    def NodeEmebedding(self, data):
        parser = argparse.ArgumentParser(description='OGBN-Arxiv (Node2Vec)')
        parser.add_argument('--device', type=int, default=0)
        parser.add_argument('--embedding_dim', type=int, default=128)  # default=128
        parser.add_argument('--walk_length', type=int, default=10)  # default=80
        parser.add_argument('--context_size', type=int, default=20)  # default=20
        parser.add_argument('--walks_per_node', type=int, default=10)  # default=10
        parser.add_argument('--batch_size', type=int, default=8)  # default=256
        parser.add_argument('--lr', type=float, default=0.004)
        parser.add_argument('--epochs', type=int, default=100)  # default=5
        parser.add_argument('--log_steps', type=int, default=0)
        args = parser.parse_args()

        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)

        # data = garphdata

        # dataset = PygNodePropPredDataset(name='ogbn-arxiv')
        # data = dataset[0]
        # Data(edge_index=[2, 1166243], x=[169343, 128], node_year=[169343, 1], y=[169343, 1])
        # print(data.edge_index)
        # print(data.num_nodes)

        # data.edge_index = to_undirected(data.edge_index, data.num_nodes)

        model = Node2Vec(data.edge_index, args.embedding_dim, args.walk_length,
                         args.context_size, args.walks_per_node,
                         sparse=True).to(device)

        loader = model.loader(batch_size=args.batch_size, shuffle=True, num_workers=4)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=args.lr)

        model.train()

        best_loss = 100
        best_model = None
        best = [0, 0, 0.]
        for epoch in range(1, args.epochs + 1):
            for i, (pos_rw, neg_rw) in enumerate(loader):
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                '''
                if (i + 1) % args.log_steps == 0:
                    print(f'Epoch: {epoch:02d}, Step: {i + 1:03d}/{len(loader)}, '
                          f'Loss: {loss:.4f}')
                '''
                # # if (i + 1) % 100 == 0:  # Save model every 100 steps.
                # self.save_embedding(model)
                # print(model.embedding.weight.data.numpy())
                # self.save_embedding(model)
                # print(model.embedding.weight.data.cpu())
                if loss < best_loss:
                    best_loss = loss
                    best[0] = epoch
                    best[1] = i + 1
                    best[2] = loss
                    best_model = model.embedding.weight.data.cpu()
        print(f'best_Epoch: {best[0]:02d}, best_Step: {best[1] + 1:03d}/{len(loader)}, '
              f'best_Loss: {best[2]:.4f}')
        # return model.embedding.weight.data.cpu()
        return best_model


if __name__ == "__main__":
    # dataset = PygNodePropPredDataset(name='ogbn-arxiv')
    DataProcess().get_lable_videos_name_dict()
    num = 1
    videos_name_path = cfg.DATA_PATH + cfg.dataset + "/train_dataset.txt"    # videos_name
    videos_name = np.loadtxt(videos_name_path, dtype=str)
    for video_name in videos_name[0:20]:
    # for video_name in videos_name:
        # print(video_name)
        print(num)
        # graph = GraphGenerate_nx().frame_to_graph(video_name)
        graph = GraphGenerate_nx().frame_to_graph_edges(video_name)
        emb = NodeEmebeddingNode2Vec().NodeEmebedding(data=Graph(graph))
        # print(emb.shape)
        num += 1
        np.save('/home/cpslabzjb/zjb/projects/zjb/GCL/data/'+cfg.dataset+"/emb/Node2Vec_edges_all/"+video_name, emb)
        # np.savetxt(cfg.DATA_PATH + cfg.dataset + "/emb/Node2Vec/" + video_name, np.array(emb))
        # np.array(emb).tofile(cfg.DATA_PATH+cfg.dataset+"/emb/Node2Vec/"+video_name+".bin")
        # f_ptr = open(cfg.DATA_PATH+cfg.dataset+"/emb/Node2Vec/" + video_name, "w")
        # f_ptr.write(' '.join(int(emb)))
        # f_ptr.close()
