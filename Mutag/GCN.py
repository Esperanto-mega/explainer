# @yinjun

import torch
from torch.nn import ModuleList
from torch.nn import Sequential
from torch.nn import ReLU, Linear, Softmax
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool
from torch_geometric.nn import GINEConv, BatchNorm

class MutagNet(torch.nn.Module):
    def __init__(self, conv_unit = 2):
        super(MutagNet, self).__init__()

        self.node_emb = Linear(14, 32)
        self.edge_emb = Linear(3, 32)
        self.relu_nn = ModuleList([ReLU() for i in range(conv_unit)])

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.relus = ModuleList()

        for i in range(conv_unit):
            conv = GINEConv( nn = Sequential(Linear(32, 75), self.relu_nn[i], Linear(75, 32)))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(32))
            self.relus.append(ReLU())

        self.lin1 = Linear(32, 16)
        self.relu = ReLU()
        self.lin2 = Linear(16, 2)
        self.softmax = Softmax(dim=1)

    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return self.get_pred(graph_x)

    def get_node_reps(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)
        for conv, batch_norm, ReLU in \
                zip(self.convs, self.batch_norms, self.relus):
            x = conv(x, edge_index, edge_attr)
            x = ReLU(batch_norm(x))
        node_x = x
        return node_x

    def get_graph_rep(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)
        return graph_x

    def get_pred(self, graph_x):
        pred = self.relu(self.lin1(graph_x))
        pred = self.lin2(pred)
        self.readout = self.softmax(pred)
        return pred

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)
