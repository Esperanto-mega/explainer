# yinjun@2024/04/13

import torch
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

# GNN model
# Define a simple GCN.
class GraphGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # (num_features, 20)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # (20, 20)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        # (20, 20)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        # (20 * 2, num_classes)
        self.linear = torch.nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, data):
        # x: Node feature matrix of shape [num_nodes, in_channels]
        # edge_index: Graph connectivity matrix of shape [2, num_edges]
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.normalize(x, p = 2, dim = 1)
        x = x.relu()
        # (num_features -> 20)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.normalize(x, p = 2, dim = 1)
        x = x.relu()
        # (20 -> 20)
        x = self.conv3(x, edge_index)
        x = torch.nn.functional.normalize(x, p = 2, dim = 1)
        x = x.relu()
        # (20 -> 20)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        # (node -> graph)
        x = torch.cat([x_mean, x_max], dim = -1)
        x = self.linear(x)
        # (20 * 2 -> num_classes)
        
        return x
