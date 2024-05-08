# @yinjun

import torch
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

class GraphGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)
        # self.linear = torch.nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.normalize(x, p = 2, dim = 1)
        x = x.relu()
      
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.normalize(x, p = 2, dim = 1)
        x = x.relu()
      
        x = self.conv3(x, edge_index)
        x = torch.nn.functional.normalize(x, p = 2, dim = 1)
        x = x.relu()
      
        x_mean = global_mean_pool(x, batch)
        # x_max = global_max_pool(x, batch)
        # x = torch.cat([x_mean, x_max], dim = -1)
        
        x = self.linear(x)
        return x
