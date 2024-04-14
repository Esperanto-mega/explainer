import torch
from torch.nn import ReLU, Linear
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

class GraphGCN(torch.nn.Module):
    """
    A graph clasification model for graphs decribed in https://arxiv.org/abs/1903.03894.
    This model consists of 3 stacked GCN layers followed by a linear layer.
    In between the GCN outputs and linear layers are pooling operations in both mean and max.
    """
    def __init__(self, num_features, num_classes):
        super(GraphGCN, self).__init__()
        self.embedding_size = 20
        self.conv1 = GCNConv(num_features, 20)
        self.relu1 = ReLU()
        self.conv2 = GCNConv(20, 20)
        self.relu2 = ReLU()
        self.conv3 = GCNConv(20, 20)
        self.relu3 = ReLU()
        self.lin = Linear(self.embedding_size * 2, num_classes)

    def forward(self, x, edge_index, batch = None, edge_weights = None):
        if batch is None: # No batch given
            batch = torch.zeros(x.size(0), dtype = torch.long)
        embed = self.embedding(x, edge_index, edge_weights)

        out1 = global_max_pool(embed, batch)
        out2 = global_mean_pool(embed, batch)
        input_lin = torch.cat([out1, out2], dim = -1)

        out = self.lin(input_lin)
        return out

    def embedding(self, x, edge_index, edge_weights = None):
        if edge_weights is None:
            edge_weights = torch.ones(edge_index.size(1))
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = torch.nn.functional.normalize(out1, p = 2, dim = 1)
        out1 = self.relu1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = torch.nn.functional.normalize(out2, p = 2, dim = 1)
        out2 = self.relu2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = torch.nn.functional.normalize(out3, p = 2, dim = 1)
        out3 = self.relu3(out3)

        input_lin = out3

        return input_lin
# GNN model
# Define a simple GCN.
class GCN(torch.nn.Module):
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
