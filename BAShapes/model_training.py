import random
import argparse
import numpy as np

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GCN
from torch_geometric.utils import k_hop_subgraph

parser = argparse.ArgumentParser(description = 'BAShapes')
parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
parser.add_argument("--device", type = str, default = 'cuda:0', help = "")
parser.add_argument("--lr", type = float, default = 0.001, help = "")
parser.add_argument("--weight_decay", type = float, default = 0.005, help = "")
parser.add_argument("--train_size", type = float, default = 0.8, help = "")
parser.add_argument("--epochs", type = int, default = 3000, help = "")
parser.add_argument("--eval_step", type = int, default = 200, help = "")
parser.add_argument("--hidden_dim", type = int, default = 20, help = "")
parser.add_argument("--num_layers", type = int, default = 3, help = "")
parser.add_argument("--num_nodes", type = int, default = 300, help = "The number of nodes.")
parser.add_argument("--num_edges", type = int, default = 5, help = "The number of edges from a new node to existing nodes.")
parser.add_argument("--num_motifs", type = int, default = 80, help = "The number of motifs to attach to the graph.")
parser.add_argument("--data_path", type = str, default = '', help = "Root directory where the dataset should be saved")
parser.add_argument("--model_path", type = str, default = '', help = "Root directory where the model should be saved")
args = parser.parse_args()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = False

# torch_geometric.transforms.Constant - Appends a constant value (default: 1.0) to each node feature.
# dataset = ExplainerDataset(
#     graph_generator = BAGraph(num_nodes = args.num_nodes, num_edges = args.num_edges),
#     motif_generator = 'house',
#     num_motifs = args.num_motifs,
#     transform = T.Constant(),
# )
# dataset.save(dataset, args.data_path)
dataset = InMemoryDataset()
dataset.load(args.data_path)
data = dataset[0]

idx = torch.arange(data.num_nodes)
train_idx, test_idx = train_test_split(idx, train_size = args.train_size, stratify = data.y)

device = torch.device(args.device)
data = data.to(device)
model = GCN(
    data.num_node_features,
    hidden_channels = args.hidden_dim,
    num_layers = args.num_layers,
    out_channels = dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_idx], data.y[train_idx])
    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x, data.edge_index).argmax(dim=-1)

    train_correct = int((pred[train_idx] == data.y[train_idx]).sum())
    train_acc = train_correct / train_idx.size(0)

    test_correct = int((pred[test_idx] == data.y[test_idx]).sum())
    test_acc = test_correct / test_idx.size(0)

    return train_acc, test_acc

pbar = tqdm(range(1, args.epochs + 1))
best_acc = 1 / dataset.num_classes
for epoch in pbar:
    loss = train()
    if epoch == 1 or epoch % args.eval_step == 0:
        train_acc, test_acc = test()
        pbar.set_description(f'Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, '
                             f'Test Acc: {test_acc:.4f}')
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'args': args,
                'epoch': epoch + 1,
                'best_accuracy': test_acc,
                'state_dict': model.state_dict()
            }
            torch.save(state, args.model_path)
pbar.close()
model.eval()
