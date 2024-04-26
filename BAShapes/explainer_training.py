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

parser = argparse.ArgumentParser(description = 'PGExplainer')
parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
parser.add_argument("--data_path", type = str, default = '', help = "Root directory where the dataset should be saved")
parser.add_argument("--model_path", type = str, default = '', help = "Model to be explained")
parser.add_argument("--device", type = str, default = 'cuda:0', help = "")
parser.add_argument("--train_ratio", type = float, default = 0.9, help = "")
parser.add_argument("--epochs", type = int, default = 30, help = "Explainer training epochs")
parser.add_argument("--eval_step", type = int, default = 5, help = "Explainer validation steps")
parser.add_argument("--lr", type = float, default = 5e-3, help = "Explainer learning rate")
parser.add_argument("--batch_size", type = int, default = 64, help = "Explainer training batchsize")

parser.add_argument("--explanation_type", type = str, default = 'phenomenon')
# 'model': Explain the model prediction.
# 'phenomenon': Explain the phenomenon that the model is trying to predict.

parser.add_argument("--node_mask_type", type = str, default = 'attributes')

parser.add_argument("--edge_mask_type", type = str, default = 'object')
# None: Will not apply any mask on nodes.
# 'object': Will mask each edge.
# "common_attributes": Will mask each edge feature.
# "attributes": Will mask each feature across all edges.

parser.add_argument("--model_mode", type = str, default = 'multiclass_classification')
# "binary_classification": A binary classification model.
# "multiclass_classification": A multiclass classification model.
# "regression": A regression model.

parser.add_argument("--model_task_level", type = str, default = 'node')
# "node": A node-level prediction model.
# "edge": An edge-level prediction model.
# "graph": A graph-level prediction model.

parser.add_argument("--model_return_type", type = str, default = 'raw')
# "raw": The model returns raw values.
# "probs": The model returns probabilities.
# "log_probs": The model returns log-probabilities.

args = parser.parse_args()
print(args)

device = torch.device(args.device)

# dataset & model
dataset = InMemoryDataset()
dataset.load(args.data_path)
data = dataset[0]
data = data.to(device)

ckpt = torch.load(args.model_path)
model_args, model_state_dict = ckpt['args'], ckpt['state_dict']
explained_model  = GCN(
    data.num_node_features,
    hidden_channels = model_args.hidden_dim,
    num_layers = model_args.num_layers,
    out_channels = dataset.num_classes
).to(device)
explained_model.load_state_dict(model_state_dict)
explained_model.to(device)
explained_model.eval()

# explainer
model_config = {
    'mode': args.model_mode,
    'task_level': args.model_task_level,
    'return_type': args.model_return_type
}
explainer = Explainer(
    model = explained_model,
    algorithm = GNNExplainer(epochs = args.epochs),
    explanation_type = args.explanation_type,
    node_mask_type = args.node_mask_type,
    edge_mask_type = args.edge_mask_type,
    model_config = model_config
)

targets, preds = [], []
node_indices = range(400, data.num_nodes, 5)
for node_index in tqdm(node_indices, leave=False, desc='Train Explainer'):
    target = data.y if explanation_type == 'phenomenon' else None
    explanation = explainer(data.x, data.edge_index, index = node_index, target = target)
    _, _, _, hard_edge_mask = k_hop_subgraph(node_index, num_hops = 3, edge_index = data.edge_index)
    targets.append(data.edge_mask[hard_edge_mask].cpu())
    preds.append(explanation.edge_mask[hard_edge_mask].cpu())

auc = roc_auc_score(torch.cat(targets), torch.cat(preds))
print(f'Mean ROC AUC (explanation type {explanation_type:10}): {auc:.4f}')
