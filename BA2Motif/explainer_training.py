# @yinjun

import random
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import BA2MotifDataset
from torch_geometric.explain import Explainer, PGExplainer
from torch_geometric.explain.metric import fidelity, unfaithfulness
from torch_geometric.loader.dataloader import DataLoader

from GCN import GraphGCN

# Command arguments
parser = argparse.ArgumentParser(description = 'PGExplainer')
parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
parser.add_argument("--data_path", type = str, default = '', help = "Root directory where the dataset should be saved")
parser.add_argument("--device", type = str, default = 'cuda:0', help = "")
parser.add_argument("--model_path", type = str, default = '', help = "Model to be explained")
parser.add_argument("--train_ratio", type = float, default = 0.9, help = "")
parser.add_argument("--epochs", type = int, default = 30, help = "Explainer training epochs")
parser.add_argument("--eval_step", type = int, default = 5, help = "Explainer validation steps")
parser.add_argument("--lr", type = float, default = 5e-3, help = "Explainer learning rate")
parser.add_argument("--batch_size", type = int, default = 64, help = "Explainer training batchsize")

parser.add_argument("--explanation_type", type = str, default = 'phenomenon')
# 'model': Explain the model prediction.
# 'phenomenon': Explain the phenomenon that the model is trying to predict.

parser.add_argument("--edge_mask_type", type = str, default = 'object')
# None: Will not apply any mask on nodes.
# 'object': Will mask each edge.
# "common_attributes": Will mask each edge feature.
# "attributes": Will mask each feature across all edges.

parser.add_argument("--model_mode", type = str, default = 'binary_classification')
# "binary_classification": A binary classification model.
# "multiclass_classification": A multiclass classification model.
# "regression": A regression model.

parser.add_argument("--model_task_level", type = str, default = 'graph')
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

# Dataset
dataset = BA2MotifDataset(args.data_path)
index = list(range(len(dataset)))
random.shuffle(index)
train_num = round(args.train_ratio * len(dataset))
trainset, testset = dataset[index[:train_num]], dataset[index[train_num:]]
trainloader = DataLoader(trainset, batch_size = args.batch_size, shuffle = True)
testloader = DataLoader(testset, batch_size = 1, shuffle = False)

# GNN model to be explained
ckpt = torch.load(args.model_path)
model_args, model_state_dict = ckpt['args'], ckpt['state_dict']
explained_model = GraphGCN(dataset.num_features, model_args.hidden_dim, dataset.num_classes)
explained_model.load_state_dict(model_state_dict)
explained_model.to(device)
explained_model.eval()

# Explainer
model_config = {
    'mode': args.model_mode,
    'task_level': args.model_task_level,
    'return_type': args.model_return_type
}
explainer = Explainer(
    model = explained_model,
    algorithm = PGExplainer(epochs = args.epochs, lr = args.lr).to(device),
    explanation_type = args.explanation_type,
    edge_mask_type = args.edge_mask_type,
    model_config = model_config
)

# Explainer training
for epoch in range(args.epochs):
    total_loss = 0
    for data in tqdm(trainloader):
        data = data.to(device)
        target = explained_model(data.x, data.edge_index, data.batch)
        explain_loss = explainer.algorithm.train(
            epoch = epoch,
            model = explained_model,
            x = data.x,
            edge_index = data.edge_index,
            batch = data.batch,
            target = target
        )
        total_loss += explain_loss
    total_loss /= train_num
    print('epoch', epoch + 1,'loss:', total_loss)

# Explainer evaluating
unfaithful_list = []
fid_pos_list = []
fid_neg_list = []
for data in tqdm(testloader):
    data = data.to(device)
    target = explained_model(data.x, data.edge_index, data.batch)
    explanation = explainer(x = data.x, edge_index = data.edge_index, batch = data.batch, target = target)
    fid = fidelity(explainer, explanation)
    unfaithful = unfaithfulness(explainer, explanation)
    fid_pos_list.append(fid[0])
    fid_neg_list.append(fid[1])
    unfaithful_list.append(unfaithful)
    

fid_pos = np.mean(fid_pos_list)
fid_neg = np.mean(fid_neg_list)
unfaithful = np.mean(unfaithful_list)
print('fid_pos:', fid_pos)
print('fid_neg:', fid_neg)
print('unfaithfulness:', unfaithful)