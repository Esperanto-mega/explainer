import random
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch_geometric.datasets import NeuroGraphDataset
from torch_geometric.explain import Explainer, DummyExplainer
from torch_geometric.explain.metric import fidelity, unfaithfulness
from torch_geometric.loader.dataloader import DataLoader

from GNN import ResidualGNN

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def topk_threshold(x, ratio = 0.3):
    topk = round(ratio * x.shape[0])
    threshold = torch.topk(x, topk).values[-1]
    return (x > threshold).float()

def label_distribution(dataset):
    label_dis = {}
    for data in dataset:
        if data.y.item() in label_dis:
            label_dis[data.y.item()] += 1
        else:
            label_dis[data.y.item()] = 1
    for label in label_dis:
        label_dis[label] /= len(dataset)
    return label_dis

# Command arguments
parser = argparse.ArgumentParser(description = 'Random-Explainer')
parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
parser.add_argument("--data_path", type = str, default = '/datain/v-yinju/GNN-Explainer/Data', help = "Root directory where the dataset should be saved")
parser.add_argument("--data_name", type = str, default = 'HCPTask', help = "Dataset name")
parser.add_argument("--device", type = str, default = 'cuda:0', help = "")
parser.add_argument("--model_path", type = str, default = '', help = "Model to be explained")
parser.add_argument("--exp_ratio", type = float, default = 0.2, help = "Ratio of explained data samples")
parser.add_argument("--repeat", type = int, default = 10, help = "Times to repeat")

parser.add_argument("--explanation_type", type = str, default = 'model')
# 'model': Explain the model prediction.
# 'phenomenon': Explain the phenomenon that the model is trying to predict.

parser.add_argument("--edge_mask_type", type = str, default = 'object')
# None: Will not apply any mask on nodes.
# 'object': Will mask each edge.
# "common_attributes": Will mask each edge feature.
# "attributes": Will mask each feature across all edges.

parser.add_argument("--model_mode", type = str, default = 'multiclass_classification')
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
set_seed(args.seed)

# Dataset
dataset = NeuroGraphDataset(root = args.data_path, name = args.data_name)
index = list(range(len(dataset)))
random.shuffle(index)
exp_ids = round(args.exp_ratio * len(dataset))
testset = dataset[index[:exp_ids]]
test_loader = DataLoader(testset, batch_size = 1, shuffle = False)

label_dis = label_distribution(testset)
print('label_dis:', label_dis)

# GNN model to be explained
ckpt = torch.load(args.model_path)
model_args, model_state_dict = ckpt['args'], ckpt['state_dict']
explained_model = ResidualGNN(
    testset.num_features, 
    testset.num_classes, 
    model_args.hidden_gnn, 
    model_args.hidden_mlp, 
    model_args.num_layers
)
explained_model.load_state_dict(model_state_dict)
explained_model = explained_model.to(device)
explained_model.eval()

eval_correct = 0
for data in tqdm(test_loader):
    data = data.to(device)
    output = explained_model(data.x, data.edge_index, data.batch, data.num_graphs)
    label = data.y
    prediction = torch.argmax(output, dim = 1)
    eval_correct += (prediction == label).sum()
eval_accuracy = eval_correct / len(test_loader.dataset)
print('Evaluation Accuracy:', eval_accuracy.item())

# Explainer
model_config = {
    'mode': args.model_mode,
    'task_level': args.model_task_level,
    'return_type': args.model_return_type
}
explainer = Explainer(
    model = explained_model,
    algorithm = DummyExplainer(),
    explanation_type = args.explanation_type,
    edge_mask_type = args.edge_mask_type,
    model_config = model_config
)

all_result = {
  'fid_pos': [],
  'fid_neg': [],
  'unfaith': []
}
for i in range(args.repeat):
    unfaithful_list = []
    fid_pos_list = []
    fid_neg_list = []
    for data in tqdm(test_loader):
        data = data.to(device)
        explanation = explainer(x = data.x, edge_index = data.edge_index, batch = data.batch, num_graphs = data.num_graphs)
        explanation.edge_mask = topk_threshold(explanation.edge_mask)
        fid = fidelity(explainer, explanation)
        unfaithful = unfaithfulness(explainer, explanation)
        fid_pos_list.append(fid[0])
        fid_neg_list.append(fid[1])
        unfaithful_list.append(unfaithful)
        
    fid_pos = np.mean(fid_pos_list)
    fid_neg = np.mean(fid_neg_list)
    unfaith = np.mean(unfaithful_list)
    all_result['fid_pos'].append(fid_pos)
    all_result['fid_neg'].append(fid_neg)
    all_result['unfaith'].append(unfaith)
    print('fid_pos:',fid_pos)
    print('fid_neg:',fid_neg)
    print('unfaith:',unfaith)
    
print('result:', all_result)
print('fid_pos:', np.mean(all_result['fid_pos']), '±', np.std(all_result['fid_pos']))
print('fid_neg:', np.mean(all_result['fid_neg']), '±', np.std(all_result['fid_neg']))
print('unfaithfulness:', np.mean(all_result['unfaith']), '±', np.std(all_result['unfaith']))
