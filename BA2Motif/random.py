import random
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import BA2MotifDataset
from torch_geometric.explain import Explainer, DummyExplainer
from torch_geometric.explain.metric import fidelity, unfaithfulness
from torch_geometric.loader.dataloader import DataLoader

from GCN import GraphGCN

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

# Command arguments
parser = argparse.ArgumentParser(description = 'PGExplainer')
parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
parser.add_argument("--data_path", type = str, default = '', help = "Root directory where the dataset should be saved")
parser.add_argument("--device", type = str, default = 'cuda:0', help = "")
parser.add_argument("--model_path", type = str, default = '', help = "Model to be explained")
parser.add_argument("--split_ratio", type = float, default = 0.8, help = "")
parser.add_argument("--repeat", type = int, default = 10, help = "Times to repeat")
args = parser.parse_args()
print(args)

device = torch.device(args.device)
set_seed(args.seed)

# Dataset
dataset = BA2MotifDataset(args.data_path)
index = list(range(len(dataset)))
random.shuffle(index)
train_num = round(args.split_ratio * len(dataset))
testset = dataset[index[train_num:]]
testloader = DataLoader(testset, batch_size = 1, shuffle = False)

# GNN model to be explained
ckpt = torch.load(args.model_path)
model_args, model_state_dict = ckpt['args'], ckpt['state_dict']
explained_model = GraphGCN(dataset.num_features, model_args.hidden_dim, dataset.num_classes)
explained_model.load_state_dict(model_state_dict)
explained_model.to(device)
explained_model.eval()

# Explainer
all_result = {
  'fid_pos': [],
  'fid_neg': [],
  'unfaith:' []
}
for i in range(args.repeat):
    unfaithful_list = []
    fid_pos_list = []
    fid_neg_list = []
    for data in tqdm(testloader):
        data = data.to(device)
        target = explained_model(data.x, data.edge_index, data.batch)
        explanation = DummyExplainer(x = data.x, edge_index = data.edge_index, batch = data.batch, target = target)
        fid = fidelity(DummyExplainer, explanation)
        unfaithful = unfaithfulness(DummyExplainer, explanation)
        fid_pos_list.append(fid[0])
        fid_neg_list.append(fid[1])
        unfaithful_list.append(unfaithful)
        
    fid_pos = np.mean(fid_pos_list)
    fid_neg = np.mean(fid_neg_list)
    unfaith = np.mean(unfaithful_list)
    all_result['fid_pos'].append(fid_pos)
    all_result['fid_neg'].append(fid_neg)
    all_result['unfaith'].append(unfaith)
    
print('result:', all_result)
print('fid_pos:', np.mean(all_result['fid_pos']), '±', np.std(all_result['fid_pos']))
print('fid_neg:', np.mean(all_result['fid_neg']), '±', np.std(all_result['fid_neg']))
print('unfaithfulness:', np.mean(all_result['unfaith']), '±', np.std(all_result['unfaith']))
