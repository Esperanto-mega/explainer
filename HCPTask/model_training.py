# @yinjun

import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import Tensor

from torch.optim import Adam
from torch_geometric.datasets import NeuroGraphDataset
from torch_geometric.loader.dataloader import DataLoader

from GNN import ResidualGNN
from GNN import fix_seed

# Command arguments
parser = argparse.ArgumentParser(description = 'PGExplainer')
parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
parser.add_argument("--data_path", type = str, default = '/datain/v-yinju/GNN-Explainer/Data', help = "Root directory where the dataset should be saved")
parser.add_argument("--data_name", type = str, default = 'HCPTask', help = "Dataset name")
parser.add_argument("--batch_size", type = int, default = 16, help = "")
parser.add_argument("--device", type = str, default = 'cuda:0', help = "")
parser.add_argument("--train_ratio", type = float, default = 0.7, help = "")
parser.add_argument("--valid_ratio", type = float, default = 0.1, help = "")
parser.add_argument("--model_path", type = str, default = '', help = "Root directory where the trained model should be saved")
parser.add_argument("--hidden_gnn", type = int, default = 32, help = "")
parser.add_argument("--hidden_mlp", type = int, default = 64, help = "")
parser.add_argument("--num_layers", type = int, default = 3, help = "")
parser.add_argument("--lr", type = float, default = 1e-5, help = "")
parser.add_argument('--weight_decay', type = float, default = 0.0005)
parser.add_argument("--epochs", type = int, default = 100, help = "")
args = parser.parse_args()
print(args)

device = torch.device(args.device)
fix_seed(args.seed)

# Dataset
dataset = NeuroGraphDataset(root = args.data_path, name = args.data_name)
# NeuroGraphDataset(7443)
# Data(x=[400, 400], edge_index=[2, 7368], y=[1])
# data.num_features = 400
# data.num_classes = 7

index = list(range(len(dataset)))
random.shuffle(index)
train_ids = round(args.train_ratio * len(dataset))
valid_ids = round((args.train_ratio + args.valid_ratio) * len(dataset))
trainset, validset, testset = dataset[index[:train_ids]], dataset[index[train_ids:valid_ids]], dataset[index[valid_ids:]]
train_loader = DataLoader(trainset, batch_size = args.batch_size, shuffle = True)
val_loader = DataLoader(validset, batch_size = args.batch_size, shuffle = False)
test_loader = DataLoader(testset, batch_size = 1, shuffle = False)

# Model
model = ResidualGNN(trainset.num_features, trainset.num_classes, args.hidden_gnn, args.hidden_mlp, args.num_layers)
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters is: {total_params}")
print(model)

optimizer = Adam(model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
loss_fn = torch.nn.CrossEntropyLoss()

# Train function
def train(train_loader):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(args.device)
        out = model(data)
        loss = loss_fn(out, data.y)
        total_loss += loss
        loss.backward()
        optimizer.step() 
    return total_loss / len(train_loader.dataset)

# Evaluate function
@torch.no_grad()
def test(loader):
    model.eval()
    correct = 0
    for data in loader:  
        data = data.to(args.device)
        out = model(data)  
        pred = out.argmax(dim = 1)  
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)

best_val_acc = 0.0
for epoch in range(args.epochs):
    loss = train(train_loader)
    val_acc = test(val_loader)
    test_acc = test(test_loader)
    print("epoch: {}, loss: {}, val_acc:{}, test_acc:{}".format(epoch, np.round(loss.item(), 8), np.round(val_acc, 8), np.round(test_acc, 8)))
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        state = {
            'args': args,
            'epoch': epoch + 1,
            'best_accuracy': val_acc,
            'state_dict': model.state_dict()
        }
        torch.save(state, args.model_path)

# Evaluation
ckpt = torch.load(args.model_path)
model_args, model_state_dict = ckpt['args'], ckpt['state_dict']
eval_model = ResidualGNN(
    trainset.num_features, 
    trainset.num_classes, 
    model_args.hidden_gnn, 
    model_args.hidden_mlp, 
    model_args.num_layers
)
eval_model = eval_model.to(device)
eval_model.eval()
test_acc = test(test_loader)
print("test_acc:{}".format(np.round(test_acc, 8)))
