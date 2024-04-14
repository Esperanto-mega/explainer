# yinjun@2024/04/13

import random
import argparse
import numpy as np
from tqdm import tqdm

import wandb
import torch
import torch.nn as nn
from torch import Tensor

from torch_geometric.nn import GCNConv
from torch_geometric.datasets import BA2MotifDataset
from torch_geometric.loader.dataloader import DataLoader
from torch_geometric.nn.pool import global_mean_pool, global_max_pool

from GCN import GraphGCN

# Command arguments
parser = argparse.ArgumentParser(description = 'PGExplainer')
# parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
parser.add_argument("--data_path", type = str, default = '', help = "Root directory where the dataset should be saved")
parser.add_argument("--batch_size", type = int, default = 128, help = "")
parser.add_argument("--device", type = str, default = 'cuda:0', help = "")
parser.add_argument("--model_path", type = str, default = '', help = "Root directory where the trained model should be saved")
parser.add_argument("--hidden_dim", type = int, default = 20, help = "")
parser.add_argument("--lr", type = float, default = 1e-3, help = "")
parser.add_argument("--epochs", type = int, default = 1000, help = "")
parser.add_argument("--eval_step", type = int, default = 50, help = "")
args = parser.parse_args()
print(args)

# wandb
# wandb.init(config = args, reinit = True)

# GPU
device = torch.device(args.device)

# Dataset
dataset = BA2MotifDataset(args.data_path)
'''
dataset: BA2MotifDataset(1000)
dataset[0]: Data(x=[25, 10], edge_index=[2, 50], y=[1])
dataset.num_features: 10
dataset.num_classes: 2
'''
# Dataloader
index = list(range(len(dataset)))
random.shuffle(index)
train_num = round(0.8 * len(dataset))
valid_num = round(0.9 * len(dataset))
trainset, validset, testset = dataset[index[:train_num]], dataset[index[train_num:valid_num]], dataset[index[valid_num:]]
trainloader = DataLoader(trainset, batch_size = args.batch_size, shuffle = True)
validloader = DataLoader(validset, batch_size = args.batch_size, shuffle = True)
testloader = DataLoader(testset, batch_size = 1, shuffle = False)

# GCN Model    
model = GraphGCN(dataset.num_features, args.hidden_dim, dataset.num_classes)
model = model.to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
loss_fn = torch.nn.CrossEntropyLoss()
# The input is expected to contain the unnormalized logits for each class 
# (which do not need to be positive or sum to 1, in general).

best_accuracy = 0.0
# Train
for epoch in range(args.epochs):
    model.train()
    correct = 0
    for data in tqdm(trainloader):
        optimizer.zero_grad()
        data = data.to(device)
        # data.shape: (num_nodes, num_features)
        output = model(data)
        # output.shape: (batchsize, num_classes)
        label = data.y
        # label.dim: (batchsize)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        # wandb.log({'Loss': loss})
        
        prediction = torch.argmax(output, dim = 1)
        # prediction.shape: (batchsize)
        correct += (prediction == label).sum()
    accuracy = correct / len(trainset)
    print('Epoch', epoch + 1, 'Accuracy:', accuracy)
    
    # wandb.log({'Accuracy': accuracy})
    
    # Validation
    if (epoch + 1) % args.eval_step == 0:
        model.eval()
        with torch.no_grad():
            val_correct = 0
            for data in tqdm(validloader):
                data = data.to(device)
                output = model(data)
                label = data.y
                val_loss = loss_fn(output, label)

                prediction = torch.argmax(output, dim = 1)
                val_correct += (prediction == label).sum()
            val_accuracy = val_correct / len(validset)
            print('Epoch', epoch + 1, 'Validation Accuracy:', val_accuracy)

            if val_accuracy > best_accuracy:
                torch.save(model.state_dict(), args.model_path)

# Evaluate
eval_model = GraphGCN(dataset.num_features, args.hidden_dim, dataset.num_classes)
eval_model.load_state_dict(torch.load(args.model_path))
eval_model = eval_model.to(device)
eval_model.eval()
eval_correct = 0
for data in tqdm(testloader):
    data = data.to(device)
    output = model(data)
    label = data.y
    prediction = torch.argmax(output, dim = 1)
    eval_correct += (prediction == label).sum()
eval_accuracy = eval_correct / len(testset)
print('Epoch', epoch + 1, 'Evaluation Accuracy:', eval_accuracy)
