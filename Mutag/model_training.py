import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader.dataloader import DataLoader

from dataset import Mutagenicity
from GCN import MutagNet

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description = 'PGExplainer')
parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
parser.add_argument("--data_path", type = str, default = '', help = "Root directory where the dataset should be saved")
parser.add_argument("--model_path", type = str, default = '', help = "Root directory where the trained model should be saved")
parser.add_argument("--batch_size", type = int, default = 128, help = "")
parser.add_argument("--device", type = str, default = 'cuda:0', help = "")
parser.add_argument("--num_layers", type = int, default = 2, help = "")
parser.add_argument("--lr", type = float, default = 1e-3, help = "")
parser.add_argument("--epochs", type = int, default = 300, help = "")
parser.add_argument("--eval_step", type = int, default = 10, help = "")
args = parser.parse_args()
print(args)

device = torch.device(args.device)
set_seed(args.seed)

train_dataset = Mutagenicity(args.data_path, mode = 'training')
val_dataset = Mutagenicity(args.data_path, mode = 'evaluation')
test_dataset = Mutagenicity(args.data_path, mode = 'testing')
train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

model = MutagNet(args.num_layers)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.8, patience = 10, min_lr = 1e-4)
loss_fn = nn.CrossEntropyLoss()

best_acc = 0
for epoch in range(args.epoch):
    model.train()
    all_loss = 0
    all_acc = 0
    for data in tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = loss_fn(pred, data.y)
        loss.backward()
        optimizer.step()
        all_loss += loss.item() * data.num_graphs
        all_acc += float(pred.argmax(dim = 1).eq(data.y).sum().item())
    print('Epoch', epoch + 1, 'training loss:', all_loss / len(train_loader.dataset),
          '; training accuracy:', all_acc / len(train_loader.dataset))

    if (epoch + 1) % args.eval_step == 0:
        model.eval()
        with torch.no_grad():
            all_val_loss = 0
            all_val_acc = 0
            for data in tqdm(val_loader):
                data = data.to(device)
                pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
                all_val_loss += loss_fn(pred, data.y) * data.num_graphs
                all_val_acc += float(pred.argmax(dim = 1).eq(data.y).sum().item())
            val_acc = all_val_acc / len(val_loader.dataset)
            print('Epoch', epoch + 1, 'validation loss:', all_val_loss.item() / len(val_loader.dataset), 
                  '; validation accuracy:', val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                state = {
                    'args': args,
                    'epoch': epoch + 1,
                    'best_accuracy': val_acc,
                    'state_dict': model.state_dict()
                }
                torch.save(state, args.model_path)

ckpt = torch.load(args.model_path)
model_args, model_state_dict = ckpt['args'], ckpt['state_dict']
eval_model = MutagNet(model_args.num_layers)
eval_model.load_state_dict(model_state_dict)
eval_model = eval_model.to(device)
eval_model.eval()

all_eval_acc = 0
for data in tqdm(test_loader):
    data = data.to(device)
    pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
    all_eval_acc += float(pred.argmax(dim = 1).eq(data.y).sum().item())
eval_acc = all_eval_acc / len(test_loader.dataset)
print('Evaluation accuracy:', eval_acc)
