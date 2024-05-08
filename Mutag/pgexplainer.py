# @yinjun

import random
import argparse
from tqdm import tqdm
import numpy as np

import torch
from torch_geometric.explain import Explainer, PGExplainer
from torch_geometric.explain.metric import fidelity, unfaithfulness
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

# Command arguments
parser = argparse.ArgumentParser(description = 'PGExplainer')
parser.add_argument("--seed", type = int, default = 42, help = "Random seed")
parser.add_argument("--data_path", type = str, default = '', help = "Root directory where the dataset should be saved")
parser.add_argument("--device", type = str, default = 'cuda:0', help = "")
parser.add_argument("--model_path", type = str, default = '', help = "Model to be explained")
parser.add_argument("--train_ratio", type = float, default = 0.8, help = "")
parser.add_argument("--epochs", type = int, default = 30, help = "Explainer training epochs")
parser.add_argument("--eval_step", type = int, default = 5, help = "Explainer validation steps")
parser.add_argument("--lr", type = float, default = 5e-3, help = "Explainer learning rate")
parser.add_argument("--batch_size", type = int, default = 64, help = "Explainer training batchsize")
parser.add_argument("--repeat", type = int, default = 10, help = "Times to repeat")

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

parser.add_argument("--model_return_type", type = str, default = 'probs')
# "raw": The model returns raw values.
# "probs": The model returns probabilities.
# "log_probs": The model returns log-probabilities.

args = parser.parse_args()
print(args)

device = torch.device(args.device)
set_seed(args.seed)

# Dataset
train_dataset = Mutagenicity(args.data_path, mode = 'training')
val_dataset = Mutagenicity(args.data_path, mode = 'evaluation')
test_dataset = Mutagenicity(args.data_path, mode = 'testing')
train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = False)
test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

positive = [data.y == 1 for data in testset]
print('positive rate:', sum(positive) / len(testset))

# GNN model to be explained
ckpt = torch.load(args.model_path)
model_args, model_state_dict = ckpt['args'], ckpt['state_dict']
explained_model =  MutagNet(model_args.num_layers)
explained_model.load_state_dict(model_state_dict)
explained_model.to(device)
explained_model.eval()

eval_correct = 0
for data in tqdm(testloader):
    data = data.to(device)
    output = explained_model(data.x, data.edge_index, data.batch)
    label = data.y
    prediction = torch.argmax(output, dim = 1)
    eval_correct += (prediction == label).sum()
eval_accuracy = eval_correct / len(testset)
print('Evaluation Accuracy:', eval_accuracy.item())

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
all_result = {
  'fid_pos': [],
  'fid_neg': [],
  'unfaith': []
}
for i in range(args.repeat):
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
    unfaith = np.mean(unfaithful_list)
    all_result['fid_pos'].append(fid_pos)
    all_result['fid_neg'].append(fid_neg)
    all_result['unfaith'].append(unfaith)
    
print('result:', all_result)
print('fid_pos:', np.mean(all_result['fid_pos']), '±', np.std(all_result['fid_pos']))
print('fid_neg:', np.mean(all_result['fid_neg']), '±', np.std(all_result['fid_neg']))
print('unfaithfulness:', np.mean(all_result['unfaith']), '±', np.std(all_result['unfaith']))
