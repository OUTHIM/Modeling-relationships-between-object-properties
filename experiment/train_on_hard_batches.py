import numpy as np
import copy
import sys
from pathlib import Path
import os
import networkx as nx
from deepsnap.hetero_gnn import HeteroSAGEConv
from deepsnap.hetero_graph import HeteroGraph
import torch
import json
import pickle

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
FATHER = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from model.heteGraphSAGE import HeteroGNN
from utils import dataset_utils, data_preprocessing, 

args = {
    'dataset_name': 'amazon',
    "device": "cuda",
    "epochs": 500,
    "lr": 0.01,
    "weight_decay": 1e-4,
    'encoder_layer_num': 3,
    "hidden_size": 32,
    'layer_num': 4,
    'directed' : True,
    'retrain_hard_samples': True
    }


print('------ Start training on hard batches -------')
hard_batch_path = os.path.join(FATHER, 'hard_batches.pickle')
with open(hard_batch_path, 'rb') as handle:
    hard_batches = pickle.load(handle)

iteration_num_for_batch = 50
hetero = HeteroGraph(hard_batches[0]['train'].G[0])
dataset_name = args['dataset_name']
hidden_size = args['hidden_size']
num_layer_hop = args['layer_num']
encoder_layer_num = args['encoder_layer_num']
model = HeteroGNN(HeteroSAGEConv, hetero, hidden_size, num_layer_hop, encoder_layer_num).to(args["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

for i, hard_batch in enumerate(hard_batches):
    for iteration in range(1, iteration_num_for_batch + 1):
        train_batch= hard_batch['train']
        train_batch.to(args["device"])
        model.train()
        optimizer.zero_grad()
        pred, _ = model(train_batch)
        loss = model.loss(pred, train_batch.edge_label)
        loss.backward()
        optimizer.step()

        accs = {}
        # eval_batches = [train_batch, val_batch, test_batch]
        for mode, batch in hard_batch.items():
            model.eval()
            acc = 0
            num = 0
            batch.to(args["device"])
            pred, _ = model(batch)
            for key in pred:
                p = torch.sigmoid(pred[key]).cpu().detach().numpy()
                pred_label = np.zeros_like(p, dtype=np.int64)
                pred_label[p > 0.5] = 1
                pred_label[p <= 0.5] = 0
                acc += np.sum(pred_label == batch.edge_label[key].cpu().numpy())
                num += len(pred_label)
            accs[mode] = acc / num

        log = 'Current batch: {:02d}/{}, Current iteration: {:03d}, Train loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
        print(log.format(i, len(hard_batches), iteration, loss.item(), accs['train'], accs['val'], accs['test']))


path1 = os.path.join(FATHER, 'hard_train.pth')
torch.save(model.state_dict(), path1)
# log = 'Final: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
# accs, _= test(model, dataloaders, args, epoch)
# print(log.format(accs['train'], accs['val'], accs['test']))
# print('average val accuracy over last 50 epoches is: ', val_avg/50)