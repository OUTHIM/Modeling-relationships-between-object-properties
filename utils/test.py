import json
import pandas as pd
import os
import networkx as nx
from pathlib import Path
import sys
import numpy as np
import torch
import networkx as nx
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch
from torch.utils.data import DataLoader
import numpy as np
import copy
import sys
from pathlib import Path
import os
import networkx as nx
from deepsnap.hetero_gnn import HeteroSAGEConv
from deepsnap.hetero_graph import HeteroGraph
import torch.nn as nn
import random


a = torch.tensor([1,2])
b = torch.ones((5,))
b[a] = 0
# a = np.arange(12)
# a = a.reshape([3,4])
# b = [1,2,3]
# print(a)
# print(a[:,b])
# dataset_name = 'amazon'
# cwd = os.getcwd()
# path = os.path.join(cwd, 'experiment/{}_graph_data.gpickle'.format(dataset_name))
# G = nx.read_gpickle(path)
# # dataset_utils.visualize_graph(G)
# hetero = HeteroGraph(G)
# directed = True
# # create dataset object in link-prediction mode
# dataset = GraphDataset([hetero], task='link_pred', edge_negative_sampling_ratio = 1, resample_negatives=True)
# amazon_split_types = [('name', 'name-Weight', 'Weight'), ('name', 'name-Volume', 'Volume')]
# # split dataset for link prediction and adapt to torch.Dataloader
# # only split on edges sourcing from 'name'
# if directed == True:
#     dataset_train, dataset_val, dataset_test = dataset.split(transductive=True,
#                                                             split_ratio=[0.8, 0.1, 0.1], 
#                                                             split_types=amazon_split_types, shuffle=False)
# else: 
#     dataset_train, dataset_val, dataset_test = dataset.split(transductive=True,
#                                                             split_ratio=[0.8, 0.1, 0.1])

# train_loader = DataLoader(dataset_train, collate_fn=Batch.collate(),
#                     batch_size=1)
# val_loader = DataLoader(dataset_val, collate_fn=Batch.collate(),
#                     batch_size=1)
# test_loader = DataLoader(dataset_test, collate_fn=Batch.collate(),
#                     batch_size=1)

# print(dataset_train)
# train_indices = []
# for i in range(3):
#     graph = next(iter(train_loader))
#     indices = graph.edge_label_index[('name', 'name-Volume', 'Volume')]
#     edge_labels = graph.edge_label[('name', 'name-Volume', 'Volume')]
#     train_indices.append(indices)
#     print(indices)
# # print(torch.sum(train_graphs[0] == train_graphs[1]))
# graph_1 = train_indices[0].numpy()
# graph_2 = train_indices[2].numpy()
# # print(np.isin(graph_1, graph_2))
# # print(graph_2[1 - np.isin(graph_1, graph_2)])
# print(np.sum(np.invert(np.isin(graph_2[0], graph_1[0]))))
# print(graph_1[0][np.invert(np.isin(graph_1[0], graph_2[0]))])
# print(torch.sum(edge_labels[np.invert(np.isin(graph_1[0], graph_2[0]))]))


# print(torch.unique(indices[1]))
# print(indices[0][-30:])
# print(indices[1][-30:])

# print('edge labels:', edge_labels[-30:])


# dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
# graph = next(iter(train_loader))
# # print(graph)
# # print(hetero)
# # print(hetero.edge_label_index[('name', 'name-Volume', 'Volume')])

# edge_label = {}
# negative_sampling_ratio = 2
# message_type = ('name', 'name-Volume', 'Volume')
# sources = hetero.edge_label_index[message_type][0]
# targets = hetero.edge_label_index[message_type][1]
# negative_labels = []
# negative_sources = []
# negative_targets = []
# edge_label[message_type] = torch.ones(size = sources.shape) # positive labels

# for idx, source_node in enumerate(sources):
#     avoidance = targets[idx]
#     target_attr_values = torch.unique(targets).tolist()
#     if negative_sampling_ratio <= len(target_attr_values):
#         negative_target_nodes = torch.tensor(random.sample(target_attr_values, negative_sampling_ratio))
#     else:
#         negative_target_nodes = torch.tensor(random.sample(target_attr_values, 1))
#     negative_labels.append(torch.zeros_like(negative_target_nodes))
#     negative_sources.append(torch.tile(source_node, (len(negative_target_nodes),)))
#     negative_targets.append(negative_target_nodes)

# negative_labels = torch.concat(negative_labels, -1)
# negative_sources = torch.concat(negative_sources, -1)
# negative_targets = torch.concat(negative_targets, -1)

# negative_edge_label_index = torch.stack([negative_sources, negative_targets], dim = 0)
# print(negative_edge_label_index)