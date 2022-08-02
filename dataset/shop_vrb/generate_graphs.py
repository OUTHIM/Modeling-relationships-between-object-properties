import json
import pandas as pd
import os
import networkx as nx
from pathlib import Path
import sys
import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import dataset_utils

# arguments used
args = {'node_number':20000,
        'if_directed':True}

cwd = os.getcwd()
path1 = os.path.join(cwd, 'dataset/shop_vrb/quantized_training_data.csv')
df = pd.read_csv(path1)
df = df.drop(df.columns[0], axis=1) # drop the first column which are indices
data = df.to_dict(orient='records')


node_label = {}
path2 = os.path.join(cwd, 'dataset\shop_vrb\quantization_levels.json')
with open(path2) as f:
    quantization_levels = json.load(f)
# create node label dict
for i, label in enumerate(quantization_levels):
    node_label[label] = i

# create a dict to store node indices of discrete values from different attributes
value_to_node_idx = quantization_levels
for title in value_to_node_idx:
    value_to_node_idx[title] = {value:None for _, value in value_to_node_idx[title].items()}

## Create Graph
node_idx = 0
attr_node_idx = 0
center_node = 0

G = nx.DiGraph()

node_num = args['node_number']

for idx in range(node_num):

    # each row represents one object dict
    row = data[idx] 
    # first, create the center node 'name' 
    node_type = 'name'
    value = float(row[node_type])
    # node feature needs to be a torch tensor
    G.add_nodes_from([
        (node_idx,{'node_type':node_type, 'node_label':value, 'node_feature':torch.Tensor([value]).long()})
    ])
    center_node = node_idx
    node_idx += 1

    # then, create other nodes
    for (node_type, value) in row.items():
        print('generating node {0}/{1}'.format(node_idx, args['node_number']))
        value = float(value)
        if node_type != 'name':
            # if the attribute value node already exists, point current attribute node to the exsiting one
            if value_to_node_idx[node_type][value] != None: 
                attr_node_idx = value_to_node_idx[node_type][value]
            
            # otherwise, create node for certain attribute value
            else:
                G.add_nodes_from([
                (node_idx,{"node_type":node_type, "node_label":value, "node_feature":torch.Tensor([value]).long()})
                ])

                # record the node index of the attribute value node created
                value_to_node_idx[node_type][value] = node_idx
                attr_node_idx = node_idx
                node_idx += 1 

            # add bi-directional edges
            if args['if_directed'] == True:
                G.add_edges_from([
                    (center_node, attr_node_idx,{'edge_type':'name-'+ node_type}),
                    (attr_node_idx, center_node, {'edge_type': node_type + '-name'})
                    ])
            else:
                G.add_edges_from([
                    (center_node, attr_node_idx,{'edge_type':'name-'+ node_type}),
                    (attr_node_idx, center_node, {'edge_type': 'name-' + node_type})
                    ])


# visualize graph 
# dataset_utils.visualize_graph(G, 'shop_vrb')

# save node attribute to json file
with open(os.path.join('dataset/shop_vrb','node_idx_of_attributes.json'), 'w') as outfile:
    json.dump(value_to_node_idx, outfile)
# save graph to pickle
nx.write_gpickle(G, os.path.join(cwd, 'dataset/shop_vrb/' + 'shop_vrb_graph_data.gpickle'))
