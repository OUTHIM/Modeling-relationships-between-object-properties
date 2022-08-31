import json
import pandas as pd
import os
import networkx as nx
from pathlib import Path
import sys
import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils import dataset_utils

def generate_graphs(quantized_data_filepath, quantization_levels_filepath, save_path, node_num = 1626, if_directed = True, save_file = True):
    df = pd.read_csv(quantized_data_filepath)
    df = df.drop(df.columns[0], axis=1) # drop the first column which are indices
    data = df.to_dict(orient='records')

    node_label = {}
    with open(quantization_levels_filepath) as f:
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

    for idx in range(node_num):

        # each row represents one object dict
        row = data[idx] 
        # first, create the center node 'name' 
        node_type = 'name'
        value = float(row[node_type])
        # node feature needs to be a torch tensor
        G.add_nodes_from([
            (node_idx,{'node_type':node_type, 'node_label':value, 'node_feature':torch.Tensor([value]).type(torch.LongTensor)})
        ])
        center_node = node_idx
        node_idx += 1

        # then, create other nodes
        
        for (node_type, value) in row.items():
            # print('generating node {0}/{1}'.format(node_idx, node_num))
            value = float(value)
            if node_type != 'name':
                # if the attribute value node already exists, point current attribute node to the exsiting one
                if value_to_node_idx[node_type][value] != None: 
                    attr_node_idx = value_to_node_idx[node_type][value]
                
                # otherwise, create node for certain attribute value
                else:
                    G.add_nodes_from([
                    (node_idx,{"node_type":node_type, "node_label":value, "node_feature":torch.Tensor([value]).type(torch.LongTensor)})
                    ])

                    # record the node index of the attribute value node created
                    value_to_node_idx[node_type][value] = node_idx
                    attr_node_idx = node_idx
                    node_idx += 1 

                # add bi-directional edges
                if if_directed == True:
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
    # dataset_utils.visualize_graph(G, dataset_name='amazon')

    # save node attribute to json file
    if save_file:
        with open(os.path.join(save_path,'node_idx_of_attributes.json'), 'w') as outfile:
            json.dump(value_to_node_idx, outfile)
        # save graph to pickle
        nx.write_gpickle(G, os.path.join(save_path, 'amazon_graph_data.gpickle'))

if __name__ == '__main__':
    # arguments used
    args = {'node_number':1626,
            'if_directed':True}
    CURRENT_FILE = Path(__file__).resolve()
    FATHER = CURRENT_FILE.parents[0]  # root directory
    quantized_data_filepath = os.path.join(os.getcwd(), 'dataset/amazon/quantized_clean_data.csv')
    quantization_levels_filepath = os.path.join(os.getcwd(), 'dataset/amazon/quantization_levels.json')
    generate_graphs(quantized_data_filepath, quantization_levels_filepath, save_path = FATHER, node_num=args['node_number'])