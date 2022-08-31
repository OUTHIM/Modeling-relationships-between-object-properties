from matplotlib.font_manager import json_load
import torch
from deepsnap.hetero_gnn import HeteroSAGEConv
from deepsnap.hetero_graph import HeteroGraph
from pathlib import Path
import sys
import os
import networkx as nx
import json
import numpy as np

from torch.utils.data import DataLoader
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from model.heteGraphSAGE import HeteroGNN
from utils import dataset_utils, data_preprocessing


def get_attribute_to_predict(attributes, attribute_values):
    attribute_to_predict = []
    for head in attributes:
        if head not in attribute_values:
            attribute_to_predict.append(head)
    return attribute_to_predict

def test_one_sample(attribute_values, folder_path, dataset_name = 'amazon'):
    # known attributes for the object
    shop_vrb_attribute_values = {
        'name': 18,
        'color':6, 
        # 'weight':2, 
        'movability':1,
        # 'shape': 1,
        'size': 2, 
        'material':2, 
        'picking': 1, 
        'powering': 1, 
        'disassembly': 3}

    amazon_attribute_values = {
        'name': 1,
        'Material': 3,
        'Colour': 23,
        # 'Weight': value_to_bin(dataset_name, 'Volume', 8000), 
        'Volume': 32, 
        'Length': 21, 
        'Width': 15, 
        'Height': 31,
        'Functionality': 3, 
        'Button':1, 
        'Lip':1, 
        'Fillability':1, 
        'Washability':2, 
        'Dismountability':2, 
        'Shape':4, 
        'Handle':2}

    # %% Things that you don't have to change
    shop_vrb_split_types = [('name', 'name-color', 'color'), ('name', 'name-weight', 'weight'), ('name', 'name-movability', 'movability'), 
                            ('name', 'name-material', 'material'), ('name', 'name-shape', 'shape'), 
                            ('name', 'name-size', 'size'), ('name', 'name-picking', 'picking'), 
                            ('name', 'name-powering', 'powering'), ('name', 'name-disassembly', 'disassembly')]

    amazon_split_types = [('name', 'name-Material', 'Material'), ('name', 'name-Colour', 'Colour'), ('name', 'name-Weight', 'Weight'), 
    ('name', 'name-Volume', 'Volume'), ('name', 'name-Length', 'Length'), ('name', 'name-Width', 'Width'), 
    ('name', 'name-Height', 'Height'), ('name', 'name-Functionality', 'Functionality'), 
    ('name', 'name-Button', 'Button'), ('name', 'name-Lip', 'Lip'), ('name', 'name-Fillability', 'Fillability'), ('name', 'name-Washability', 'Washability'), 
    ('name', 'name-Dismountability', 'Dismountability'), ('name', 'name-Shape', 'Shape'), ('name', 'name-Handle', 'Handle')]

    shop_vrb_attributes = set(['color', 'weight', 'movability', 'material', 'shape', 'size', 'name', 'picking', 'powering', 'disassembly'])
    amazon_attributes = set(['Material', 'Colour', 'Weight', 'Volume', 'Length', 'Width', 'Height', 'Functionality', 
    'Button', 'Lip', 'Fillability', 'Washability', 'Dismountability', 'Shape', 'Handle', 'name'])

    # dataset_name_to_known_attributes = {
    #     'amazon': amazon_attribute_values,
    #     'shop_vrb': shop_vrb_attribute_values
    # }

    dataset_name_to_total_attributes = {
        'amazon': amazon_attributes,
        'shop_vrb': shop_vrb_attributes
    }

    dataset_name_to_split_types = {
        'amazon': amazon_split_types,
        'shop_vrb': shop_vrb_split_types
    }

    # ---------------- settings of variables for the experiment ----------------------
    path_args = os.path.join(folder_path, '{0}_args.json'.format(dataset_name))
    with open(path_args) as f:
        args = json.load(f)

    # load graph
    # for shop-vrb
    # path = os.path.join(cwd, 'dataset\shop_vrb\graph_data.gpickle')
    # for Amazon
    path = os.path.join(folder_path, '{0}_graph_data.gpickle'.format(dataset_name))
    G = nx.read_gpickle(path)

    # load attribute node indices
    path2 = os.path.join(folder_path, 'node_idx_of_attributes.json'.format(dataset_name))
    with open(path2) as f:
        attr_to_node = json.load(f)
    # print('attribute values and their corresponding node indices', attr_to_node)

    ################################################# Create the test node on 'shape' and 'size' ######################################################
    # ----------- variable setting part ---------------------
    # generate a center node with object 'name' index
    name_index = attribute_values['name']
    node_idx = len(G.nodes)
    G.add_nodes_from([
                    (node_idx,{"node_type":'name',"node_label": -100, "node_feature": torch.Tensor([name_index]).long()})
                    ])

    # for Amazon, noted that the digitized value are float numbers
    # create edges
    for node_type in attribute_values:
        if node_type != 'name':
            G.add_edges_from([
                (node_idx, attr_to_node[node_type][str(attribute_values[node_type])],{'edge_type':'name-'+ node_type}),
                (attr_to_node[node_type][str(attribute_values[node_type])], node_idx, {'edge_type': node_type + '-name'})
                ])

    # print('neighbour of the node to predict', list(G.neighbors(node_idx)))            
    # print('total num of nodes', len(G.nodes))
    # load the full knowledge graph
    hetero = HeteroGraph(G)
    test_graph = hetero
    # print(test_graph.get_num_labels('name'))

    # %% Add the attribute that we want the model to predict
    attribute_to_predict = get_attribute_to_predict(dataset_name_to_total_attributes[dataset_name], attribute_values)
    edge_label_index = {}
    name_node_idx = hetero.num_nodes('name') - 1
    # print('\033[94m' + 'number of name nodes:' + '\033[0m', hetero.num_nodes('name'))
    # print('\033[94m' + 'Attribute to predict:' + '\033[0m', attribute_to_predict)
    for attribute_name in attribute_to_predict:
        tup = ('name', 'name-{0}'.format(attribute_name), attribute_name)
        tensor = torch.Tensor([[name_node_idx for _ in range(len(test_graph.node_feature[attribute_name]))],[x for x in range(len(test_graph.node_feature[attribute_name]))]])
        edge_label_index[tup] = tensor
        # print('\033[94m' + 'Message type to predict:' + '\033[0m', tup)

    test_graph.edge_label_index = edge_label_index
    # for shop-vrb
    # test_graph.edge_label_index = {('name', 'name-shape', 'shape'):torch.Tensor([[20000 for _ in range(len(test_graph.node_feature['shape']))],[x for x in range(len(test_graph.node_feature['shape']))]]),
    # ('name', 'name-size', 'size'):torch.Tensor([[20000 for _ in range(len(test_graph.node_feature['size']))],[x for x in range(len(test_graph.node_feature['size']))]])}
    # for amazon
    # test_graph.edge_label_index = {('name', 'name-Weight', 'Weight'):torch.Tensor([[1000 for _ in range(len(test_graph.node_feature['Weight']))],[x for x in range(len(test_graph.node_feature['Weight']))]])}

    # set edge label (not important for testing experiment)
    edge_label = {}
    for edge_type in test_graph.edge_label_index:
        edge_label[edge_type] = torch.Tensor([1])

    test_graph.edge_label = edge_label
    # print('\033[94m' + 'Edge labels are:' + '\033[0m', edge_label)

    # %% Some print things to prove that the added node is exactly the last row vector in node_feature['name]
    # print(test_graph)
    # print(test_graph.node_label['name'])
    # print(len(test_graph.node_feature['name']))
    # l = hetero._convert_to_graph_index(test_graph.node_label_index['name'], 'name').tolist()
    # # print(test_graph.node_label_index['name'])
    # print(l)

    # print('\033[94m' + 'Edge label indices are:' + '\033[0m', edge_label_index)


    # %%%%%%%%%%%%%%%%%%%%%%%%%% Start to predict %%%%%%%%%%%%%%%%%%%%%%%%%%
    # print(test_graph.edge_index)
    # print(test_graph.node_feature['Weight']) # features dim of node in certain type
    # print(test_graph.get_num_labels('shape'))

    hidden_size = args['hidden_size']
    num_layer_hop = args['layer_num']
    encoder_layer_num = args['encoder_layer_num']
    model = HeteroGNN(HeteroSAGEConv, test_graph, hidden_size, num_layer_hop, encoder_layer_num).to('cpu')
    PATH = os.path.join(folder_path, '{0}_best.pth'.format(dataset_name))
    model.load_state_dict(torch.load(PATH))

    model.eval()
    pred , corresponding_attr_values = model(test_graph.to('cpu'))
    torch.set_printoptions(precision=2)

    result = []
    real_bins = []
    attr_names = []
    for i in range(len(pred)):
        attr_names.append(list(corresponding_attr_values.items())[i][0][2])
        real_bins.append(list(corresponding_attr_values.items())[i][1].detach().numpy())
        result.append(torch.sigmoid(list(pred.items())[i][1]).detach().numpy())
        # most_likely_indices = np.argsort(result)[-5:]
    
    return result, real_bins, attr_names
        # print('\033[94m' + '-----------------------------------' + '\033[0m')
        # print('\033[94m' + 'The most likely {0} are:'.format(attr_name) + '\033[0m')
        # for index in np.flip(most_likely_indices):
        #     value = bin_to_value(dataset_name, attr_name, real_bins[index])
        #     print('\033[94m' + 'Value: {0}, Confidence {1:.2f} %'.format(value, result[index]*100) + '\033[0m')

