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
import torch.nn.functional as F

from torch.utils.data import DataLoader
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from model.heteGraphSAGE import HeteroGNN
from utils import dataset_utils, data_preprocessing
from model.softmax_heteGraphSAGE import softmax_HeteroGNN

def get_attribute_to_predict(attributes, attribute_values):
    attribute_to_predict = []
    for head in attributes:
        if head not in attribute_values:
            attribute_to_predict.append(head)
    return attribute_to_predict

def test_samples(samples, folder_path, dataset_name = 'amazon', softmax_model=False, model = None):
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

    # amazon_split_types = [('name', 'name-Material', 'Material'), ('name', 'name-Colour', 'Colour'), ('name', 'name-Weight', 'Weight'), 
    # ('name', 'name-Volume', 'Volume'), ('name', 'name-Length', 'Length'), ('name', 'name-Width', 'Width'), 
    # ('name', 'name-Height', 'Height'), ('name', 'name-Functionality', 'Functionality'), 
    # ('name', 'name-Button', 'Button'), ('name', 'name-Lip', 'Lip'), ('name', 'name-Fillability', 'Fillability'), ('name', 'name-Washability', 'Washability'), 
    # ('name', 'name-Dismountability', 'Dismountability'), ('name', 'name-Shape', 'Shape'), ('name', 'name-Handle', 'Handle')]

    amazon_split_types = [('name', 'name-Material', 'Material'), ('name', 'name-Colour', 'Colour'), ('name', 'name-Weight', 'Weight'), 
    ('name', 'name-Volume', 'Volume'), ('name', 'name-Length', 'Length'), ('name', 'name-Width', 'Width'), 
    ('name', 'name-Height', 'Height'), ('name', 'name-Shape', 'Shape'), ('name', 'name-Handle', 'Handle')]

    shop_vrb_attributes = set(['color', 'weight', 'movability', 'material', 'shape', 'size', 'name', 'picking', 'powering', 'disassembly'])
    amazon_attributes = set(['Material', 'Colour', 'Weight', 'Volume', 'Length', 'Width', 'Height', 'Functionality', 
    'Button', 'Lip', 'Fillability', 'Washability', 'Dismountability', 'Shape', 'Handle', 'name'])
    
    # amazon_attributes = set(['Material', 'Colour', 'Weight', 'Volume', 'Length', 'Width', 'Height', 'Shape', 'Handle', 'name'])

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

    hetero_training = HeteroGraph(G)
    # the start node of new 'name' nodes
    start_node_idx = hetero_training.num_nodes('name') - 1 

    ################################################# Create the test node on 'shape' and 'size' ######################################################
    # ----------- variable setting part ---------------------
    # generate center nodes with object 'name' index
    
    for attribute_values in samples:
        name_index = attribute_values['name']
        node_idx = len(G.nodes)
        G.add_nodes_from([
                        (node_idx,{"node_type":'name',"node_label": -100, "node_feature": torch.Tensor([name_index]).long()})
                        ])

        # for Amazon, noted that the digitized value are float numbers
        # create edges
        for node_type in attribute_values:
            # if node_type != 'name':
            G.add_edges_from([
                (node_idx, attr_to_node[node_type][str(attribute_values[node_type])],{'edge_type':'name-'+ node_type}),
                (attr_to_node[node_type][str(attribute_values[node_type])], node_idx, {'edge_type': node_type + '-name'})
                ])
    
    test_graph = HeteroGraph(G) # the heterogeneous graph with all test nodes inserted and linked

    # It is assumed that each sample requires same types of attributes to predict 
    attribute_to_predict = get_attribute_to_predict(dataset_name_to_total_attributes[dataset_name], attribute_values)
    edge_label_index = {}
    # print('\033[94m' + 'number of name nodes:' + '\033[0m', hetero.num_nodes('name'))
    # print('\033[94m' + 'Attribute to predict:' + '\033[0m', attribute_to_predict)

    for attribute_name in attribute_to_predict:
        temp = []
        tup = ('name', 'name-{0}'.format(attribute_name), attribute_name)
        name_node_idx = start_node_idx
        # generate the edge_label_index tensor for all the samples of the same attribute
        if softmax_model:
            # For each sample, we create a pair of edge_label_index to predict. 
            # The ground truth target in this case is given an arbitrary number, which is therefore not correct target value.
            tensor = torch.Tensor([[x for x in range(name_node_idx, name_node_idx + len(samples))], [0 for _ in range(len(samples))]])
            edge_label_index[tup] = tensor
        else:
            for _ in range(len(samples)):
                tensor = torch.Tensor([[name_node_idx for _ in range(len(test_graph.node_feature[attribute_name]))],[x for x in range(len(test_graph.node_feature[attribute_name]))]])
                temp.append(tensor)
                name_node_idx += 1
            edge_label_index[tup] = torch.cat(temp, dim=-1)

        # print('\033[94m' + 'test use:' + '\033[0m', torch.cat(temp, dim=-1)[0,:50])
        # print('\033[94m' + 'Message type to predict:' + '\033[0m', tup)

    test_graph.edge_label_index = edge_label_index

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
    hidden_size = args['hidden_size']
    num_layer_hop = args['layer_num']
    encoder_layer_num = args['encoder_layer_num']

    '''
    # Test on training graphs where all the links are preserved.
    test_graph = hetero_training
    for attribute_name in attribute_to_predict:
        temp = []
        tup = ('name', 'name-{0}'.format(attribute_name), attribute_name)
        edge_label_index[tup] = test_graph.edge_label_index[tup]
        name_node_idx = start_node_idx
    test_graph.edge_label_index = edge_label_index
    edge_label = {}
    for edge_type in test_graph.edge_label_index:
        edge_label[edge_type] = torch.Tensor([1])

    test_graph.edge_label = edge_label
    '''


    # if the model is not given, then load from directory
    if model == None:
        if softmax_model:
            model = softmax_HeteroGNN(HeteroSAGEConv, test_graph, hidden_size, num_layer_hop, encoder_layer_num, drop_softmax_ratio=None).to('cpu')
        else:
            model = HeteroGNN(HeteroSAGEConv, test_graph, hidden_size, num_layer_hop, encoder_layer_num).to('cpu')
        PATH = os.path.join(folder_path, '{0}_best.pth'.format(dataset_name))
        model.load_state_dict(torch.load(PATH))

    model.to('cpu')
    model.eval()

    if softmax_model:
        # The ground-truth edge label is useless in test mode
        pred, _, corresponding_attr_values = model(test_graph.to('cpu'))
    else:
        pred , corresponding_attr_values = model(test_graph.to('cpu'))

    torch.set_printoptions(precision=2)

    results = []
    real_bins = []
    attr_names = []
    if softmax_model:
        for i in range(len(pred)):
            result = F.softmax(list(pred.items())[i][1]).detach().cpu().numpy()
            # print for test
            print(result.shape)

            result = result.reshape([1,-1]).squeeze()
            results.append(result)
            attr_names.append(list(corresponding_attr_values.items())[i][0][2])
            real_bin_piece = list(corresponding_attr_values.items())[i][1].detach().cpu().numpy()
            real_bins.append(real_bin_piece)
    else:
        for i in range(len(pred)):
            result = torch.sigmoid(list(pred.items())[i][1]).detach().cpu().numpy()
            results.append(result)
            attr_names.append(list(corresponding_attr_values.items())[i][0][2])
            real_bin_piece = list(corresponding_attr_values.items())[i][1].detach().cpu().numpy()
            real_bins.append(real_bin_piece)

    quantization_num = len(test_graph.node_feature[attribute_name])
    return results, real_bins, attr_names, quantization_num
        # print('\033[94m' + '-----------------------------------' + '\033[0m')
        # print('\033[94m' + 'The most likely {0} are:'.format(attr_name) + '\033[0m')
        # for index in np.flip(most_likely_indices):
        #     value = bin_to_value(dataset_name, attr_name, real_bins[index])
        #     print('\033[94m' + 'Value: {0}, Confidence {1:.2f} %'.format(value, result[index]*100) + '\033[0m')

