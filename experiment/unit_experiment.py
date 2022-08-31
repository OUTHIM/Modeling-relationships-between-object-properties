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
import time

from torch.utils.data import DataLoader
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from model.softmax_heteGraphSAGE import softmax_HeteroGNN
from utils import dataset_utils, data_preprocessing

# %% some util functions
def get_attribute_to_predict(attributes, attribute_values):
    attribute_to_predict = []
    for head in attributes:
        if head not in attribute_values:
            attribute_to_predict.append(head)
    return attribute_to_predict

def adjective_to_bin(dataset_name, attribute, adjective):
    path_quantization = os.path.join(os.getcwd(), 'dataset/{0}/quantization_levels.json'.format(dataset_name))
    with open(path_quantization) as f:
        quantization_map = json.load(f)
    
    return quantization_map[attribute][adjective]

def value_to_bin(dataset_name, attribute, value):
    # input: value can be float or int
    # output: a string that is the quantized float(bin) of the value
    path_quantization = os.path.join(os.getcwd(), 'dataset/{0}/quantization_levels.json'.format(dataset_name))
    with open(path_quantization) as f:
        quantization_map = json.load(f)
    
    levels = []
    # get all the quantization levels
    for level in quantization_map[attribute]:
        level = float(level)
        levels.append(level)
    levels = np.array(levels)
    closest_idx = np.abs(value - levels).argmin()
    closest_value = levels[closest_idx]
    return quantization_map[attribute][str(closest_value)]

def bin_to_value(dataset_name, attribute, value):
    # input: value should be a string here
    # return: the string format of the value
    path_quantization = os.path.join(os.getcwd(), 'dataset/{0}/quantization_levels.json'.format(dataset_name))
    with open(path_quantization) as f:
        quantization_map = json.load(f)
    bin_to_value = quantization_map
    # reverse the quantization map dictionary
    for attr_name in bin_to_value: 
        bin_to_value[attr_name] = {bin:value for value, bin in bin_to_value[attr_name].items()}
    return bin_to_value[attribute][value]

# %%%%%%%%%%%%%%%% Things that you can customize %%%%%%%%%%%%%%%%%%%%%%%%%%
# ------------ Basic settings for different dataset ---------------------------
experiment_args = {
    'dataset_name': 'amazon'
}

dataset_name = experiment_args['dataset_name']

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
    'name': adjective_to_bin(dataset_name, 'name','bakingtray'),
    'Material': adjective_to_bin(dataset_name, 'Material','metal'),
    'Colour': adjective_to_bin(dataset_name, 'Colour','silver'),
    # 'Weight': value_to_bin(dataset_name, 'Volume', 8000), 
    'Volume': value_to_bin(dataset_name, 'Volume', 1000), 
    'Length': value_to_bin(dataset_name, 'Length', 20), 
    'Width': value_to_bin(dataset_name, 'Width', 50), 
    'Height': value_to_bin(dataset_name, 'Height', 12),
    'Functionality': adjective_to_bin(dataset_name, 'Functionality','tool'), 
    'Button':1, 
    'Lip':1, 
    'Fillability':1, 
    'Washability':2, 
    'Dismountability':2, 
    'Shape':adjective_to_bin(dataset_name, 'Shape','other'), 
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

dataset_name_to_known_attributes = {
    'amazon': amazon_attribute_values,
    'shop_vrb': shop_vrb_attribute_values
}

dataset_name_to_total_attributes = {
    'amazon': amazon_attributes,
    'shop_vrb': shop_vrb_attributes
}

dataset_name_to_split_types = {
    'amazon': amazon_split_types,
    'shop_vrb': shop_vrb_split_types
}

# ---------------- settings of variables for the experiment ----------------------
attribute_values = dataset_name_to_known_attributes[dataset_name]

path_args = os.path.join(os.getcwd(), 'experiment/{0}_args.json'.format(dataset_name))
with open(path_args) as f:
    args = json.load(f)

# load graph
cwd = os.getcwd()
# for shop-vrb
# path = os.path.join(cwd, 'dataset\shop_vrb\graph_data.gpickle')
# for Amazon

t_total_0 = time.time()
t_load_graph_0 = time.time()
path = os.path.join(cwd, 'experiment/{}_graph_data.gpickle'.format(dataset_name))
G = nx.read_gpickle(path)
t_load_graph_1 = time.time()

# load attribute node indices
path2 = os.path.join(cwd, 'experiment/node_idx_of_attributes.json')
with open(path2) as f:
    attr_to_node = json.load(f)
# print('attribute values and their corresponding node indices', attr_to_node)

################################################# Create the test node on 'shape' and 'size' ######################################################
# ----------- variable setting part ---------------------
# generate a center node with object 'name' index

t_append_node_0 = time.time()

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

print('neighbour of the node to predict', list(G.neighbors(node_idx)))            
print('total num of nodes', len(G.nodes))
# load the full knowledge graph
hetero = HeteroGraph(G)
test_graph = hetero
# print(test_graph.get_num_labels('name'))

# %% Add the attribute that we want the model to predict
attribute_to_predict = get_attribute_to_predict(dataset_name_to_total_attributes[dataset_name], attribute_values)
edge_label_index = {}
name_node_idx = hetero.num_nodes('name') - 1
print('\033[94m' + 'number of name nodes:' + '\033[0m', hetero.num_nodes('name'))
print('\033[94m' + 'Attribute to predict:' + '\033[0m', attribute_to_predict)
for attribute_name in attribute_to_predict:
    tup = ('name', 'name-{0}'.format(attribute_name), attribute_name)
    tensor = torch.Tensor([[name_node_idx for _ in range(len(test_graph.node_feature[attribute_name]))],[x for x in range(len(test_graph.node_feature[attribute_name]))]])
    edge_label_index[tup] = tensor
    print('\033[94m' + 'Message type to predict:' + '\033[0m', tup)

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
print('\033[94m' + 'Edge labels are:' + '\033[0m', edge_label)
t_append_node_1 = time.time()

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
drop_softmax_ratio = args['drop_softmax_ratio']
model = softmax_HeteroGNN(HeteroSAGEConv, test_graph, hidden_size, num_layer_hop, encoder_layer_num, drop_softmax_ratio=None).to('cpu')
PATH = 'experiment/{0}_best.pth'.format(dataset_name)

t_load_model_0 = time.time()
model.load_state_dict(torch.load(PATH))
t_load_model_1 = time.time()

model.eval()
test_graph.to('cpu')
t_inference_0 = time.time()
pred, _, corresponding_attr_values = model(test_graph)
t_inference_1 = time.time()
torch.set_printoptions(precision=2)

t_post_processing_0 = time.time()
for i in range(len(pred)):
    attr_name = list(corresponding_attr_values.items())[i][0][2]
    real_bins = list(corresponding_attr_values.items())[i][1].cpu().detach().numpy()
    result = F.softmax(list(pred.items())[i][1]).cpu().detach().numpy()
    most_likely_indices = np.argsort(result[0])[-5:]
    print('\033[94m' + '-----------------------------------' + '\033[0m')
    print('\033[94m' + 'The most likely {0} are:'.format(attr_name) + '\033[0m')
    for index in np.flip(most_likely_indices):
        value = bin_to_value(dataset_name, attr_name, real_bins[index])
        print('\033[94m' + 'Value: {0}, Confidence {1:.2f} %'.format(value, result[0,index]*100) + '\033[0m')

t_post_processing_1 = time.time()
t_total_1 = time.time()
print('Total time: ', t_total_1 - t_total_0)
print('Time of loading graph: ', t_load_graph_1 - t_load_graph_0)
print('Time of appending test nodes: ', t_append_node_1 - t_append_node_0)
print('Time of inference: ', t_inference_1 - t_inference_0)
print('Time of post-processing: ', t_post_processing_1 - t_post_processing_0)