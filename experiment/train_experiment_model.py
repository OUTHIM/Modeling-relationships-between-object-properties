from pathlib import Path
import sys
import os
import pandas as pd
import networkx as nx
from deepsnap.hetero_graph import HeteroGraph
import wandb
import json
import warnings
import time

# warnings.filterwarnings('ignore')

FILE = Path(__file__).resolve()
FATHER = FILE.parents[0]  # root directory
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(FATHER) not in sys.path:
    sys.path.append(str(FATHER))

from utils.dataset_utils import visualize_graph
from dataset.amazon import quantize, generate_graphs
from model import train
from model.heteGraphSAGE import HeteroGNN
from model.train import start_training
from sklearn.utils import shuffle

wandb_mode = 'disabled'
# wandb_mode = None
wandb_name = 'test'
args = {
        'custom_train': False,
        'train_with_softmax': True,
        'disjoint': False,

        'dataset_name': 'amazon',

        'sampling_epoch': 50,
        'evaluation_epoch': 5,
        'drop_softmax_ratio': None,
        'quantization_strategy': 'uniform',
        'num_quantization_level': 30,
        'message_passing_edge_ratio': 0.7,
        'node_num': 1431,
        "device": "cuda",
        "epochs": 200,
        "lr": 0.01,
        "weight_decay": 1e-4,
        'encoder_layer_num': 1,
        "hidden_size": 32,
        'layer_num': 1,
        'directed' : True,
        'retrain_hard_samples': False,
        'save_hard_samples': False,
        'negative_sampling_ratio':1
    }

wandb.init(
    mode = wandb_mode,
    project = 'modelling relationship between object properties',
    name = wandb_name,
    config = args
    )

# save the hyper_parameters of training
path2 = os.path.join(FATHER, '{0}_args.json'.format(args['dataset_name']))
with open(path2, 'w') as outfile:
    json.dump(args, outfile)

num_levels = args['num_quantization_level']
node_num = args['node_num']

# Load training data
path = os.path.join(ROOT, 'dataset/amazon/clean_data.csv')
df = pd.read_csv(path)
df = df.drop(df.columns[0:3], axis=1)

# # things changed
# df = df.loc[df['name'] == 'bakingtray']
# df = df.drop(['Functionality','Button','Lip','Fillability','Washability','Dismountability'], axis = 1)
# # ------------------------------

strategy = args['quantization_strategy']
quantize.quantization(df, num_levels, save_path = FATHER, save_file = True, strategy=strategy, figure_on = False)

# Split training and test data: for test data, we choose 15 samples from each type of object
quantized_data = pd.read_csv(os.path.join(FATHER, 'quantized_clean_data.csv'))
# quantized_data = shuffle(quantized_data)

names = pd.unique(quantized_data['name'])
test_data = []
training_data = []
for name in names:
    temp = quantized_data.loc[quantized_data['name'] == name]
    test_data.append(temp.iloc[:15])
    training_data.append(temp.iloc[15:])

test_data =  pd.concat(test_data, ignore_index=True)
training_data = pd.concat(training_data, ignore_index=True)

# save training data and test data
path2 = os.path.join(FATHER, 'training_data.csv')
path3 = os.path.join(FATHER, 'test_data.csv')
test_data.to_csv(path3, index=False)
training_data.to_csv(path2, index=False)

path = os.path.join(FATHER, 'training_data.csv')
train_data = pd.read_csv(path)
train_data = train_data.drop(train_data.columns[0], axis=1)

# Generate test graphs
quantized_data_filepath = os.path.join(FATHER, 'training_data.csv')
quantization_levels_filepath = os.path.join(FATHER, 'quantization_levels.json')
node_num = train_data['Weight'].size
generate_graphs.generate_graphs(quantized_data_filepath, quantization_levels_filepath, save_path = FATHER, node_num = node_num, if_directed = True, save_file = True)

path = os.path.join(FATHER, 'amazon_graph_data.gpickle')
G = nx.read_gpickle(path)
# visualize_graph(G, folder_path = FATHER)
hetero = HeteroGraph(G)

start_time = time.time()
start_training(
    hetero, args, save_path = FATHER, save_file=True, save_hard_samples=args['save_hard_samples'],
    retrain_hard_samples = args['retrain_hard_samples'], custom_train=args['custom_train'],
    negative_sampling = True, negative_sampling_ratio = args['negative_sampling_ratio'], sampling_epoch = args['sampling_epoch'], train_with_softmax = args['train_with_softmax'],
    evaluation_epoch = args['evaluation_epoch'], mp_edge_ratio = args['message_passing_edge_ratio'], wandb = wandb, drop_softmax_ratio=args['drop_softmax_ratio'],
    disjoint = args['disjoint'])
end_time = time.time()
wandb.finish()
print('Total time cost is:', end_time - start_time)