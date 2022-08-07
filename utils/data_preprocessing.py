import networkx as nx
import torch
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch
from torch.utils.data import DataLoader
import random
from torch.utils.data import Dataset
import copy
import random
import math
class CustomDataset(Dataset):
    def __init__(self, hetero, split_types, disjoint = True, mp_edge_ratio = 0.8, negative_sampling = False, negative_sampling_ratio = 1, sampling_epoch = 50):
        self.hetero = hetero
        self.split_types = split_types
        self.mp_edge_ratio = mp_edge_ratio
        self.negative_sampling = negative_sampling
        self.negative_sampling_ratio = negative_sampling_ratio
        self.sampling_epoch = sampling_epoch
        self.epoch_counter = 0
        self.disjoint_hetero = self.create_partly_disjoint_dataset(copy.deepcopy(self.hetero[0]), self.split_types, self.mp_edge_ratio)
        self.sampled_hetero = copy.deepcopy(self.disjoint_hetero)

    def __len__(self):
        return len(self.hetero)
    
    def __getitem__(self, index):
        if self.negative_sampling:
            if self.epoch_counter % self.sampling_epoch == 0:
                self.sampled_hetero = self.create_negative_sampling_dataset(hetero = copy.deepcopy(self.disjoint_hetero), split_types=self.split_types, negative_sampling_ratio=self.negative_sampling_ratio)
            self.epoch_counter += 1

        return self.sampled_hetero

    def create_partly_disjoint_dataset(self, hetero, split_types, mp_edge_ratio):
        # This is not completely disjoint. Some of the message-passing links will still be used in training.
        # Link from both directions are removed for disjoint samples
        edge_label = {}
        for message_type in split_types:
            # Randomly drop some message-passing edges
            opposite_message_type = (message_type[2], message_type[2] + '-' + message_type[0], message_type[0])
            total_mp_edges = hetero.edge_index[message_type].size()[1]
            mp_edge_indices = list(range(total_mp_edges))
            disjoint_mp_edge_num = math.ceil(total_mp_edges * mp_edge_ratio)
            disjoint_mp_edge_indices = random.sample(mp_edge_indices, disjoint_mp_edge_num)
            # Apply the sampling results in both directions
            hetero.edge_index[message_type] = hetero.edge_index[message_type][:, disjoint_mp_edge_indices]
            opposite_edge_index = torch.stack((copy.deepcopy(hetero.edge_index[message_type][1]), copy.deepcopy(hetero.edge_index[message_type][0])), dim = 0)
            hetero.edge_index[opposite_message_type] = opposite_edge_index
            # Add supervision edge label
            sources = hetero.edge_label_index[message_type][0]
            edge_label[message_type] = torch.ones(size = sources.shape) # positive labels

        hetero.edge_label = edge_label
        return hetero

    # def create_disjoint_dataset(self, hetero, split_types, mp_edge_ratio):
    #     for message_type in split_types:
    #         pass
        '''If you need to test the effect on single attribute e.g. Weight, uncomment the following code'''
        # message_type = ('name','name-Weight','Weight')
        # edge_label = {}
        # edge_label_index = {}
        # edge_label_index[message_type] = copy.deepcopy(hetero.edge_index[message_type])
        # hetero.edge_index[message_type] = torch.tensor([[0],[0]])
        # # for split_type in split_types:
        # #     link_type = (split_type[2], split_type[2] + '-' + split_type[0], split_type[0])
        # #     edge_index[link_type] = copy.deepcopy(hetero.edge_index[link_type])

        # hetero.edge_label_index = edge_label_index
        # sources = hetero.edge_label_index[message_type][0]
        # edge_label[message_type] = torch.ones(size = sources.shape) # positive labels
        # hetero.edge_label = edge_label
        
        # return hetero


    def create_negative_sampling_dataset(self, hetero, split_types, negative_sampling_ratio):
        # The custom dataset is specially designed for the object property modelling experiment
        # links are sampled with 'typed negative-sampling strategy'
        # The dataset only has training set, the validation set split is done in the experiment part
        edge_label = {}
        for message_type in split_types:
            negative_labels = []
            negative_sources = []
            negative_targets = []
            sources = hetero.edge_label_index[message_type][0]
            targets = hetero.edge_label_index[message_type][1]
            edge_label[message_type] = torch.ones(size = sources.shape) # positive labels
            for idx, source_node in enumerate(sources):
                avoidance = targets[idx]
                target_attr_values = torch.unique(targets).tolist()
                if negative_sampling_ratio <= len(target_attr_values):
                    negative_target_nodes = torch.tensor(random.sample(target_attr_values, negative_sampling_ratio))
                else:
                    negative_target_nodes = torch.tensor(random.sample(target_attr_values, 1))
                negative_labels.append(torch.zeros_like(negative_target_nodes))
                negative_sources.append(torch.tile(source_node, (len(negative_target_nodes),)))
                negative_targets.append(negative_target_nodes)

            negative_labels = torch.concat(negative_labels, -1)
            negative_sources = torch.concat(negative_sources, -1)
            negative_targets = torch.concat(negative_targets, -1)
            negative_edge_label_index = torch.stack([negative_sources, negative_targets], dim = 0)

            edge_label[message_type] = torch.concat([edge_label[message_type], negative_labels], -1)
            hetero.edge_label_index[message_type] = torch.concat([hetero.edge_label_index[message_type], negative_edge_label_index], -1)
        
        hetero.edge_label = edge_label
        return hetero
        

def process_dataset(hetero, directed, split_types, split_ratio = [0.8, 0.1, 0.1]):
# The official process for a link prediction task where links are randomly sampled
# create dataset object in link-prediction mode
    dataset = GraphDataset(
        [hetero], 
        task='link_pred', 
        edge_negative_sampling_ratio = 1.5, 
        edge_train_mode = 'all', 
        resample_negatives = True, 
        # resample_disjoint=False, 
        # resample_disjoint_period=30
    )

    # split dataset for link prediction and adapt to torch.Dataloader
    # only split on edges sourcing from 'name'
    if directed == True:
        dataset_train, dataset_val, dataset_test = dataset.split(transductive=True,
                                                                split_ratio=split_ratio, 
                                                                split_types=split_types)
    else: 
        dataset_train, dataset_val, dataset_test = dataset.split(transductive=True,
                                                                split_ratio=split_ratio)
    
    train_loader = DataLoader(dataset_train, collate_fn=Batch.collate(),
                        batch_size=1)
    val_loader = DataLoader(dataset_val, collate_fn=Batch.collate(),
                        batch_size=1)
    test_loader = DataLoader(dataset_test, collate_fn=Batch.collate(),
                        batch_size=1)
    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    return dataloaders