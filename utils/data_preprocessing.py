import networkx as nx
import torch
from deepsnap.dataset import GraphDataset
from deepsnap.batch import Batch
from torch.utils.data import DataLoader
import random
from torch.utils.data import Dataset
import copy

class CustomDataset(Dataset):
    def __init__(self, hetero, split_types, negative_sampling = True, negative_sampling_ratio = 1, sampling_epoch = 50):
        self.hetero = hetero
        self.split_types = split_types
        self.negative_sampling = negative_sampling
        self.negative_sampling_ratio = negative_sampling_ratio
        self.sampling_epoch = sampling_epoch
        self.epoch_counter = 0
        self.sampled_hetero = self.create_custom_dataset(hetero = copy.deepcopy(self.hetero), split_types=self.split_types, negative_sampling_ratio=self.negative_sampling_ratio)


    def __len__(self):
        return len(self.hetero)
    
    def __getitem__(self, index):
        self.epoch_counter += 1
        if self.negative_sampling:
            if self.epoch_counter % self.sampling_epoch == 0:
                self.sampled_hetero = self.create_custom_dataset(hetero = copy.deepcopy(self.hetero), split_types=self.split_types, negative_sampling_ratio=self.negative_sampling_ratio)
        
        return self.sampled_hetero


    def create_custom_dataset(self, hetero, split_types, negative_sampling_ratio):
        # The custom dataset specially designed for the object property modelling experiment
        # links are sampled with 'typed negative-sampling strategy'
        # The dataset only has training set, the validation set split is done in the experiment part
        edge_label = {}
        hetero = hetero[0]
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
        resample_negatives=True, 
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