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
from torch.utils.data import DataLoader
from deepsnap.batch import Batch
import torch.nn.functional as F

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
FATHER = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
from model.heteGraphSAGE import HeteroGNN
from utils.data_preprocessing import CustomDataset, process_dataset
from model.softmax_heteGraphSAGE import softmax_HeteroGNN

# Train function
def train(model, dataloaders, optimizer, args, save_hard_samples, retrain_hard_samples, save_path):
    val_max = 0
    iteration_num_for_batch = 50
    t_accu = []
    v_accu = []
    e_accu = []
    val_avg = 0
    hard_batches = []
    for epoch in range(1, args["epochs"] + 1):
        for _, batch in enumerate(dataloaders['train']):
            batch.to(args["device"])
            model.train()
            optimizer.zero_grad()
            pred, _ = model(batch)
            loss = model.loss(pred, batch.edge_label)
            loss.backward()
            optimizer.step()

            log = 'Epoch: {:03d}, Train loss: {:.4f}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            accs, hard_batch = test(model, dataloaders, args, epoch)
            # record hard batches
            if save_hard_samples:
                if hard_batch:
                    hard_batches.append(hard_batch)

            t_accu.append(accs['train'])
            v_accu.append(accs['val'])
            e_accu.append(accs['test'])

            print(log.format(epoch, loss.item(), accs['train'], accs['val'], accs['test']))
            if val_max < accs['val']:
                val_max = accs['val']
            
            if epoch > args['epochs'] - 50:
                val_avg += accs['val']

    path = os.path.join(save_path, 'hard_batches.pickle')
    with open(path, 'wb') as handle:
        pickle.dump(hard_batches, handle, protocol=pickle.HIGHEST_PROTOCOL)

    if retrain_hard_samples:
        before_hard_training_model = copy.deepcopy(model)
        # save the hard_batches
        print('------ Start training on hard batches -------')
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


    log = 'Final: Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    accs, _ = test(model, dataloaders, args, epoch)
    print(log.format(accs['train'], accs['val'], accs['test']))
    print('average val accuracy over last 50 epoches is: ', val_avg/50)

    if retrain_hard_samples:
        return t_accu, v_accu, e_accu, model, before_hard_training_model
    else:
        return t_accu, v_accu, e_accu, model

# For training on custom dataset with negative sampling
def train_custom(model, dataloader, optimizer, args):
    for epoch in range(1, args['epochs'] + 1):
        for train_batch in dataloader:
            train_batch.to(args["device"])
            model.train()
            optimizer.zero_grad()
            pred, _ = model(train_batch)
            loss = model.loss(pred, train_batch.edge_label)
            loss.backward()
            optimizer.step()

            acc = 0
            # eval_batches = [train_batch, val_batch, test_batch]
            model.eval()
            num = 0
            train_batch.to(args["device"])
            pred, _ = model(train_batch)
            for key in pred:
                p = torch.sigmoid(pred[key]).cpu().detach().numpy()
                pred_label = np.zeros_like(p, dtype=np.int64)
                pred_label[p > 0.5] = 1
                pred_label[p <= 0.5] = 0
                acc += np.sum(pred_label == train_batch.edge_label[key].cpu().numpy())
                num += len(pred_label)
            acc = acc / num

            log = 'Current epoch: {:02d}/{} Train loss: {:.4f}, Train accuracy: {:.4f}'
            print(log.format(epoch, args['epochs'], loss.item(), acc))

    return model

# For training with softmax heteGraphSAGE
def train_softmax(model, dataloader, optimizer, args):
    print('Training with softmax model...')
    for epoch in range(1, args['epochs'] + 1):
        for train_batch in dataloader:
            train_batch.to(args["device"])
            model.train()
            optimizer.zero_grad()
            pred, y, _ = model(train_batch)
            loss = model.loss(pred, y)
            loss.backward()
            optimizer.step()

            acc = 0
            # eval_batches = [train_batch, val_batch, test_batch]
            model.eval()
            num = 0
            train_batch.to(args["device"])
            pred, y, _ = model(train_batch)
            for key in pred:
                p = F.softmax(pred[key])
                target = torch.reshape(y[key], (-1,1))
                target_p = torch.gather(p, 1, target).cpu().detach().numpy()
                acc += np.sum(target_p > 0.5)
                num += len(p)
            acc = acc / num

            log = 'Current epoch: {:02d}/{} Train loss: {:.4f}, Train accuracy: {:.4f}'
            print(log.format(epoch, args['epochs'], loss.item(), acc))

    return model

def test(model, dataloaders, args, epoch):
    hard_sample_threshold = 0.75
    epoch_threshold = 100
    model.eval()
    accs = {}
    hard_batch = {}
    for mode, dataloader in dataloaders.items():
        acc = 0
        for i, batch in enumerate(dataloader):
            hard_batch[mode] = batch
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
        batch.to('cpu')
        accs[mode] = acc / num

    # if epoch > 300:
    #     hard_sample_threshold += 0.05
    if accs['train'] <= hard_sample_threshold and epoch >= epoch_threshold:
        return accs, hard_batch

    return accs, False


# start training
# print(hetero.num_edges(('name','name-movability','movability')))
# print(hetero.message_types)

## arguments to tune the model
def start_training(hetero, args, save_path, save_file = True, file_name = 'best_model',save_hard_samples = False, retrain_hard_samples = False, 
                    custom_train = False, negative_sampling = True, negative_sampling_ratio = 1, sampling_epoch = 50, train_with_softmax = False):

    shop_vrb_split_types = [('name', 'name-color', 'color'), ('name', 'name-weight', 'weight'), ('name', 'name-movability', 'movability'), 
                            ('name', 'name-material', 'material'), ('name', 'name-shape', 'shape'), 
                            ('name', 'name-size', 'size'), ('name', 'name-picking', 'picking'), 
                            ('name', 'name-powering', 'powering'), ('name', 'name-disassembly', 'disassembly')]

    amazon_split_types = [('name', 'name-Material', 'Material'), ('name', 'name-Colour', 'Colour'), ('name', 'name-Weight', 'Weight'), 
    ('name', 'name-Volume', 'Volume'), ('name', 'name-Length', 'Length'), ('name', 'name-Width', 'Width'), 
    ('name', 'name-Height', 'Height'), ('name', 'name-Functionality', 'Functionality'), 
    ('name', 'name-Button', 'Button'), ('name', 'name-Lip', 'Lip'), ('name', 'name-Fillability', 'Fillability'), ('name', 'name-Washability', 'Washability'), 
    ('name', 'name-Dismountability', 'Dismountability'), ('name', 'name-Shape', 'Shape'), ('name', 'name-Handle', 'Handle')]
    
    # for testing baking tray only
    # amazon_split_types = [('name', 'name-Material', 'Material'), ('name', 'name-Colour', 'Colour'), ('name', 'name-Weight', 'Weight'), 
    # ('name', 'name-Volume', 'Volume'), ('name', 'name-Length', 'Length'), ('name', 'name-Width', 'Width'), 
    # ('name', 'name-Height', 'Height'), ('name', 'name-Shape', 'Shape'), ('name', 'name-Handle', 'Handle')]

    # Build the model and start training
    # load graph
    dataset_name = args['dataset_name']
    hidden_size = args['hidden_size']
    num_layer_hop = args['layer_num']
    encoder_layer_num = args['encoder_layer_num']
    split_types = amazon_split_types if args['dataset_name'] == 'amazon' else shop_vrb_split_types

    if train_with_softmax:
        model = softmax_HeteroGNN(HeteroSAGEConv, hetero, hidden_size, num_layer_hop, encoder_layer_num).to(args["device"])
    else:
        model = HeteroGNN(HeteroSAGEConv, hetero, hidden_size, num_layer_hop, encoder_layer_num).to(args["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    
    # train
    if custom_train:
        custom_dataset = CustomDataset([hetero], split_types, negative_sampling = negative_sampling, negative_sampling_ratio = negative_sampling_ratio, sampling_epoch = sampling_epoch)
        dataloader = DataLoader(custom_dataset, batch_size=1, collate_fn=Batch.collate())
        best_model = train_custom(model, dataloader, optimizer, args)
    elif train_with_softmax:
        custom_dataset = CustomDataset([hetero], split_types, negative_sampling = False)
        dataloader = DataLoader(custom_dataset, batch_size=1, collate_fn=Batch.collate())
        best_model = train_softmax(model, dataloader, optimizer, args)
    else:
        dataloaders = process_dataset(hetero, directed = args['directed'], split_types=split_types)
        if retrain_hard_samples:
            t_accu, v_accu, e_accu, best_model, before_hard_training_model = train(model, dataloaders, optimizer, args, save_hard_samples = save_hard_samples, retrain_hard_samples = retrain_hard_samples, save_path=save_path)
        else:
            t_accu, v_accu, e_accu, best_model = train(model, dataloaders, optimizer, args, save_hard_samples = save_hard_samples, retrain_hard_samples = retrain_hard_samples, save_path=save_path)
    
    if save_file:
        path1 = os.path.join(save_path, '{0}.pth'.format(file_name))
        torch.save(best_model.state_dict(), path1)
        if retrain_hard_samples:
            path3 = os.path.join(save_path, 'before_hard_training.pth')
            torch.save(before_hard_training_model.state_dict(), path3)
        # save the hyper_parameters of training
        path2 = os.path.join(save_path, '{0}_args.json'.format(dataset_name))
        with open(path2, 'w') as outfile:
            json.dump(args, outfile)


if __name__ == '__main__':
    # # arguments to tune
    args = {
        'dataset_name': 'shop_vrb',
        "device": "cuda",
        "epochs": 200,
        "lr": 0.01,
        "weight_decay": 1e-4,
        'encoder_layer_num': 3,
        "hidden_size": 16,
        'layer_num': 3,
        'directed' : True
        }

    # args = {
    #     'dataset_name': 'amazon',
    #     "device": "cuda",
    #     "epochs": 400,
    #     "lr": 0.01,
    #     "weight_decay": 1e-4,
    #     'encoder_layer_num': 3,
    #     "hidden_size": 32,
    #     'layer_num': 4,
    #     'directed' : True
    #     }

    dataset_name = args['dataset_name']
    cwd = os.getcwd()
    path = os.path.join(cwd, 'dataset/{0}/{1}_graph_data.gpickle'.format(dataset_name, dataset_name))
    G = nx.read_gpickle(path)
    # dataset_utils.visualize_graph(G)
    hetero = HeteroGraph(G)
    CURRENT_FILE = Path(__file__).resolve()
    FATHER = CURRENT_FILE.parents[0]  # root directory
    start_training(hetero, args, save_path = FATHER, save_file=True)