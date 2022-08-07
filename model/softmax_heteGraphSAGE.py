import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsnap.hetero_gnn import forward_op, HeteroConv
import math
import random
import copy

# Define the heterogeneous GNN for the link prediction task
class softmax_HeteroGNN(torch.nn.Module):
    def __init__(self, gnn_conv, hetero, hidden_size, num_layer_hop, encoder_layer_num, drop_softmax_ratio):
        super(softmax_HeteroGNN, self).__init__()
        
        # Wrap the heterogeneous GNN layers with HeteroConv
        # The wrapper will (use adj matrix as indices): 
        #       1. propagate source node messages with different GraphSAGE layers according to message types
        #       2. for each destination node, aggregate propagation embeddings of different message types
        self.drop_softmax_ratio = drop_softmax_ratio
        self.embedding_layer = nn.ModuleDict()
        self.encoder_mlp = nn.ModuleDict()
        self.layer_convs = torch.nn.ModuleList()

        # Wrap the hetero convolutional layers with HeterConv wrapper
        layer_convs = self.initialize_hetero_gnn_layers(hetero, gnn_conv, hidden_size, num_layer_hop)
        for convs in layer_convs:
            # transfer to modulelist in avoidance of device inconsistance
            self.layer_convs.append(HeteroConv(convs))
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.bns = torch.nn.ModuleList()
        self.relus = torch.nn.ModuleList()
        for _ in range(num_layer_hop):
            self.bns.append(nn.ModuleDict())
            self.relus.append(nn.ModuleDict())
        # the embedding layer
        for node_type in hetero.node_types:
            self.embedding_layer[node_type] = self.initialize_embedding_layers(int(torch.max(hetero.node_feature[node_type]).item())+1, hidden_size)
        # Each node type has an encoder
        # After updating node features, the BN and ReLU layers are set according to node types
        for node_type in hetero.node_types:
            self.encoder_mlp[node_type] = self.initialize_encoder_mlp_layers(hetero.num_node_features('name'), hidden_size, encoder_layer_num)
            for i in range(num_layer_hop):
                self.bns[i][node_type] = nn.BatchNorm1d(hidden_size)
                self.relus[i][node_type] = nn.LeakyReLU()

    def forward(self, data):
        # data to generate embedding
        x = data.node_feature

        edge_index = data.edge_index
        # embedding layer
        x = forward_op(x, self.embedding_layer)
        for node_type in x:
            x[node_type] = x[node_type].squeeze()

        # encoder layer
        x = forward_op(x, self.encoder_mlp)
        for i in range(len(self.layer_convs)):
            x = self.layer_convs[i](x, edge_index)
            # forward_op apply layers that matches the type of nodes
            x = forward_op(x, self.bns[i])
            # not using ReLU at the last layer improves a lot the performance
            if i != len(self.layer_convs)-1:
                x = forward_op(x, self.relus[i])

        pred = {}
        pred_attribute_values = {}
        true_attr_node_labels = {}

        # data with labels for prediction
        # only predict edges sourcing from 'name'
        for message_type in data.edge_label: # must use edge_label instead of edge_label_index here
            name_nodes = torch.index_select(x[message_type[0]], 0, data.edge_label_index[message_type][0,:].long())
            attr_nodes = torch.transpose(x[message_type[2]], 0, 1)
            node_distmult = torch.mm(name_nodes, attr_nodes).to('cuda')

            # Drop attribute nodes before softmax to reduce overfitting
            # Only in training mode
            if self.training and self.drop_softmax_ratio != None:
                drop_dismult_num = math.floor(node_distmult.size()[1] * self.drop_softmax_ratio)
                mask_drop = torch.zeros_like(node_distmult).to('cuda')
                for _ in range(drop_dismult_num):
                    indices_to_drop = torch.randint(0, node_distmult.size()[1], (node_distmult.size()[0],)).to('cuda')
                    temp = F.one_hot(indices_to_drop, num_classes = node_distmult.size()[1]).to('cuda')
                    mask_drop = torch.logical_or(temp, mask_drop).to('cuda')

                mask_drop = torch.logical_not(mask_drop)
                indices_to_preserve = data.edge_label_index[message_type][1,:].to('cuda')
                mask_preserve = F.one_hot(indices_to_preserve.long(), num_classes = node_distmult.size()[1]).to('cuda')
                mask_final = torch.logical_or(mask_drop, mask_preserve).to('cuda')
                node_distmult = node_distmult * mask_final

            attribute_nodes_label = torch.tile(data.node_label[message_type[2]], (len(name_nodes),))

            pred[message_type] = node_distmult
            true_attr_node_labels[message_type] = data.edge_label_index[message_type][1,:].long()
            pred_attribute_values[message_type] = attribute_nodes_label

        return pred, true_attr_node_labels, pred_attribute_values

    def loss(self, pred, y):
        loss = 0
        for key in pred:
            p = F.softmax(pred[key])
            loss += self.loss_fn(p, y[key].long())
        return loss

    # initialize k-hop conv layers for different message/edge types
    def initialize_hetero_gnn_layers(self, hete, gnn_conv, hidden_size, num_layer_hop):
        # hete: the heterogeneous graph (to set the input and output dims)
        # gnn_conv: the basic GNN layer for each message type (e.g. GraphSAGE)
        # hidden_size: the hidden dim of node features in the GNN
        # num_layer_hop: number of layers(hops) for the hetero GNN model
        layer_convs = []
        for i in range(num_layer_hop):
            layer_convs.append({}) # each hop owns a message-type dict of GNNs
            for message_type in hete.message_types:
                # n_type = message_type[0] # neighbour node type
                # s_type = message_type[2] # source node type
                # n_feat_dim = hete.num_node_features(n_type)
                # s_feat_dim = hete.num_node_features(s_type)
                layer_convs[i][message_type] = gnn_conv(hidden_size, hidden_size, hidden_size)
        
        return layer_convs

    def initialize_encoder_mlp_layers(self, input_dim, hidden_size, layer_num):
        mlp = nn.Sequential()
        for i in range(layer_num):
            # if i == 0:
            #     mlp.append(nn.Linear(input_dim, hidden_size))
            # else:
            #     mlp.append(nn.Linear(hidden_size, hidden_size))
            mlp.append(nn.Linear(hidden_size, hidden_size))
            mlp.append(nn.LeakyReLU())
        
        return mlp

    def initialize_embedding_layers(self, num_embeddings, embedding_dims):
        return nn.Embedding(num_embeddings, embedding_dims)