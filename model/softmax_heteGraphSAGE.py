import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsnap.hetero_gnn import forward_op, HeteroConv

# Define the heterogeneous GNN for the link prediction task
class HeteroGNN(torch.nn.Module):
    def __init__(self, gnn_conv, hetero, hidden_size, num_layer_hop, encoder_layer_num):
        super(HeteroGNN, self).__init__()
        
        # Wrap the heterogeneous GNN layers with HeteroConv
        # The wrapper will (use adj matrix as indices): 
        #       1. propagate source node messages with different GraphSAGE layers according to message types
        #       2. for each destination node, aggregate propagation embeddings of different message types

        self.embedding_layer = nn.ModuleDict()
        self.encoder_mlp = nn.ModuleDict()
        self.layer_convs = torch.nn.ModuleList()

        # Wrap the hetero convolutional layers with HeterConv wrapper
        layer_convs = self.initialize_hetero_gnn_layers(hetero, gnn_conv, hidden_size, num_layer_hop)
        for convs in layer_convs:
            # transfer to modulelist in avoidance of device inconsistance
            self.layer_convs.append(HeteroConv(convs))
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
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
            # self.encoder[node_type] = nn.Linear(hetero.num_node_features('name'),hidden_size)
            # self.relus0[node_type] = nn.LeakyReLU()
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
        # data with labels for prediction
        # only predict edges sourcing from 'name'
        for message_type in data.edge_label: # must use edge_label instead of edge_label_index here
            nodes_first = torch.index_select(x[message_type[0]], 0, data.edge_label_index[message_type][0,:].long())
            nodes_second = torch.index_select(x[message_type[2]], 0, data.edge_label_index[message_type][1,:].long())
            
            attribute_nodes_label = torch.index_select(data.node_label[message_type[2]], 0, data.edge_label_index[message_type][1,:].long())
            pred[message_type] = torch.sum(nodes_first * nodes_second, dim=-1)
            pred_attribute_values[message_type] = attribute_nodes_label
        return pred, pred_attribute_values

    def loss(self, pred, y):
        loss = 0
        for key in pred:
            p = torch.sigmoid(pred[key])
            # loss += self.loss_fn(pred[key], y[key].type(pred[key].dtype))
            loss += self.loss_fn(p, y[key].type(pred[key].dtype))
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