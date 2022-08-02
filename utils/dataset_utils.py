import networkx as nx
import os
import matplotlib.pyplot as plt
import json

def visualize_graph(G, folder_path):
    cwd = os.getcwd()
    # get node labels
    node_label = {}
    # if dataset_name == 'shop_vrb':
    #     path2 = os.path.join(cwd, 'dataset\shop_vrb\quantization_levels.json')
    # elif dataset_name == 'amazon':
    #     path2 = os.path.join(cwd, 'Dataset\Amazon\quantization_levels.json')
    path2 = os.path.join(folder_path, 'quantization_levels.json')
    with open(path2) as f:
        quantization_levels = json.load(f)
    # create node label dict
    for i, label in enumerate(quantization_levels):
        node_label[label] = i

    node_color = [node_label[node[1]['node_type']] for node in G.nodes(data=True)]
    labels = nx.get_node_attributes(G, 'node_type')
    pos = nx.spring_layout(G)

    plt.figure(figsize=(20, 20))
    nx.draw(G, pos=pos, cmap=plt.get_cmap('coolwarm'), node_color=node_color,  labels=labels, font_color='black')
    plt.show()