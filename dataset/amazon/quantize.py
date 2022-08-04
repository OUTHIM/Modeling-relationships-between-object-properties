from sklearn.preprocessing import KBinsDiscretizer
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
from pathlib import Path

def array_to_list(x):
    return x.reshape([1,-1]).squeeze().tolist()

def test_on_quantization_levels(df, max_levels, strategy = 'kmeans', figure_on = True):
    heads = ['Weight', 'Volume', 'Length', 'Width', 'Height']
    quantization_levels = {}
    avg_loss_history = {}
    # test the impact of quantization levels
    for head in heads:
        avg_quantization_loss = []
        for level in range(5, max_levels + 1):
            temp = {}
            data = df[head].to_numpy()
            data = data.reshape([-1,1])
            enc = KBinsDiscretizer(n_bins = level, encode="ordinal", strategy= strategy)
            X_binned = enc.fit_transform(data)
            X_quantized = enc.inverse_transform(X_binned)
            # average loss after quantization
            avg_quantization_loss.append(np.mean(abs(X_quantized - data)))
            # plt.scatter(X_binned.reshape([1,-1]).squeeze(),data.reshape([1,-1]).squeeze())
            levels = list(set(enc.inverse_transform(X_binned).reshape([1,-1]).squeeze().tolist()))

            # print(quantized_data)
            # print('levels are: ', levels)
            
            # save the quantized data
            # df[head] = X_binned
            X_binned += 1
            X_quantized = array_to_list(X_quantized)
            X_binned = array_to_list(X_binned)

            for i in range(len(X_quantized)):
                temp[str(X_quantized[i])] = X_binned[i] + 1
            
            # quantization_levels[head] = temp
        
        avg_loss_history[head] = avg_quantization_loss

    # visualize the quantization loss for different number of levels
    if figure_on:
        for i, head in enumerate(avg_loss_history):
            plt.figure(i)
            plt.title(head)
            plt.plot(avg_loss_history[head])
        plt.show()

def quantization(df, num_levels, save_path, save_file = True, figure_on = True, strategy = 'kmeans'):
    quantization_levels = {}
    heads = ['Weight', 'Volume', 'Length', 'Width', 'Height']
    #%% Quantization
    for i, head in enumerate(heads):
        avg_quantization_loss = []
        temp = {}
        data = df[head].to_numpy()
        data_copy = data[:]
        data = data.reshape([-1,1])
        enc = KBinsDiscretizer(n_bins = num_levels, encode="ordinal", strategy= strategy)
        X_binned = enc.fit_transform(data)
        X_quantized = enc.inverse_transform(X_binned)
        # plt.scatter(X_binned.reshape([1,-1]).squeeze(),data.reshape([1,-1]).squeeze())
        levels = np.sort(np.unique(X_quantized.reshape([1,-1]).squeeze()))
        levels = levels.tolist()
        X_quantized = array_to_list(X_quantized)
        X_binned = array_to_list(X_binned)

        # The bin value of levels are set according to their real values
        # create the dict for quantization
        for i in range(len(levels)):
            temp[str(levels[i])] = i+1
        quantization_levels[head] = temp

        # substitute the values into bins
        for i in range(len(X_quantized)):
            X_binned[i] = temp[str(X_quantized[i])]
        
        # save the quantized data
        df[head] = X_binned

        # plot the clusters
        if figure_on:
            plt.figure(i)
            labels = np.array(X_binned)
            for j in range(1, num_levels+1):
                cluster = data_copy[labels==j]
                plt.scatter(np.ones(len(cluster)), cluster)
        
            plt.title(head)
            plt.show()


    # quantize the discrete labels
    for head in df:
        temp = {}
        if head in heads:
            continue
        levels = df[head].unique()

        for i, level in enumerate(levels):
            if level == 0:
                temp[str(level)] = 1
            elif level == 1:
                temp[str(level)] = 2
            else:
                temp[str(level)] = i+1

        quantization_levels[head] = temp
        for i in range(len(df[head])):
            value = str(df[head][i])
            df[head][i] = temp[value]

    # df = df.drop(df.columns[0:2], axis=1)

    # save the quantized data
    if save_file:
        path2 = os.path.join(save_path, 'quantized_clean_data.csv')
        df.to_csv(path2)

        # save the quantization levels
        path3 = os.path.join(save_path, 'quantization_levels.json')
        with open(path3, 'w') as outfile:
            json.dump(quantization_levels, outfile)

    print('Quantization finished!')
    # plt.hist(X_binned)
    # plt.show()

# print(data)
# print(np.concatenate([enc.inverse_transform(X_binned), data], axis=-1))

if __name__ == '__main__':
    path = os.path.join(os.getcwd(), 'dataset/amazon\clean_data.csv')
    df = pd.read_csv(path)
    df = df.drop(df.columns[0:3], axis=1)
    CURRENT_FILE = Path(__file__).resolve()
    FATHER = CURRENT_FILE.parents[0]  # root directory
    max_levels = 200
    num_levels = 100
    strategy = 'kmeans'
    test_on_quantization_levels(df, max_levels, strategy = 'kmeans', figure_on = False)
    quantization(df, num_levels, save_path = FATHER, save_file = True, strategy='kmeans')