'''
This is a customized version for experiment.
Arbitrary number of missing attributes can be tested here.
'''

import pandas as pd
from pathlib import Path
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

FILE = Path(__file__).resolve()
FATHER = FILE.parents[0]  # root directory
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(FATHER) not in sys.path:
    sys.path.append(str(FATHER))

from experiment.batch_experiment_utils import test_samples
from dataset.amazon import quantize

def cal_hit_n(n, pred, true_false_list):
    indices = np.argsort(pred)
    sorted_true_false_list = true_false_list[indices]

    return np.sum(sorted_true_false_list[-n:])


# args
test_attr = ['Volume']
threshold = 0.5
correct_pred = 0
acc = 0
highest_among_others = True
softmax_model = True

# load data
labels = {}
test_data = pd.read_csv(os.path.join(FATHER, 'training_data.csv'))
for attr_name in test_attr:
    temp = test_data[attr_name].to_numpy()
    labels[attr_name] = temp

test_data = test_data.drop(test_attr, axis=1)
test_data = test_data.drop(test_data.columns[0], axis=1).to_dict(orient='records')

# start experiment
results, real_bins, attr_names, quantization_num = test_samples(test_data, dataset_name='amazon', folder_path=FATHER, softmax_model=softmax_model)
for key in labels:
    labels[key] = np.repeat(labels[key], quantization_num)

acc = 0
hit_10 = 0
for i, attr_name in enumerate(attr_names):
    true_false_list = real_bins[i] == labels[attr_name]
    temp_pred = results[i][true_false_list]
    if highest_among_others:
        reduce_largest = np.maximum.reduceat(results[i], np.r_[:len(results[i]):quantization_num])
        temp_acc = np.sum(np.logical_or(temp_pred > threshold, temp_pred == reduce_largest))/len(test_data)
    else:
        temp_acc = (np.sum(temp_pred > threshold))/len(test_data)

    # temp_hit_10 = cal_hit_n(10, temp_pred, true_false_list)
    # hit_10 += temp_hit_10
    print('Accuracy on attribute {} is {}:'.format(attr_name, temp_acc))
    acc += temp_acc

print('Average accuracy is:', acc/(i+1))
print('Hit@10 is:', hit_10/(i+1))
print(results[0].shape)
print(len(results[0]))
print(np.sum(results[0]>0.5))
# print(results[0])
