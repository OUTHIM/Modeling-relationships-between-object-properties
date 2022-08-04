
import pandas as pd
from pathlib import Path
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import copy

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

# Due to limited time, the loop is not deleted in this function 
# even though every time only one missing attribute is tested.
def evaluation(
        experiment_folder_path,
        threshold = 0.5,
        highest_among_others = True,
        softmax_model = True,
        model = None
        ):
    '''
    experiment_folder_path: 
        under this path should contain
            1. The test model named as 'amazon_best.pth'
            2. The evaluation data named as 'test_data.csv'
    return
        attr_acc: a dict contains acc of each attribute
        acc: the average acc of all attributes
    '''

    # load data
    test_data_path = os.path.join(experiment_folder_path, 'test_data.csv')
    data = pd.read_csv(test_data_path)
    data = data.drop(data.columns[0], axis=1)

    test_attrs = data.columns.values.tolist()
    test_attrs = [x for x in test_attrs if x != 'name']

    # Only evaluate in the situation that 1 attribute is missed for each object
    avg_acc = 0
    attr_acc = {}
    for attr_name in test_attrs:
        test_data = copy.deepcopy(data)
        temp_acc = 0
        labels = {}
        temp = test_data[attr_name].to_numpy()
        labels[attr_name] = temp
        test_data = test_data.drop(attr_name, axis=1) # drop the attribute that needs evaluation
        test_data = test_data.to_dict(orient='records')

        # start experiment
        results, real_bins, ordered_attr_names, quantization_num = test_samples(test_data, dataset_name='amazon', folder_path=experiment_folder_path, softmax_model=softmax_model, model = model)
        for key in labels:
            labels[key] = np.repeat(labels[key], quantization_num)

        for i, ordered_attr_name in enumerate(ordered_attr_names):
            true_false_list = real_bins[i] == labels[ordered_attr_name]
            temp_pred = results[i][true_false_list]
            if highest_among_others:
                reduce_largest = np.maximum.reduceat(results[i], np.r_[:len(results[i]):quantization_num])
                temp_acc = np.sum(np.logical_or(temp_pred > threshold, temp_pred == reduce_largest))/len(test_data)
            else:
                temp_acc = (np.sum(temp_pred > threshold))/len(test_data)
        
        print('Accuracy on attribute {} is {}'.format(ordered_attr_name, temp_acc))
        attr_acc[ordered_attr_name] = temp_acc
        avg_acc += temp_acc
    
    avg_acc = avg_acc/len(test_attrs)
    print('Overall evaluation accuracy is:', avg_acc)
    return attr_acc, avg_acc

if __name__ == '__main__':
    FILE = Path(__file__).resolve()
    FATHER = FILE.parents[0]  # root directory
    ROOT = FILE.parents[1]
    evaluation(experiment_folder_path=FATHER)
