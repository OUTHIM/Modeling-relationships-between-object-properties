from cv2 import threshold
import pandas as pd
from pathlib import Path
import sys
import os
import torch
import torch.nn.functional as F

FILE = Path(__file__).resolve()
FATHER = FILE.parents[0]  # root directory
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(FATHER) not in sys.path:
    sys.path.append(str(FATHER))

from experiment.old_batch_experiment_utils import test_one_sample
from dataset.amazon import quantize

test_data = pd.read_csv(os.path.join(FATHER, 'test_data.csv'))
test_data = test_data.drop(test_data.columns[0], axis=1).to_dict(orient='records')

# start experiment
threshold = 0.5
correct_pred = 0
acc = 0
test_attr = ['Volume']

labels = {}
for sample_index, sample in enumerate(test_data):
    for name in test_attr:
        labels[name] = sample.pop(name)
    results, real_bins, attr_names = test_one_sample(sample, dataset_name='amazon', folder_path=FATHER)
    # results = F.sigmoid(torch.from_numpy(results)).cpu().detach().numpy()
    for i in range(len(results)):
        if results[i][real_bins[i] == labels[attr_names[i]]] >= threshold:
            print('For sample {}, \033[96m {} \033[0m is tested \033[92;1m Correct \033[0m'.format(sample_index, attr_names[i]))
            correct_pred += 1
        else:
            print('For sample {}, \033[96m {} \033[0m is tested \u001b[31;1m Wrong \033[0m'.format(sample_index, attr_names[i]))
        print('Ground truth label:', labels[attr_names[i]])
        print('Confidence:', results[i][real_bins[i] == labels[attr_names[i]]])

acc = correct_pred / len(test_data)
print(acc)


