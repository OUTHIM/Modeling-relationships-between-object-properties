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

from shop_vrb.batch_experiment_utils import test_one_sample
from dataset.amazon import quantize

test_data = pd.read_csv(os.path.join(FATHER, 'quantized_training_data.csv'))
test_data = test_data.drop(test_data.columns[0], axis=1).to_dict(orient='records')

# start experiment
threshold = 0.5
correct_pred = 0
acc = 0

for i, sample in enumerate(test_data[0:50]):
    label = sample.pop('weight')
    results, real_bins = test_one_sample(sample, dataset_name='shop_vrb', folder_path=FATHER)
    # results = F.sigmoid(torch.from_numpy(results)).cpu().detach().numpy()
    if results[real_bins == label] >= threshold:
        print('Sample {} is tested Correct!'.format(i))
        correct_pred += 1
    else:
        print('Sample {} is tested Wrong!'.format(i))
    print('Ground truth label:', label)
    print('Confidence:', results[real_bins == label])

acc = correct_pred / len(test_data)
print(acc)


