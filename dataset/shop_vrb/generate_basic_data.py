import os
import json
import pandas as pd

cwd = os.getcwd()

filepath_train = os.path.join(cwd, 'dataset\SHOP_VRB_scenes\SHOP_VRB_train_scenes.json')
with open(filepath_train) as f:
    file = json.load(f)

data = []
scenes = file['scenes']
for image in scenes:
    for object in image['objects']:
        data.append(object)

df = pd.DataFrame(data)
# df = df.drop(columns = ['mask','3d_coords','rotation','pixel_coords','attribute','powering','disassembly'])
df = df.drop(columns = ['mask','3d_coords','rotation','pixel_coords','attribute'])

df.to_csv(os.path.join(cwd, 'Dataset/GRAPH_SHOP_VRB/basic_training_data.csv'))
