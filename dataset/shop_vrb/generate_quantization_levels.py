import json
import pandas as pd
import os

def get_unique_values(df, col_name):
    return df[col_name].unique()

indices = {}

# use the original object name indices
cwd = os.getcwd()
path1 = os.path.join(cwd, 'dataset\SHOP_VRB_scenes\SHOP_VRB_obj_name_to_num.json')
obj_name_file = path1
with open(obj_name_file) as f:
    obj_names = json.load(f)
indices['name'] = obj_names

# generate indices for other attributes
path2 = os.path.join(cwd, 'dataset/shop_vrb/basic_training_data.csv')
df = pd.read_csv(path2)
df = df.drop(df.columns[0], axis=1)
df = df.drop(columns = ['name'])
for title in df:
    dict = {}
    for i, element in enumerate(get_unique_values(df, title)):
        dict[element] = i+1
    indices[title] = dict

# manually correct some values
indices['weight']['light'] = 1
indices['weight']['medium-weight'] = 2
indices['weight']['heavy'] = 3

indices['size']['small'] = 1
indices['size']['medium-sized'] = 2
indices['size']['large'] = 3

# print(len(indices))
with open(os.path.join('dataset/shop_vrb','quantization_levels.json'), 'w') as outfile:
    json.dump(indices, outfile)


