import os
import pandas as pd
import json

cwd = os.getcwd()
path1 = os.path.join(cwd, 'dataset/shop_vrb/basic_training_data.csv')
df = pd.read_csv(path1)

path2 = os.path.join(cwd, 'dataset\shop_vrb\quantization_levels.json')
with open(path2) as f:
    quantization_levels = json.load(f)

df = df.drop(df.columns[0], axis=1)

# replace quantization values
for title in df:
     df[title] = df[title].replace(quantization_levels[title])


df.to_csv(os.path.join(cwd, 'dataset/shop_vrb/quantized_training_data.csv'))
