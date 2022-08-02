'''
This script generates the clean_data.csv for quantization use.
'''

import pandas as pd
import os
import glob
import numpy as np

def delete_row(df, idx):
    df = df.drop(labels = idx, axis=0)
    return df

def fill_possible_weight(i, df):
    target_coordinate = np.array([df['Length'][i], df['Width'][i], df['Height'][i]], dtype=np.float16)
    closest_position = 0
    closest_dist = float('inf')
    for j in range(len(df['Weight'])):
        if str(df['Weight'][j]) != 'other' and str(df['Weight'][j]) != 'nan' and df['Weight'][j] != None:
            cur_coordinate = np.array([df['Length'][j], df['Width'][j], df['Height'][j]], dtype=np.float16)
            cur_dist = np.sum(np.power(target_coordinate - cur_coordinate, 2))
            if cur_dist <= closest_dist:
                closest_dist = cur_dist
                closest_position = j
    return closest_position

# load the original data
path0 = os.path.join(os.getcwd(), 'dataset/amazon/original_clean_data.csv')
df = pd.read_csv(path0)

rows_to_delete = []
# delete 'Movability'
df = df.drop(columns=['Movability'])

# delete 'fork'
for i in range(len(df['name'])):
    if df['name'][i] == 'fork':
        rows_to_delete.append(i)

# Delete the entry where L,W,H contains OTHER
cols = [df['Length'], df['Width'], df['Height']]
for col in cols:
    for i in range(len(col)):
        if str(col[i]) == 'other' or str(col[i]) == 'nan':
            rows_to_delete.append(i)

# Clean and fill Volume
df = delete_row(df, rows_to_delete)
df = df.reset_index(drop = False)
# clean Volumn column
col = df['Volume']
print(col)
for i in range(len(col)):
    string = str(col[i])
    if string[-1] == 'L':
        col[i] = 1000 * float(string.strip('L'))
    elif string[-2:] == 'oz':
        col[i] = 29.5735 * float(string.strip('oz'))
    elif string[-2:] == 'cl':
        col[i] = 10 * float(string.strip('cl'))
    elif string[-2:] == 'cc':
         col[i] = string.strip('cc')
    elif string[-6:] == 'gallon':
        col[i] = 4546.09 * float(string.strip('gallon'))
    elif string == 'other' or string == 'nan':
        col[i] = float(df['Height'][i]) * float(df['Length'][i]) * float(df['Width'][i])
    else:
        col[i] = string.strip('ml')

# Clean and fill weight
col = df['Weight']
weight_to_fill = []
for i in range(len(col)):
    string = str(col[i])
    if string[-2:] == 'kg':
        col[i] = 1000 * float(string.strip('kg'))
    elif string[-6:] == 'pounds':
        col[i] = 453.592 * float(string.strip('pounds'))
    elif string == 'other' or string == 'nan' or col[i] == None:
        weight_to_fill.append(i)
    else:
        col[i] = string.strip('g')

# fill OTHER in Weights after they are cleaned
for i in weight_to_fill:
    col[i] = df['Weight'][fill_possible_weight(i, df)]

path2 = os.path.join(os.getcwd(), 'dataset/amazon/clean_data.csv')
df.to_csv(path2)
