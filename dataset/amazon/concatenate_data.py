import pandas as pd
import os
import glob
import numpy as np


'''
Some extra cleaning operation are done manually on the original_clean_data.csv.
Please do not run this file to re-generate the original data
'''

# loop over the list of csv files
dataFrames = []
path = os.path.join(os.getcwd(), 'dataset/amazon\clean_data')
files = glob.glob(os.path.join(path, "*.xlsx"))
for f in files:
    name = f.split('\\')[-1].split('.')[0]
    # drop 3 colunms and add 'name' according to file_name
    df = pd.read_excel(f)
    df = df.drop(df.columns[0:3], axis=1)
    df['name'] = name
    dataFrames.append(df)

# concatenate files
df = pd.concat(dataFrames, ignore_index=True)
path0 = os.path.join(os.getcwd(), 'dataset/amazon/new_original_clean_data.csv')
# save the original data file
df.to_csv(path0)