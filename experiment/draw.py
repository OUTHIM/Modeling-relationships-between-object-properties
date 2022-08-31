import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('C:/Users/yjn_1/Documents/GitHub/Modeling-relationships-between-object-properties/experiment/quantized_clean_data.csv')
print(df.head(10))

sns.scatterplot(x=df['Weight'],y = np.ones(len(df['Weight'],)) + 0.1*np.random.normal(0, 1, len(df['Weight'])), hue=df['Material'])
plt.show()