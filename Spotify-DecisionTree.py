#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("../input/spotify-dataset-19212020-160k-tracks/data_by_year.csv", sep=",")

x = df.instrumentalness.values.reshape(-1, 1) 
y = df.danceability.values.reshape(-1, 1)

import matplotlib.pyplot as plt

plt.scatter(x, y)
plt.show()

from sklearn.tree import DecisionTreeRegressor

decision_tree = DecisionTreeRegressor()
decision_tree.fit(x, y)

x_ = np.arange(min(x), max(x), 0.05).reshape(-1, 1)
y_predict = decision_tree.predict(x_)

plt.scatter(x, y)
plt.xlabel("Instrumentalness")
plt.ylabel("Danceability")
plt.plot(x_, y_predict, color="red")
plt.show()

        
        
        
        
        

