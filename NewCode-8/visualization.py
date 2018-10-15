# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:04:56 2018

@author: vvaki
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./feature/gisette_original_schema_1_dense_none_linear_layer_1_test.txt', ',')
data = data.values

feature = data[400:500, 1:]
target = data[400:500, 0]

plt.scatter(feature[(target == 0), 0], feature[(target == 0), 1])
plt.hold(True)
plt.scatter(feature[(target == 1), 0], feature[(target == 1), 1])
plt.xlim(feature.min()-1, feature.max()+1)