
import os
import _pickle
import numpy as np
import pandas as pd

CVNAMES = ['A', 'B', 'C', 'D', 'E']

# %% Dataset loader function


X_train = y_train = np.array([])
for cv in CVNAMES:
    data = pd.read_csv('./'+cv+'.txt').values
    X_data = 255 - data[:, 1:].astype('float32')
    y_data = data[:, 0].astype('int')
    concat = np.c_[y_data, X_data]
    np.savetxt(cv+'_fixed.txt', concat, fmt='%d', delimiter=',')


