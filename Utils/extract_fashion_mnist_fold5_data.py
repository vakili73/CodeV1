# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 23:01:26 2018

@author: vvaki
"""

from utils import mnist_reader

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

np.random.seed(0)

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')

concat_X = np.concatenate((X_train, X_test), axis=0)
concat_y = np.concatenate((y_train, y_test), axis=0)
concat = np.c_[concat_y, concat_X]

plt.imshow(np.reshape(concat_X[5, :], (28, 28))); plt.show()

skf = StratifiedKFold(n_splits=5, shuffle=True)

file_name = ['A.txt', 'B.txt', 'C.txt', 'D.txt', 'E.txt']
file_index = 0

for _, test_index in skf.split(concat_X, concat_y):
    np.savetxt(file_name[file_index], concat[test_index, :], fmt='%d', delimiter=',')
    file_index += 1
    test = concat_y[test_index]
    unique, counts = np.unique(test, return_counts=True)
    print(dict(zip(unique, counts)))
