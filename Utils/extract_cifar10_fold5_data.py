# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 23:01:26 2018

@author: vvaki
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold

np.random.seed(0)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

if __name__ == '__main__':
    
    data_file = ['data_batch_1', 'data_batch_2', 'data_batch_3',
                 'data_batch_4', 'data_batch_5', 'test_batch']
    
    flag = True
    for file in data_file:
        temp = unpickle('./cifar-10-batches-py/' + file)
        
        if flag:
            X = temp[b'data']
            y = temp[b'labels']
            flag = False
        else:
            X = np.concatenate((X, temp[b'data']))
            y = np.concatenate((y, temp[b'labels']))
            
    
    X = np.transpose(np.reshape(X, (-1, 3, 32, 32)), (0, 2, 3, 1)).reshape((-1, 32*32*3))
    plt.imshow(np.reshape(X[52, :], (32, 32, 3))); plt.show()
    
    concat = np.c_[y, X]
    
    skf = StratifiedKFold(n_splits=5, shuffle=True)

    file_name = ['A.txt', 'B.txt', 'C.txt', 'D.txt', 'E.txt']
    file_index = 0
    
    for _, test_index in skf.split(X, y):
        np.savetxt(file_name[file_index], concat[test_index, :], fmt='%d', delimiter=',')
        file_index += 1
        test = y[test_index]
        unique, counts = np.unique(test, return_counts=True)
        print(dict(zip(unique, counts)))
        