import os
import numpy as np
import pandas as pd
import datasets as dbs
import algorithms as algs

CVNAMES = ['A', 'B', 'C', 'D', 'E']

import sys
sys._enablelegacywindowsfsencoding()

# %% Load data to Dataset instance


def db_load(dataset):
    dataset = dbs.get_dataset(dataset)
    X_train = y_train = np.array([])
    for cv in CVNAMES:
        data = pd.read_csv(dataset.get_path() + cv + '.txt').values
        if cv == 'E':
            X_test = data[:, 1:].astype('float32')
            y_test = data[:, 0].astype('int')
        else:
            X_train = np.concatenate((X_train, data[:, 1:].astype(
                'float32')), axis=0) if X_train.size else data[:, 1:].astype('float32')
            y_train = np.concatenate((y_train, data[:, 0].astype(
                'int')), axis=0) if y_train.size else data[:, 0].astype('int')
    dataset.set_data(X_train, y_train, X_test, y_test)
    return dataset


def alg_load(algorithm, dataset):
    module = 'algorithm.' + algorithm.name
    module = __import__(module)
    module = getattr(module, algorithm.name)
    model = getattr(module, "get_model_v" + str(dataset.origin_model)
                    )(dataset.input_shape, dataset.nb_classes)
    algorithm.set_model(model)
    return algorithm
